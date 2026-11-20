mod config;
mod decoder;
mod output;
mod summarizer;
mod transcriber;
mod watcher;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::Parser;
use config::Config;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = Config::parse();
    config.ensure_dirs()?;

    // First SIGTERM/SIGINT sets `shutdown` (finish cleanly); a second one exits
    // immediately. The conditional-shutdown hook must be registered first.
    let shutdown = Arc::new(AtomicBool::new(false));
    for sig in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register_conditional_shutdown(sig, 1, Arc::clone(&shutdown))?;
        signal_hook::flag::register(sig, Arc::clone(&shutdown))?;
    }

    whisper_rs::install_logging_hooks();
    let language = transcriber::LanguageMode::parse(&config.language)?;
    let transcriber = transcriber::Transcriber::new(&config.model, language)?;
    tracing::info!(
        model = %config.model.display(),
        language = %config.language,
        "model loaded"
    );

    // Keep `_watcher` alive: dropping it stops the events.
    let default_mode = output::ReportMode::parse_default(&config.llm_default)?;
    let watched = [
        config.input_dir.clone(),
        config.input_dir.join(output::WITH_REPORT_DIR),
        config.input_dir.join(output::TRANSCRIBE_ONLY_DIR),
    ];
    let (_watcher, mut queue) = watcher::watch(
        &watched,
        Duration::from_secs(config.scan_interval_secs.max(1)),
    )?;
    tracing::info!(dir = %config.input_dir.display(), "watching for new audio files");

    let mut failures = watcher::FailureTracker::default();

    let summarizer = summarizer::Summarizer::from_config(&config)?;
    if let Some(s) = &summarizer {
        tracing::info!(llm = %s.describe(), "meeting reports enabled");
    }
    // Transcriptions that asked for a report and have none (also covers files
    // edited to `report: requested`). Runs at startup and then periodically.
    let mut report_failures = watcher::FailureTracker::default();
    let report_every = Duration::from_secs(config.report_scan_interval_secs);
    if let Some(s) = &summarizer {
        backfill_reports(s, &config.output_dir, &mut report_failures, &shutdown);
    }
    let mut last_report_scan = Instant::now();

    let model_name = config
        .model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    loop {
        let path = match queue.next_within(&shutdown, Duration::from_secs(1)) {
            watcher::Wait::File(p) => p,
            watcher::Wait::Idle => {
                if let Some(s) = &summarizer
                    && !report_every.is_zero()
                    && last_report_scan.elapsed() >= report_every
                {
                    backfill_reports(s, &config.output_dir, &mut report_failures, &shutdown);
                    last_report_scan = Instant::now();
                }
                continue;
            }
            watcher::Wait::Stop => break,
        };
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if failures.should_skip(&path) {
            continue;
        }
        if !watcher::wait_until_stable(&path, Duration::from_secs(1), 2, &shutdown) {
            tracing::debug!(file = %path.display(), "file disappeared, skipping");
            continue;
        }
        let mode = output::ReportMode::for_path(&path, &config.input_dir, default_mode);
        let started = Instant::now();
        let audio = match decoder::decode_file(&path) {
            Ok(a) => a,
            Err(e) => {
                tracing::error!(file = %path.display(), "decoding failed: {e:#}");
                failures.record(&path);
                continue;
            }
        };
        tracing::info!(
            file = %path.display(),
            duration_secs = audio.duration_secs().round(),
            "audio decoded, transcribing (long recordings take a while, no progress is logged)"
        );
        let transcript = match transcriber.transcribe(&audio.samples, &shutdown) {
            Ok(Some(t)) => t,
            Ok(None) => {
                tracing::info!(file = %path.display(), "transcription interrupted, audio left in place");
                break;
            }
            Err(e) => {
                tracing::error!(file = %path.display(), "transcription failed: {e:#}");
                failures.record(&path);
                continue;
            }
        };
        let meta = output::Meta {
            source_name: &file_name,
            recorded_at: std::fs::metadata(&path).and_then(|m| m.modified()).ok(),
            duration_secs: audio.duration_secs(),
            model: &model_name,
            report: mode,
        };
        match output::store_result(
            &path,
            &config.output_dir,
            &config.done_dir,
            &transcript,
            &meta,
        ) {
            // Never log transcript text: it is private.
            Ok(md) => {
                tracing::info!(
                file = %path.display(),
                transcription = %md.display(),
                duration_secs = audio.duration_secs(),
                // Processing time and speed (x real time): handy to compare models.
                elapsed_secs = started.elapsed().as_secs_f64().round(),
                speed = %format!("{:.1}x", audio.duration_secs() / started.elapsed().as_secs_f64().max(0.001)),
                language = %transcript.language,
                segments = transcript.segments.len(),
                chars = transcript.text().chars().count(),
                "transcribed and moved"
                );
                if mode == output::ReportMode::Requested {
                    match &summarizer {
                        Some(s) => {
                            report(s, &md);
                        }
                        None => tracing::warn!(
                            transcription = %md.display(),
                            "report requested but TA_LLM is none; it will be written at a later start with an LLM configured"
                        ),
                    }
                }
            }
            Err(e) => {
                tracing::error!(file = %path.display(), "saving result failed: {e:#}");
                failures.record(&path);
            }
        }
    }
    if shutdown.load(Ordering::Relaxed) {
        tracing::info!("shutdown requested, exiting cleanly");
    }
    Ok(())
}

/// Writes the report for one transcription; failures are logged only (the
/// transcription and the audio are already safe).
/// Returns whether the report was written.
fn report(summarizer: &summarizer::Summarizer, transcription: &std::path::Path) -> bool {
    let started = Instant::now();
    match summarizer::write_report(summarizer, transcription) {
        Ok(path) => {
            tracing::info!(
                report = %path.display(),
                elapsed_secs = started.elapsed().as_secs_f64().round(),
                "meeting report written"
            );
            true
        }
        Err(e) => {
            tracing::error!(
                transcription = %transcription.display(),
                "meeting report failed: {e:#}"
            );
            false
        }
    }
}

/// Writes the reports that were asked for (`report: requested`) and are
/// missing. A failed one is retried only after its transcription file changes
/// (same rule as failed audio), so a broken LLM does not cause a retry storm.
fn backfill_reports(
    summarizer: &summarizer::Summarizer,
    output_dir: &std::path::Path,
    failures: &mut watcher::FailureTracker,
    shutdown: &AtomicBool,
) {
    for md in summarizer::missing_reports(output_dir) {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        if failures.should_skip(&md) {
            continue;
        }
        if !report(summarizer, &md) {
            failures.record(&md);
        }
    }
}
