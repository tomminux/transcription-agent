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
    let (_watcher, mut queue) = watcher::watch(
        &config.input_dir,
        Duration::from_secs(config.scan_interval_secs.max(1)),
    )?;
    tracing::info!(dir = %config.input_dir.display(), "watching for new audio files");

    let mut failures = watcher::FailureTracker::default();

    let summarizer = summarizer::Summarizer::from_config(&config)?;
    if let Some(s) = &summarizer {
        tracing::info!(llm = %s.describe(), "meeting reports enabled");
        // Backfill: transcriptions without a report (older files, or an LLM
        // failure last time). A failure is logged and not retried until restart.
        for md in summarizer::missing_reports(&config.output_dir) {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            report(s, &md);
        }
    }

    let model_name = config
        .model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    while let Some(path) = queue.next(&shutdown) {
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
        let started = Instant::now();
        let audio = match decoder::decode_file(&path) {
            Ok(a) => a,
            Err(e) => {
                tracing::error!(file = %path.display(), "decoding failed: {e:#}");
                failures.record(&path);
                continue;
            }
        };
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
                if let Some(s) = &summarizer {
                    report(s, &md);
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
fn report(summarizer: &summarizer::Summarizer, transcription: &std::path::Path) {
    let started = Instant::now();
    match summarizer::write_report(summarizer, transcription) {
        Ok(path) => tracing::info!(
            report = %path.display(),
            elapsed_secs = started.elapsed().as_secs_f64().round(),
            "meeting report written"
        ),
        Err(e) => tracing::error!(
            transcription = %transcription.display(),
            "meeting report failed: {e:#}"
        ),
    }
}
