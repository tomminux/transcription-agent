mod config;
mod decoder;
mod output;
mod transcriber;
mod watcher;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

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
    let transcriber = transcriber::Transcriber::new(&config.model)?;
    tracing::info!(model = %config.model.display(), "model loaded");

    // Keep `_watcher` alive: dropping it stops the events.
    let (_watcher, mut queue) = watcher::watch(
        &config.input_dir,
        Duration::from_secs(config.scan_interval_secs.max(1)),
    )?;
    tracing::info!(dir = %config.input_dir.display(), "watching for new audio files");

    let mut failures = watcher::FailureTracker::default();

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
            Ok(md) => tracing::info!(
                file = %path.display(),
                transcription = %md.display(),
                duration_secs = audio.duration_secs(),
                language = %transcript.language,
                chars = transcript.text.chars().count(),
                "transcribed and moved"
            ),
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
