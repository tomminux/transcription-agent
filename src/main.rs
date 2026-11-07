mod config;
mod decoder;
mod transcriber;
mod watcher;

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

    while let Some(path) = queue.next() {
        if failures.should_skip(&path) {
            continue;
        }
        if !watcher::wait_until_stable(&path, Duration::from_secs(1), 2) {
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
        let transcript = match transcriber.transcribe(&audio.samples) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(file = %path.display(), "transcription failed: {e:#}");
                failures.record(&path);
                continue;
            }
        };
        // TODO: write the transcription, move the audio file.
        // Never log transcript text: it is private.
        tracing::info!(
            file = %path.display(),
            duration_secs = audio.duration_secs(),
            language = %transcript.language,
            chars = transcript.text.chars().count(),
            "transcribed (writer/mover not implemented yet)"
        );
    }
    Ok(())
}
