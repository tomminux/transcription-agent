mod config;
mod decoder;
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
        // TODO: transcribe, write the transcription, move the audio file.
        tracing::info!(
            file = %path.display(),
            duration_secs = audio.duration_secs(),
            "decoded (transcription not implemented yet)"
        );
    }
    Ok(())
}
