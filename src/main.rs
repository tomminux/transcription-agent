mod config;
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
    let (_watcher, mut queue) = watcher::watch(&config.input_dir)?;
    tracing::info!(dir = %config.input_dir.display(), "watching for new audio files");

    while let Some(path) = queue.next() {
        if !watcher::wait_until_stable(&path, Duration::from_secs(1), 2) {
            tracing::debug!(file = %path.display(), "file disappeared, skipping");
            continue;
        }
        // TODO: decode, transcribe, write the transcription, move the audio file.
        tracing::info!(file = %path.display(), "new audio file ready (processing not implemented yet)");
    }
    Ok(())
}
