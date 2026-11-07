use std::path::PathBuf;

use clap::Parser;

/// Command-line configuration for the transcription agent.
#[derive(Debug, Clone, Parser)]
#[command(
    version,
    about = "Watches a folder and transcribes new audio files locally"
)]
pub struct Config {
    /// Folder watched for new audio files.
    #[arg(long, default_value = "new-audio-files")]
    pub input_dir: PathBuf,

    /// Folder where Markdown transcriptions are written.
    #[arg(long, default_value = "transcriptions")]
    pub output_dir: PathBuf,

    /// Folder where audio files are moved after transcription.
    #[arg(long, default_value = "transcribed-audio-files")]
    pub done_dir: PathBuf,

    /// Path to the whisper.cpp ggml model file.
    #[arg(long, default_value = "models/ggml-base.bin")]
    pub model: PathBuf,
}

impl Config {
    /// Creates the three working folders if they do not exist yet.
    pub fn ensure_dirs(&self) -> anyhow::Result<()> {
        use anyhow::Context;
        for dir in [&self.input_dir, &self.output_dir, &self.done_dir] {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("cannot create directory {}", dir.display()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_project_layout() {
        let c = Config::parse_from(["transcription-agent"]);
        assert_eq!(c.input_dir, PathBuf::from("new-audio-files"));
        assert_eq!(c.output_dir, PathBuf::from("transcriptions"));
        assert_eq!(c.done_dir, PathBuf::from("transcribed-audio-files"));
    }

    #[test]
    fn ensure_dirs_creates_folders() {
        let base = std::env::temp_dir().join(format!("ta-test-{}", std::process::id()));
        let c = Config::parse_from([
            "x",
            "--input-dir",
            base.join("a").to_str().unwrap(),
            "--output-dir",
            base.join("b").to_str().unwrap(),
            "--done-dir",
            base.join("c").to_str().unwrap(),
        ]);
        c.ensure_dirs().unwrap();
        assert!(base.join("a").is_dir() && base.join("b").is_dir() && base.join("c").is_dir());
        std::fs::remove_dir_all(base).unwrap();
    }
}
