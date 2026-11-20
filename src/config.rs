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
    #[arg(long, env = "TA_INPUT_DIR", default_value = "new-audio-files")]
    pub input_dir: PathBuf,

    /// Folder where Markdown transcriptions are written.
    #[arg(long, env = "TA_OUTPUT_DIR", default_value = "transcriptions")]
    pub output_dir: PathBuf,

    /// Folder where audio files are moved after transcription.
    #[arg(long, env = "TA_DONE_DIR", default_value = "transcribed-audio-files")]
    pub done_dir: PathBuf,

    /// Path to the whisper.cpp ggml model file.
    #[arg(long, env = "TA_MODEL", default_value = "models/ggml-base.bin")]
    pub model: PathBuf,

    /// Spoken language: `auto` (Whisper decides from the first 30 s, can be
    /// wrong), a code like `it`, or a list like `it,en` (detect only among
    /// these, sampling several points of the recording).
    #[arg(long, env = "TA_LANGUAGE", default_value = "auto")]
    pub language: String,

    /// Seconds between rescans of the input folder. Needed because file
    /// events from a macOS host often do not reach a container.
    #[arg(long, env = "TA_SCAN_INTERVAL_SECS", default_value_t = 5)]
    pub scan_interval_secs: u64,

    /// LLM used to turn each transcription into a meeting report: `none`
    /// (default), `openai` (any OpenAI-compatible server, e.g. a private one),
    /// `ollama` (local model) or `anthropic` (Claude API: the transcript is
    /// sent to Anthropic).
    #[arg(long, env = "TA_LLM", default_value = "none")]
    pub llm: String,

    /// What to do with files placed directly in the input folder: `report`
    /// (transcription + AI report) or `none` (transcription only). Files in
    /// `with-report/` or `transcribe-only/` ignore this.
    #[arg(long, env = "TA_LLM_DEFAULT", default_value = "none")]
    pub llm_default: String,

    /// Model name (anthropic default: claude-opus-5-5; ollama: required).
    #[arg(long, env = "TA_LLM_MODEL")]
    pub llm_model: Option<String>,

    /// Base URL of the LLM server (ollama default:
    /// http://host.containers.internal:11434).
    #[arg(long, env = "TA_LLM_URL")]
    pub llm_url: Option<String>,

    /// API key for the `openai`-compatible private server (sent as a Bearer token).
    #[arg(long, env = "TA_LLM_API_KEY", hide_env_values = true)]
    pub llm_api_key: Option<String>,

    /// PEM file with the CA certificate of an intranet HTTPS server.
    /// A String, not a PathBuf: compose passes an empty value when unset, and
    /// clap rejects an empty path ("a value is required").
    #[arg(long, env = "TA_LLM_CA_CERT")]
    pub llm_ca_cert: Option<String>,

    /// API key for `anthropic`.
    #[arg(long, env = "ANTHROPIC_API_KEY", hide_env_values = true)]
    pub anthropic_api_key: Option<String>,

    /// Context window requested from Ollama (it truncates silently beyond it).
    #[arg(long, env = "TA_LLM_NUM_CTX", default_value_t = 32768)]
    pub llm_num_ctx: u32,

    /// Seconds between checks for transcriptions whose front matter says
    /// `report: requested` but that have no report yet (0 = only at startup).
    #[arg(long, env = "TA_REPORT_SCAN_INTERVAL_SECS", default_value_t = 30)]
    pub report_scan_interval_secs: u64,

    /// Timeout for one LLM request, in seconds.
    #[arg(long, env = "TA_LLM_TIMEOUT_SECS", default_value_t = 900)]
    pub llm_timeout_secs: u64,

    /// Language of the report, e.g. `Italian`; empty = same as the transcript.
    #[arg(long, env = "TA_REPORT_LANGUAGE", default_value = "")]
    pub report_language: String,
}

impl Config {
    /// Creates the three working folders if they do not exist yet.
    pub fn ensure_dirs(&self) -> anyhow::Result<()> {
        use anyhow::Context;
        let with_report = self.input_dir.join(crate::output::WITH_REPORT_DIR);
        let only = self.input_dir.join(crate::output::TRANSCRIBE_ONLY_DIR);
        for dir in [
            &self.input_dir,
            &with_report,
            &only,
            &self.output_dir,
            &self.done_dir,
        ] {
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
        assert_eq!(c.scan_interval_secs, 5);
        assert_eq!(c.language, "auto");
        assert_eq!(c.llm, "none");
    }

    #[test]
    fn empty_ca_cert_means_unset() {
        // `--llm-ca-cert ""` is what compose produces for an unset variable.
        let c = Config::parse_from(["x", "--llm-ca-cert", ""]);
        assert_eq!(c.llm_ca_cert.as_deref(), Some(""));
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
