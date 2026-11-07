//! Speech-to-text with whisper.cpp (via `whisper-rs`), CPU only.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, get_lang_str,
};

/// Result of transcribing one audio file.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    /// Full text, segments joined and trimmed.
    pub text: String,
    /// Detected language code (e.g. "it", "en"), or "unknown".
    pub language: String,
}

/// Loaded Whisper model, reused for every file.
pub struct Transcriber {
    ctx: WhisperContext,
    threads: i32,
}

impl Transcriber {
    /// Loads the ggml model at `model_path`. Fails fast if it is missing.
    pub fn new(model_path: &Path) -> Result<Self> {
        if !model_path.is_file() {
            return Err(anyhow!(
                "model not found at {} (run scripts/download-model.sh)",
                model_path.display()
            ));
        }
        let path = model_path
            .to_str()
            .ok_or_else(|| anyhow!("model path is not valid UTF-8"))?;
        let ctx = WhisperContext::new_with_params(path, WhisperContextParameters::default())
            .with_context(|| format!("cannot load model {}", model_path.display()))?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        Ok(Self { ctx, threads })
    }

    /// Transcribes 16 kHz mono f32 samples, auto-detecting the language.
    pub fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
        let mut state = self.ctx.create_state().context("cannot create state")?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // "auto" = detect the language (do NOT use set_detect_language: in
        // whisper.cpp that mode only detects and skips the transcription).
        params.set_language(Some("auto"));
        params.set_n_threads(self.threads);
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state
            .full(params, samples)
            .context("transcription failed")?;

        let parts: Vec<String> = state
            .as_iter()
            .map(|s| s.to_str_lossy().map(|t| t.into_owned()))
            .collect::<std::result::Result<_, _>>()
            .context("cannot read segment text")?;
        let language = get_lang_str(state.full_lang_id_from_state())
            .unwrap_or("unknown")
            .to_string();

        Ok(Transcript {
            text: join_segments(&parts),
            language,
        })
    }
}

/// Joins segment texts into one trimmed string, dropping empty segments.
fn join_segments(parts: &[String]) -> String {
    parts
        .iter()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_trims_and_drops_empty() {
        let parts = vec![
            " Hello ".to_string(),
            "".into(),
            "  ".into(),
            "world\n".into(),
        ];
        assert_eq!(join_segments(&parts), "Hello world");
        assert_eq!(join_segments(&[]), "");
    }

    #[test]
    fn missing_model_is_a_clear_error() {
        let err = Transcriber::new(Path::new("/definitely/not/here.bin"))
            .err()
            .expect("must fail");
        assert!(err.to_string().contains("model not found"));
    }

    /// Runs only when `TA_TEST_MODEL` points at a ggml model; silence in,
    /// no crash and a language out.
    #[test]
    fn transcribes_silence_with_real_model() {
        let Ok(model) = std::env::var("TA_TEST_MODEL") else {
            eprintln!("TA_TEST_MODEL not set, skipping");
            return;
        };
        let t = Transcriber::new(Path::new(&model)).unwrap();
        let out = t.transcribe(&vec![0.0f32; 16_000 * 2]).unwrap();
        assert!(!out.language.is_empty());
    }
}
