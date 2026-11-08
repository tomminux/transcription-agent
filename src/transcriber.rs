//! Speech-to-text with whisper.cpp (via `whisper-rs`), CPU only.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, get_lang_str,
};

/// One timed piece of speech, as segmented by Whisper.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Start time in centiseconds (Whisper's unit).
    pub start_cs: i64,
    /// End time in centiseconds.
    pub end_cs: i64,
    /// Trimmed, non-empty text.
    pub text: String,
}

/// Result of transcribing one audio file.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    /// Timed segments in order (empty segments already dropped).
    pub segments: Vec<Segment>,
    /// Detected language code (e.g. "it", "en"), or "unknown".
    pub language: String,
}

impl Transcript {
    /// The whole text without timestamps.
    pub fn text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
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
    /// Returns `Ok(None)` if `abort` was set while running (shutdown).
    pub fn transcribe(
        &self,
        samples: &[f32],
        abort: &Arc<AtomicBool>,
    ) -> Result<Option<Transcript>> {
        let mut state = self.ctx.create_state().context("cannot create state")?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // "auto" = detect the language (do NOT use set_detect_language: in
        // whisper.cpp that mode only detects and skips the transcription).
        params.set_language(Some("auto"));
        params.set_n_threads(self.threads);
        // NOTE: `set_abort_callback_safe` in whisper-rs 0.16.0 is broken (its
        // trampoline misreads the user-data pointer, making whisper fail with
        // "failed to encode"), so use the raw API with our own callback.
        // SAFETY: `abort` outlives `state.full(..)` below, which is the only
        // place the pointer is used, and AtomicBool is safe to share.
        unsafe {
            params.set_abort_callback(Some(abort_trampoline));
            params.set_abort_callback_user_data(Arc::as_ptr(abort) as *mut std::ffi::c_void);
        }
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let result = state.full(params, samples);
        if abort.load(Ordering::Relaxed) {
            return Ok(None);
        }
        result.context("transcription failed")?;

        let mut segments = Vec::new();
        for seg in state.as_iter() {
            let text = seg.to_str_lossy().context("cannot read segment text")?;
            let text = text.trim();
            if !text.is_empty() {
                segments.push(Segment {
                    start_cs: seg.start_timestamp(),
                    end_cs: seg.end_timestamp(),
                    text: text.to_string(),
                });
            }
        }
        let language = get_lang_str(state.full_lang_id_from_state())
            .unwrap_or("unknown")
            .to_string();

        Ok(Some(Transcript { segments, language }))
    }
}

/// C callback polled by whisper.cpp; `user_data` is a `*const AtomicBool`.
unsafe extern "C" fn abort_trampoline(user_data: *mut std::ffi::c_void) -> bool {
    // SAFETY: set by `transcribe` to a live `AtomicBool`.
    let flag = unsafe { &*(user_data as *const AtomicBool) };
    flag.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_joins_segments() {
        let seg = |t: &str| Segment {
            start_cs: 0,
            end_cs: 100,
            text: t.into(),
        };
        let t = Transcript {
            segments: vec![seg("Hello"), seg("world")],
            language: "en".into(),
        };
        assert_eq!(t.text(), "Hello world");
        let empty = Transcript {
            segments: vec![],
            language: "en".into(),
        };
        assert_eq!(empty.text(), "");
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
        let no = Arc::new(AtomicBool::new(false));
        let out = t
            .transcribe(&vec![0.0f32; 16_000 * 2], &no)
            .unwrap()
            .unwrap();
        assert!(!out.language.is_empty());

        // Already-aborted run yields None, not an error.
        let yes = Arc::new(AtomicBool::new(true));
        assert!(
            t.transcribe(&vec![0.0f32; 16_000 * 2], &yes)
                .unwrap()
                .is_none()
        );
    }
}
