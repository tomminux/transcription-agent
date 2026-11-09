//! Speech-to-text with whisper.cpp (via `whisper-rs`), CPU only.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    get_lang_id, get_lang_str,
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

/// How the spoken language is chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum LanguageMode {
    /// Whisper detects it from the first 30 seconds (can be wrong, e.g. after
    /// a silent or noisy intro).
    Auto,
    /// Always this language, e.g. `it`.
    Fixed(String),
    /// Detect the language, but only among these ones, sampling several points
    /// of the recording instead of just the start.
    Candidates(Vec<String>),
}

impl LanguageMode {
    /// Parses `auto`, a code (`it`) or a comma-separated list (`it,en`).
    pub fn parse(s: &str) -> Result<Self> {
        let mut codes: Vec<String> = Vec::new();
        for c in s.split(',') {
            let c = c.trim().to_ascii_lowercase();
            if !c.is_empty() && !codes.contains(&c) {
                codes.push(c);
            }
        }
        if codes.is_empty() || codes == ["auto"] {
            return Ok(Self::Auto);
        }
        for c in &codes {
            if c == "auto" || get_lang_id(c).is_none() {
                bail!(
                    "unknown language code {c:?}: use `auto`, a Whisper code like `it`, \
                     or a comma-separated list like `it,en`"
                );
            }
        }
        Ok(if codes.len() == 1 {
            Self::Fixed(codes.remove(0))
        } else {
            Self::Candidates(codes)
        })
    }
}

/// How many points of the audio are sampled for language detection.
const DETECT_WINDOWS: usize = 5;
/// Whisper looks at 30 seconds of audio per detection.
const DETECT_WINDOW_MS: usize = 30_000;

/// Start offsets (ms) of the detection windows, evenly spread over the audio.
fn detection_offsets(duration_ms: usize, windows: usize) -> Vec<usize> {
    if duration_ms <= DETECT_WINDOW_MS || windows <= 1 {
        return vec![0];
    }
    let last = duration_ms - DETECT_WINDOW_MS;
    (0..windows).map(|i| last * i / (windows - 1)).collect()
}

/// Index of the highest score (the first one wins a tie).
fn best_index(scores: &[f32]) -> usize {
    let mut best = 0;
    for (i, s) in scores.iter().enumerate() {
        if *s > scores[best] {
            best = i;
        }
    }
    best
}

/// Loaded Whisper model, reused for every file.
pub struct Transcriber {
    ctx: WhisperContext,
    threads: i32,
    language: LanguageMode,
}

impl Transcriber {
    /// Loads the ggml model at `model_path`. Fails fast if it is missing.
    pub fn new(model_path: &Path, language: LanguageMode) -> Result<Self> {
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
        Ok(Self {
            ctx,
            threads,
            language,
        })
    }

    /// Picks the most likely language among `candidates` by running Whisper's
    /// language detection on several 30 s windows spread over the audio and
    /// summing, per candidate, the probability it gets. Windows without speech
    /// give low probability to every candidate, so they barely count.
    /// Returns `Ok(None)` if `abort` was set meanwhile.
    fn detect_among(
        &self,
        state: &mut WhisperState,
        samples: &[f32],
        candidates: &[String],
        abort: &AtomicBool,
    ) -> Result<Option<String>> {
        let ids: Vec<usize> = candidates
            .iter()
            .map(|c| {
                get_lang_id(c)
                    .map(|id| id as usize)
                    .ok_or_else(|| anyhow!("unknown language {c}"))
            })
            .collect::<Result<_>>()?;

        let threads = self.threads.max(1) as usize;
        state
            .pcm_to_mel(samples, threads)
            .context("cannot compute spectrogram for language detection")?;

        let duration_ms = samples.len() / 16; // 16 samples per ms at 16 kHz
        let offsets = detection_offsets(duration_ms, DETECT_WINDOWS);
        let mut scores = vec![0f32; candidates.len()];
        for offset in &offsets {
            if abort.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let (_best, probs) = state
                .lang_detect(*offset, threads)
                .context("language detection failed")?;
            for (score, id) in scores.iter_mut().zip(&ids) {
                *score += probs.get(*id).copied().unwrap_or(0.0);
            }
        }

        let chosen = &candidates[best_index(&scores)];
        let summary: Vec<String> = candidates
            .iter()
            .zip(&scores)
            .map(|(c, s)| format!("{c}={:.2}", s / offsets.len() as f32))
            .collect();
        tracing::info!(
            language = %chosen,
            "language detection among candidates over {} windows (mean probability): {}",
            offsets.len(),
            summary.join(" ")
        );
        Ok(Some(chosen.clone()))
    }

    /// Transcribes 16 kHz mono f32 samples; the language is chosen according
    /// to the configured [`LanguageMode`].
    /// Returns `Ok(None)` if `abort` was set while running (shutdown).
    pub fn transcribe(
        &self,
        samples: &[f32],
        abort: &Arc<AtomicBool>,
    ) -> Result<Option<Transcript>> {
        let mut state = self.ctx.create_state().context("cannot create state")?;

        let language_code: Option<String> = match &self.language {
            LanguageMode::Auto => None,
            LanguageMode::Fixed(code) => Some(code.clone()),
            LanguageMode::Candidates(codes) => {
                match self.detect_among(&mut state, samples, codes, abort)? {
                    Some(code) => Some(code),
                    None => return Ok(None),
                }
            }
        };

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // "auto" = whisper detects the language (do NOT use set_detect_language:
        // in whisper.cpp that mode only detects and skips the transcription).
        params.set_language(Some(language_code.as_deref().unwrap_or("auto")));
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
        let language = match language_code {
            Some(code) => code,
            None => get_lang_str(state.full_lang_id_from_state())
                .unwrap_or("unknown")
                .to_string(),
        };

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
    fn language_mode_parsing() {
        use LanguageMode::*;
        assert_eq!(LanguageMode::parse("auto").unwrap(), Auto);
        assert_eq!(LanguageMode::parse("").unwrap(), Auto);
        assert_eq!(LanguageMode::parse(" AUTO ").unwrap(), Auto);
        assert_eq!(LanguageMode::parse("it").unwrap(), Fixed("it".into()));
        assert_eq!(
            LanguageMode::parse("it,en").unwrap(),
            Candidates(vec!["it".into(), "en".into()])
        );
        assert_eq!(
            LanguageMode::parse(" IT , en,it ").unwrap(),
            Candidates(vec!["it".into(), "en".into()])
        );
        assert_eq!(LanguageMode::parse("it,it").unwrap(), Fixed("it".into()));
        assert!(LanguageMode::parse("xx").is_err());
        assert!(LanguageMode::parse("auto,it").is_err());
        assert!(LanguageMode::parse("it,klingon").is_err());
    }

    #[test]
    fn detection_windows_are_spread_over_the_audio() {
        assert_eq!(detection_offsets(10_000, 5), [0]);
        assert_eq!(detection_offsets(30_000, 5), [0]);
        assert_eq!(detection_offsets(120_000, 1), [0]);
        assert_eq!(
            detection_offsets(120_000, 5),
            [0, 22_500, 45_000, 67_500, 90_000]
        );
        // The last window ends exactly at the end of the audio.
        assert_eq!(*detection_offsets(1_234_789, 5).last().unwrap(), 1_204_789);
    }

    #[test]
    fn best_index_picks_max_first_on_ties() {
        assert_eq!(best_index(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(best_index(&[0.5, 0.5]), 0);
        assert_eq!(best_index(&[0.0]), 0);
    }

    #[test]
    fn missing_model_is_a_clear_error() {
        let err = Transcriber::new(Path::new("/definitely/not/here.bin"), LanguageMode::Auto)
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
        let t = Transcriber::new(Path::new(&model), LanguageMode::Auto).unwrap();
        let no = Arc::new(AtomicBool::new(false));
        let out = t
            .transcribe(&vec![0.0f32; 16_000 * 2], &no)
            .unwrap()
            .unwrap();
        assert!(!out.language.is_empty());

        // Candidate mode picks one of the candidates, even on silence.
        let t = Transcriber::new(Path::new(&model), LanguageMode::parse("it,en").unwrap()).unwrap();
        let out = t
            .transcribe(&vec![0.0f32; 16_000 * 40], &no)
            .unwrap()
            .unwrap();
        assert!(out.language == "it" || out.language == "en");

        // Already-aborted run yields None, not an error.
        let yes = Arc::new(AtomicBool::new(true));
        assert!(
            t.transcribe(&vec![0.0f32; 16_000 * 2], &yes)
                .unwrap()
                .is_none()
        );
    }
}
