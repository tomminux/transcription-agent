//! Writing the Markdown transcription and moving the processed audio file.
//!
//! Order matters (never lose audio): the transcription is written first, via
//! a temporary file + rename; only then is the audio moved. If the move
//! fails, the transcription just written is removed and the audio stays put.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, SecondsFormat, Utc};

use crate::transcriber::{Segment, Transcript};

/// Metadata written to the front matter of the Markdown file.
/// Subfolders of the input folder that choose what happens to a file.
pub const WITH_REPORT_DIR: &str = "with-report";
pub const TRANSCRIBE_ONLY_DIR: &str = "transcribe-only";

/// Whether an AI meeting report is wanted for a recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportMode {
    Requested,
    Skipped,
}

impl ReportMode {
    /// Value written to the front matter (`report: ...`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Requested => "requested",
            Self::Skipped => "skipped",
        }
    }

    /// Parses `TA_LLM_DEFAULT`: `report` or `none`.
    pub fn parse_default(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "none" | "transcribe-only" | "off" => Ok(Self::Skipped),
            "report" | "with-report" | "on" => Ok(Self::Requested),
            other => Err(anyhow!(
                "unknown TA_LLM_DEFAULT {other:?} (use report or none)"
            )),
        }
    }

    /// Mode for an audio file: its folder under `input_dir` decides, files
    /// directly in `input_dir` get `default`.
    pub fn for_path(path: &Path, input_dir: &Path, default: Self) -> Self {
        match path
            .parent()
            .and_then(|p| p.strip_prefix(input_dir).ok())
            .and_then(|rel| rel.to_str())
        {
            Some(WITH_REPORT_DIR) => Self::Requested,
            Some(TRANSCRIBE_ONLY_DIR) => Self::Skipped,
            _ => default,
        }
    }
}

pub struct Meta<'a> {
    /// Original audio file name.
    pub source_name: &'a str,
    /// Audio file modification time (≈ recording time), if known.
    pub recorded_at: Option<SystemTime>,
    pub duration_secs: f64,
    /// Model file name, e.g. `ggml-base.bin`.
    pub model: &'a str,
    /// Whether the user asked for an AI report (decides the startup backfill).
    pub report: ReportMode,
}

/// Quotes a string as a YAML double-quoted scalar (safe for any content).
pub(crate) fn yaml_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn rfc3339(t: SystemTime) -> String {
    DateTime::<Utc>::from(t).to_rfc3339_opts(SecondsFormat::Secs, true)
}

/// Formats centiseconds as `HH:MM:SS` (hours can exceed 99 for very long audio).
fn format_timestamp(cs: i64) -> String {
    let total = (cs.max(0) / 100) as u64;
    format!(
        "{:02}:{:02}:{:02}",
        total / 3600,
        (total / 60) % 60,
        total % 60
    )
}

/// One body line: `[HH:MM:SS - HH:MM:SS] text`.
fn format_segment(seg: &Segment) -> String {
    format!(
        "[{} - {}] {}",
        format_timestamp(seg.start_cs),
        format_timestamp(seg.end_cs),
        seg.text
    )
}

/// Renders the Markdown document: YAML front matter followed by one
/// timestamped line per segment.
pub fn render_markdown(meta: &Meta, transcript: &Transcript, transcribed_at: SystemTime) -> String {
    let mut s = String::from("---\n");
    s.push_str(&format!("source: {}\n", yaml_quote(meta.source_name)));
    if let Some(t) = meta.recorded_at {
        s.push_str(&format!("recorded_at: {}\n", rfc3339(t)));
    }
    s.push_str(&format!("transcribed_at: {}\n", rfc3339(transcribed_at)));
    s.push_str(&format!("duration_seconds: {:.1}\n", meta.duration_secs));
    s.push_str(&format!("language: {}\n", yaml_quote(&transcript.language)));
    s.push_str(&format!("model: {}\n", yaml_quote(meta.model)));
    s.push_str(&format!("report: {}\n", meta.report.as_str()));
    s.push_str("---\n\n");
    for seg in &transcript.segments {
        s.push_str(&format_segment(seg));
        s.push('\n');
    }
    s
}

/// Picks a file stem that is free in both folders: `name`, `name-2`, `name-3`...
/// so the transcription and its audio file always share the same stem.
fn unique_stem(output_dir: &Path, done_dir: &Path, stem: &str, audio_ext: &str) -> String {
    let free = |s: &str| {
        !output_dir.join(format!("{s}.md")).exists()
            && !done_dir.join(format!("{s}.{audio_ext}")).exists()
    };
    if free(stem) {
        return stem.to_string();
    }
    (2..)
        .map(|n| format!("{stem}-{n}"))
        .find(|s| free(s))
        .expect("unbounded range")
}

/// Writes `data` to `dest` through a hidden temp file in the same folder,
/// fsyncs it, then renames it into place.
pub(crate) fn write_atomic(dest: &Path, data: &[u8]) -> Result<()> {
    let tmp = tmp_path(dest);
    let result = (|| -> io::Result<()> {
        let mut f = File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
        fs::rename(&tmp, dest)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result.with_context(|| format!("cannot write {}", dest.display()))
}

fn tmp_path(dest: &Path) -> PathBuf {
    let name = dest.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    dest.with_file_name(format!(".{name}.tmp"))
}

/// Moves `src` to `dst`. Tries `rename`; if the folders are on different
/// mounts, copies (temp file + fsync + rename) and only then deletes `src`.
pub fn move_file(src: &Path, dst: &Path) -> Result<()> {
    match fs::rename(src, dst) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::CrossesDevices => copy_then_remove(src, dst),
        Err(e) => Err(e).with_context(|| format!("cannot move {}", src.display())),
    }
}

fn copy_then_remove(src: &Path, dst: &Path) -> Result<()> {
    let tmp = tmp_path(dst);
    let copied = (|| -> io::Result<()> {
        fs::copy(src, &tmp)?;
        File::open(&tmp)?.sync_all()?;
        fs::rename(&tmp, dst)
    })();
    if let Err(e) = copied {
        let _ = fs::remove_file(&tmp);
        return Err(e)
            .with_context(|| format!("cannot copy {} to {}", src.display(), dst.display()));
    }
    fs::remove_file(src).with_context(|| format!("copied but cannot remove {}", src.display()))
}

/// Writes the transcription and moves the audio. Returns the Markdown path.
pub fn store_result(
    audio: &Path,
    output_dir: &Path,
    done_dir: &Path,
    transcript: &Transcript,
    meta: &Meta,
) -> Result<PathBuf> {
    let stem = audio
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("audio file name is not valid UTF-8"))?;
    let ext = audio.extension().and_then(|e| e.to_str()).unwrap_or("m4a");

    let stem = unique_stem(output_dir, done_dir, stem, ext);
    let md_path = output_dir.join(format!("{stem}.md"));
    let audio_dest = done_dir.join(format!("{stem}.{ext}"));

    let doc = render_markdown(meta, transcript, SystemTime::now());
    write_atomic(&md_path, doc.as_bytes())?;

    if let Err(e) = move_file(audio, &audio_dest) {
        // Keep the audio in place and don't leave an orphan transcription.
        let _ = fs::remove_file(&md_path);
        return Err(e);
    }
    Ok(md_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn transcript(text: &str) -> Transcript {
        Transcript {
            segments: vec![Segment {
                start_cs: 0,
                end_cs: 1234,
                text: text.into(),
            }],
            language: "it".into(),
        }
    }

    #[test]
    fn report_mode_from_folder_and_default() {
        let root = Path::new("/in");
        let d = ReportMode::Skipped;
        let m = |p: &str, def| ReportMode::for_path(Path::new(p), root, def);
        assert_eq!(m("/in/a.m4a", d), ReportMode::Skipped);
        assert_eq!(m("/in/a.m4a", ReportMode::Requested), ReportMode::Requested);
        assert_eq!(m("/in/with-report/a.m4a", d), ReportMode::Requested);
        assert_eq!(
            m("/in/transcribe-only/a.m4a", ReportMode::Requested),
            ReportMode::Skipped
        );
        assert_eq!(m("/in/other/a.m4a", d), d);
        assert_eq!(
            ReportMode::parse_default("report").unwrap(),
            ReportMode::Requested
        );
        assert_eq!(
            ReportMode::parse_default("none").unwrap(),
            ReportMode::Skipped
        );
        assert!(ReportMode::parse_default("x").is_err());
    }

    #[test]
    fn front_matter_records_report_choice() {
        let md = render_markdown(&meta("a.m4a"), &transcript("hi"), SystemTime::UNIX_EPOCH);
        assert!(md.contains("\nreport: requested\n---\n"));
    }

    fn meta(name: &str) -> Meta<'_> {
        Meta {
            source_name: name,
            recorded_at: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)),
            duration_secs: 12.34,
            model: "ggml-base.bin",
            report: ReportMode::Requested,
        }
    }

    #[test]
    fn yaml_quote_escapes_specials() {
        assert_eq!(yaml_quote("a: \"b\"\\c"), r#""a: \"b\"\\c""#);
        assert_eq!(yaml_quote("x\ny"), r#""x\ny""#);
        assert_eq!(yaml_quote("\u{1}"), r#""\u0001""#);
        assert_eq!(
            yaml_quote("Nuova registrazione è"),
            "\"Nuova registrazione è\""
        );
    }

    #[test]
    fn markdown_has_front_matter_then_text() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_100);
        let md = render_markdown(&meta("a b.m4a"), &transcript("Ciao mondo"), now);
        assert_eq!(
            md,
            "---\nsource: \"a b.m4a\"\nrecorded_at: 2023-11-14T22:13:20Z\n\
             transcribed_at: 2023-11-14T22:15:00Z\nduration_seconds: 12.3\n\
             language: \"it\"\nmodel: \"ggml-base.bin\"\nreport: requested\n---\n\n[00:00:00 - 00:00:12] Ciao mondo\n"
        );
    }

    #[test]
    fn timestamps_format_hours_minutes_seconds() {
        assert_eq!(format_timestamp(0), "00:00:00");
        assert_eq!(format_timestamp(99), "00:00:00");
        assert_eq!(format_timestamp(6150), "00:01:01");
        assert_eq!(format_timestamp(366_150), "01:01:01");
        assert_eq!(format_timestamp(-5), "00:00:00");
    }

    #[test]
    fn multiple_segments_one_line_each_and_empty_has_no_body() {
        let now = SystemTime::UNIX_EPOCH;
        let mut t = transcript("one");
        t.segments.push(Segment {
            start_cs: 1500,
            end_cs: 366_150,
            text: "two".into(),
        });
        let md = render_markdown(&meta("a.m4a"), &t, now);
        assert!(md.ends_with("---\n\n[00:00:00 - 00:00:12] one\n[00:00:15 - 01:01:01] two\n"));

        t.segments.clear();
        let md = render_markdown(&meta("a.m4a"), &t, now);
        assert!(md.ends_with("---\n\n"));
    }

    #[test]
    fn unique_stem_checks_both_folders() {
        let out = tempfile::tempdir().unwrap();
        let done = tempfile::tempdir().unwrap();
        assert_eq!(unique_stem(out.path(), done.path(), "rec", "m4a"), "rec");
        fs::write(out.path().join("rec.md"), "x").unwrap();
        assert_eq!(unique_stem(out.path(), done.path(), "rec", "m4a"), "rec-2");
        fs::write(done.path().join("rec-2.m4a"), "x").unwrap();
        assert_eq!(unique_stem(out.path(), done.path(), "rec", "m4a"), "rec-3");
    }

    #[test]
    fn store_result_writes_md_and_moves_audio() {
        let base = tempfile::tempdir().unwrap();
        let (inp, out, done) = (
            base.path().join("in"),
            base.path().join("out"),
            base.path().join("done"),
        );
        for d in [&inp, &out, &done] {
            fs::create_dir(d).unwrap();
        }
        let audio = inp.join("rec.m4a");
        fs::write(&audio, b"audio").unwrap();

        let md = store_result(&audio, &out, &done, &transcript("Hi"), &meta("rec.m4a")).unwrap();
        assert_eq!(md, out.join("rec.md"));
        assert!(
            fs::read_to_string(&md)
                .unwrap()
                .ends_with("\n\n[00:00:00 - 00:00:12] Hi\n")
        );
        assert!(!audio.exists());
        assert_eq!(fs::read(done.join("rec.m4a")).unwrap(), b"audio");
        // No temp files left behind.
        assert!(
            fs::read_dir(&out).unwrap().all(|e| !e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp"))
        );
    }

    #[test]
    fn same_name_twice_gets_suffix_and_never_overwrites() {
        let base = tempfile::tempdir().unwrap();
        let (inp, out, done) = (
            base.path().join("in"),
            base.path().join("out"),
            base.path().join("done"),
        );
        for d in [&inp, &out, &done] {
            fs::create_dir(d).unwrap();
        }
        for content in [b"first".as_slice(), b"second"] {
            let audio = inp.join("rec.m4a");
            fs::write(&audio, content).unwrap();
            store_result(&audio, &out, &done, &transcript("t"), &meta("rec.m4a")).unwrap();
        }
        assert_eq!(fs::read(done.join("rec.m4a")).unwrap(), b"first");
        assert_eq!(fs::read(done.join("rec-2.m4a")).unwrap(), b"second");
        assert!(out.join("rec.md").exists() && out.join("rec-2.md").exists());
    }

    #[test]
    fn failed_move_keeps_audio_and_removes_transcription() {
        let base = tempfile::tempdir().unwrap();
        let (out, done) = (base.path().join("out"), base.path().join("missing-done"));
        fs::create_dir(&out).unwrap();
        let audio = base.path().join("rec.m4a");
        fs::write(&audio, b"audio").unwrap();
        // `done` does not exist, so the move fails after the md was written.
        assert!(store_result(&audio, &out, &done, &transcript("t"), &meta("rec.m4a")).is_err());
        assert!(audio.exists());
        assert!(!out.join("rec.md").exists());
    }

    #[test]
    fn copy_fallback_moves_content_and_removes_source() {
        let d = tempfile::tempdir().unwrap();
        let (src, dst) = (d.path().join("a.m4a"), d.path().join("b.m4a"));
        fs::write(&src, b"data").unwrap();
        copy_then_remove(&src, &dst).unwrap();
        assert!(!src.exists());
        assert_eq!(fs::read(&dst).unwrap(), b"data");
        assert!(!tmp_path(&dst).exists());
    }
}
