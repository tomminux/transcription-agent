# CLAUDE.md

## Project

`transcription-agent`: a Rust background agent that transcribes audio recordings locally.

Supported inputs:
- `.m4a` recorded with QuickTime Player on macOS
- Audio recorded on iPhone with the Voice Memos app (`.m4a`, possibly shared/AirDropped/synced into the folder)

Target runtime: **macOS on an Intel Mac** (x86_64). Development may happen on Linux, so the code must also build there.

## Workflow

1. The agent watches `new-audio-files/`.
2. When a new audio file appears (and is fully written), it is transcribed.
3. The transcription is saved to `transcriptions/<audio-base-name>.md`.
4. The audio file is moved to `transcribed-audio-files/`.

Directories (at repo root, contents are gitignored, `.gitkeep` keeps them in git):
- `new-audio-files/` – input, watched
- `transcriptions/` – output Markdown
- `transcribed-audio-files/` – processed audio

## Decisions made

- **Language:** Rust (stable, edition 2021+).
- **Engine:** local whisper.cpp via the `whisper-rs` crate. Offline, no cloud. Intel Mac means CPU inference only (no Metal gain), so default to a modest model (`base` or `small`, multilingual, optionally quantized) and keep the model path configurable. Models live in `models/` (gitignored), never committed.
- **Language handling:** auto-detect per file (use multilingual models, not `.en`).
- **Audio decoding:** pure Rust where possible: `symphonia` (AAC in MP4/M4A) + resample to 16 kHz mono f32 (e.g. `rubato`) before feeding Whisper.
- **Folder watching:** `notify` crate. Must also scan the folder at startup for files already present. Wait until a file's size is stable before processing, since Voice Memos/QuickTime/AirDrop may still be writing it.
- **Output format:** Markdown with YAML front matter (source filename, date, duration, detected language, model) followed by the transcript text.
- **Background execution:** manual foreground run for now (`cargo run --release`). The binary is a plain foreground process; launchd LaunchAgent packaging is a possible later step, do not build daemonization into the binary.

## Behavior rules

- Never lose audio: only move the audio file after the transcription file has been written successfully. On failure, leave the file in `new-audio-files/` and log the error (avoid infinite retry loops; track failures).
- Never overwrite an existing transcription silently; on name collision, add a numeric/timestamp suffix.
- Process files sequentially (Whisper is CPU heavy); queue new arrivals.
- Ignore hidden/temp files (`.DS_Store`, `.gitkeep`, partial downloads) and non-audio extensions.
- Audio and transcripts are private. Never commit them or send them to external services.

## Commands

```
cargo build --release
cargo run --release
cargo test
cargo fmt --check
cargo clippy -- -D warnings
```

Build prerequisites for `whisper-rs`: `cmake` and a C/C++ toolchain (Xcode Command Line Tools on macOS).

## Conventions

- Keep modules small and separated: watcher, decoder, transcriber, writer/mover, config.
- Use `anyhow` for app errors (or `thiserror` for library-style modules) and `tracing` for logs.
- Add unit tests for pure logic (file filtering, name collision, front matter); keep tests independent of model downloads.
- Run `cargo fmt` and `cargo clippy` before each commit.

## Git

- Repository is private. Commit in small, logical steps with clear messages; commit after each meaningful milestone.
- Never commit `target/`, `models/`, or the contents of the three audio/transcript folders.
