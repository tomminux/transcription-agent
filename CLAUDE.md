# CLAUDE.md

## Project

`transcription-agent`: a Rust background agent that transcribes audio recordings locally.

Supported inputs:
- `.m4a` recorded with QuickTime Player on macOS
- Audio recorded on iPhone with the Voice Memos app (`.m4a`, possibly shared/AirDropped/synced into the folder)

Target runtime: a **Linux x86_64 container** running on an **Intel Mac** (via Docker Desktop, Colima or Podman). The agent is not meant to run natively on macOS. The audio files live on the Mac and are shared with the container through bind mounts. Development happens on Linux, where the same container can be built and tested.

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
- **Background execution:** the agent runs in a container (replaces the earlier "manual run / launchd" plan). The binary stays a plain foreground process (PID 1 in the container); `docker compose up -d` with `restart: unless-stopped` keeps it running in the background and restarts it after a crash or reboot. Do not build daemonization into the binary, and do not add launchd.
- **Container packaging:** multi-stage `Dockerfile` (Rust builder with `cmake` + C++ toolchain, slim Debian runtime) and a `compose.yaml`. Keep it runtime-agnostic (no Docker-only features) so Docker Desktop, Colima and Podman all work. Build for `linux/amd64`.

## Container constraints (important)

- **Bind mounts:** on the Mac the data lives in `/Users/paolo/dockerhost-storage/transcription-agent/` with the subfolders `new-audio-files/`, `transcriptions/` and `transcribed-audio-files/` (same structure as in this repo). `compose.yaml` mounts that **single parent folder** at `/data` (`${HOST_DATA_DIR:-/Users/paolo/dockerhost-storage/transcription-agent}:/data`), so the agent sees `/data/new-audio-files`, `/data/transcriptions` and `/data/transcribed-audio-files` as ordinary local files. `models/` is mounted read-only at `/models` (`${HOST_MODELS_DIR:-./models}:/models:ro`); models are not baked into the image. The agent creates missing subfolders at startup. The in-repo folders are only for local development on Linux (contents gitignored).
- **File watching on a Mac host is unreliable:** with Docker Desktop/Colima, changes made on macOS often do not generate inotify events inside the container. The agent must therefore also do a **periodic rescan** of the input folder (e.g. every few seconds, interval configurable) in addition to `notify` events (or use `notify`'s `PollWatcher`). The startup scan already exists. Never rely only on events.
- **Moves:** since all three folders are under one mount, `rename()` works. Still keep a fallback to copy + fsync + delete on `EXDEV` (only delete the source after a successful copy), in case the folders are configured on separate mounts.
- **Configuration through environment variables** as well as CLI flags (clap `env`), with container defaults pointing at `/data/...` and `/models/...`.
- **Permissions:** run as a non-root user; support mapping the host UID/GID (`user: "${HOST_UID}:${HOST_GID}"` in compose, values come from a gitignored `.env` written by `scripts/setup.sh`; do not use `UID`, it is a read-only shell variable and not exported) so files written to the Mac folders are owned by the Mac user.
- **CPU build:** do not build the image with `-C target-cpu=native` / host-specific whisper.cpp flags (the build machine may differ from the run machine). The Dockerfile sets `GGML_NATIVE=OFF` plus `GGML_AVX/AVX2/FMA/F16C=ON` (env vars starting with `GGML_` are forwarded to cmake by `whisper-rs-sys`), so the image needs an Intel Mac with AVX2 (2013+).
- **Resources:** Whisper is CPU- and memory-heavy; document the Docker Desktop CPU/RAM limits needed for the chosen model (`base` ~1 GB, `small` ~2 GB).
- **Logs** go to stdout/stderr (`docker compose logs -f`), not to files.
- **Graceful shutdown:** handle SIGTERM (container stop). Finish or abandon the current file safely; the audio stays in `new-audio-files/` unless fully processed.

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
scripts/download-model.sh base   # models/ggml-base.bin (use `tiny` for quick tests)
TA_TEST_MODEL=models/ggml-tiny.bin cargo test   # also runs the real-model test
cargo test
cargo fmt --check
cargo clippy -- -D warnings

scripts/setup.sh              # write .env (uid/gid, data dir) and create folders
docker compose build          # build the linux/amd64 image
docker compose up -d          # run the agent in the background
docker compose logs -f        # follow the logs
docker compose down           # stop it
```

Build prerequisites for `whisper-rs`: `cmake`, a C/C++ toolchain and `clang`/`libclang` (bindgen; set `LIBCLANG_PATH` if it is not found). Inside the container these are provided by the builder stage, so nothing needs to be installed on the Mac except Docker (or Colima/Podman).

## Conventions

- Keep modules small and separated: watcher, decoder, transcriber, writer/mover, config.
- Use `anyhow` for app errors (or `thiserror` for library-style modules) and `tracing` for logs.
- Add unit tests for pure logic (file filtering, name collision, front matter); keep tests independent of model downloads.
- Run `cargo fmt` and `cargo clippy` before each commit.

## Gotchas

- `whisper-rs` 0.16.0: `FullParams::set_abort_callback_safe` is broken (wrong user-data pointer type, whisper fails with "failed to encode"). `transcriber.rs` uses the raw `set_abort_callback` with its own C trampoline instead. Re-test before switching back after a `whisper-rs` upgrade.
- `whisper-rs`: use `set_language(Some("auto"))`, never `set_detect_language(true)` (detects only, no transcription).
- Shutdown: first SIGTERM/SIGINT finishes cleanly (aborts a running transcription, leaves the audio in `new-audio-files/`); a second signal exits immediately.

## Git

- Repository is private. Commit in small, logical steps with clear messages; commit after each meaningful milestone.
- Never commit `target/`, `models/`, or the contents of the three audio/transcript folders. Add a `.dockerignore` so these (and `.git`) are never sent to the image build context.
