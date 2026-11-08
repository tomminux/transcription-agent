# transcription-agent

A background agent that transcribes audio recordings **locally** (no cloud):
QuickTime Player `.m4a` files and iPhone Voice Memos.

It runs in a Linux container (built for an Intel Mac) and watches a folder.
For each new audio file it writes a Markdown transcription and moves the audio
out of the way:

```
new-audio-files/  ──►  transcriptions/<name>.md
                  └──►  transcribed-audio-files/<name>.m4a
```

Audio is decoded in-process; files that variant the built-in decoder rejects
(HE-AAC, multichannel...) are decoded with `ffmpeg`, which is in the image.
The language is detected automatically. Transcription uses whisper.cpp
(CPU only), so expect it to take a fraction of the recording's length to a bit
more than real time, depending on the model.

## Run it (Mac with Docker Desktop, Colima or Podman)

```bash
git clone <this repo> && cd transcription-agent
scripts/setup.sh                  # writes .env (your uid/gid, data folder) and creates the folders
scripts/download-model.sh base    # downloads models/ggml-base.bin (~150 MB)
docker compose up -d --build      # first build compiles whisper.cpp: a few minutes
docker compose logs -f
```

Then drop a `.m4a` into
`/Users/paolo/docker-host-containers/transcription-agent/new-audio-files/`.
(Pass another folder to `scripts/setup.sh <dir>` to change it.)

Stop with `docker compose down` (the agent shuts down cleanly; a recording
being transcribed stays in `new-audio-files/` and is picked up on the next
start).

Give Docker at least 2 CPUs and 2 GB RAM (`base` needs ~1 GB, `small` ~2 GB).
The image needs a CPU with AVX2 (Intel Macs from 2013 on).

## Output

`transcriptions/<name>.md`:

```
---
source: "Nuova registrazione.m4a"
recorded_at: 2026-11-07T17:16:47Z
transcribed_at: 2026-11-07T17:16:52Z
duration_seconds: 63.4
language: "it"
model: "ggml-base.bin"
---

[00:00:00 - 00:00:04] First sentence as segmented by Whisper.
[00:00:04 - 00:00:09] Second sentence...
```

One line per segment: `[start - end] text`, in `HH:MM:SS`. The timestamps
(and the gaps between segments) are useful for a later step such as speaker
labelling or summarising with an LLM.

If a name is already used, the new pair is saved as `<name>-2.md` /
`<name>-2.m4a`; nothing is ever overwritten.

## Configuration

Environment variables (set in `.env`) or CLI flags:

| Variable | Default (container) | Meaning |
|---|---|---|
| `HOST_DATA_DIR` | `/Users/paolo/docker-host-containers/transcription-agent` | Host folder holding the three subfolders (mounted at `/data`) |
| `HOST_MODELS_DIR` | `./models` | Host folder with the ggml models (mounted read-only at `/models`) |
| `TA_MODEL` | `/models/ggml-base.bin` | Model file (e.g. `ggml-small.bin` for better accuracy) |
| `TA_SCAN_INTERVAL_SECS` | `5` | How often the input folder is rescanned |
| `RUST_LOG` | `info` | Log level |

To use another model: `scripts/download-model.sh small`, then set
`TA_MODEL=/models/ggml-small.bin` in `.env`.

## Troubleshooting

- **`model not found`**: run `scripts/download-model.sh base`.
- **A file stays in `new-audio-files/`**: check `docker compose logs`. A file
  that fails (corrupt, unsupported) is logged once and retried only if it
  changes. Delete it or replace it.
- **Files written as root / permission denied**: re-run `scripts/setup.sh`
  and `docker compose up -d --force-recreate`.
- **Nothing is picked up**: file events from macOS often don't reach the
  container, so the agent rescans every few seconds; wait a bit longer or
  lower `TA_SCAN_INTERVAL_SECS`.

## Development (Linux)

```bash
cargo test                                          # fast unit tests
TA_TEST_MODEL=models/ggml-tiny.bin cargo test       # also runs the real-model test
cargo clippy --all-targets -- -D warnings
```

Building outside Docker needs `cmake`, a C++ compiler and `clang`/`libclang`
(`LIBCLANG_PATH`). See `CLAUDE.md` for the design decisions and constraints.

Your recordings and transcriptions are private: they are gitignored and never
leave your machine.
