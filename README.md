# transcription-agent

A background agent that transcribes audio recordings **locally** (no cloud):
QuickTime Player `.m4a` files and iPhone Voice Memos.

It runs in a Linux container and watches a folder. The image is built natively
for the machine's architecture: **arm64 on Apple Silicon Macs** (the main target,
with Podman), amd64 on x86 machines (Intel Macs, Linux).
For each new audio file it writes a Markdown transcription and moves the audio
out of the way:

```
new-audio-files/  ──►  transcriptions/<name>.md
                  └──►  transcribed-audio-files/<name>.m4a
```

Audio is decoded in-process; files that the built-in decoder rejects
(HE-AAC, multichannel...) are decoded with `ffmpeg`, which is in the image.
The language is detected automatically. Transcription uses whisper.cpp on the
CPU: a container on macOS runs in a Linux VM with no access to the Mac's GPU
(no Metal). Apple Silicon CPUs are still fast, and the log line printed for each
file shows the processing time and speed (e.g. `speed=4.2x` = 4.2 times faster
than the recording length), so you can compare models on your own audio.

## Run it with Docker (Mac with Docker Desktop or Colima)

```bash
git clone <this repo> && cd transcription-agent
scripts/setup.sh                  # writes .env (your uid/gid, data folder) and creates the folders
scripts/download-model.sh base    # downloads models/ggml-base.bin (~150 MB)
docker compose up -d --build      # first build compiles whisper.cpp: a few minutes
docker compose logs -f
```

Then drop a `.m4a` into
`$HOME/docker-host-containers/transcription-agent/new-audio-files/`.
(Pass another folder to `scripts/setup.sh <dir>` to change it.)

Stop with `docker compose down` (the agent shuts down cleanly; a recording
being transcribed stays in `new-audio-files/` and is picked up on the next
start).

Long recordings need memory: the whole decoded audio is held in RAM
(about 230 MB per hour, up to roughly twice that while loading). A 3-hour
file is fine with 4 GB for the VM, 6-8 hour files want 8 GB
(`podman machine set --memory 8192`). If the container restarts in a loop on a
long file and the log stops right after `trying ffmpeg`, it was killed for
memory: check with
`podman inspect <container> --format '{{.State.OOMKilled}} {{.RestartCount}}'`.

Give Docker at least 2 CPUs and 2 GB RAM (`base` needs ~1 GB, `small` ~2 GB).
On x86 the image needs a CPU with AVX2 (Intel Macs from 2013 on). On Apple
Silicon it is a native arm64 build; do not force `--platform linux/amd64`
(that would run under slow emulation).

## Run it with Podman (macOS)

> Not yet tested on a real Mac: if something differs, see Troubleshooting and
> tell us what you got.

Podman on macOS runs containers inside a small Linux VM (`podman machine`).
On an Apple Silicon Mac (M1...M5) the VM is **arm64**, so the image is built
and runs natively (the Dockerfile picks arm64 CPU flags automatically). Never
pass `--platform linux/amd64`: it would run under emulation and be very slow.

```bash
# 1. Install and create the VM (once). CPUs/RAM go to the VM. Whisper needs
#    at least 2 CPUs and 2 GB RAM (4096 MB for the `small` model). More threads
#    than the chip has performance cores rarely help (e.g. --cpus 4).
brew install podman
podman machine init --cpus 4 --memory 4096
podman machine start

# 2. Repo setup. --podman runs the agent as container "root" (uid 0), which
#    rootless Podman maps to your Mac user, so files in the mounted folders are
#    yours. (Do not use your own uid here: it would map to an unrelated uid.)
git clone git@github.com:tomminux/transcription-agent.git && cd transcription-agent
scripts/setup.sh --podman
scripts/download-model.sh small    # `base` is faster but less accurate; `small` is a good fit for an M-series CPU
```

The default `podman machine` shares your home folder (`$HOME`) with the
VM, so `$HOME/docker-host-containers/transcription-agent` and the
repo's `models/` folder are visible to containers without extra options.

Then start it, **either** with compose:

```bash
brew install podman-compose        # or docker-compose; `podman compose` uses whichever is installed
podman compose up -d --build
podman compose logs -f
podman compose down                # stop
```

**or** with plain Podman (no compose needed):

```bash
podman build -t transcription-agent:local .

DATA="$HOME/docker-host-containers/transcription-agent"
podman run -d --name transcription-agent --restart unless-stopped --init \
  -v "$DATA":/data \
  -v "$PWD/models":/models:ro \
  -e TA_MODEL=/models/ggml-small.bin \
  -e TA_SCAN_INTERVAL_SECS=5 \
  transcription-agent:local

podman logs -f transcription-agent
podman stop transcription-agent    # clean shutdown; `podman start transcription-agent` to resume
podman rm -f transcription-agent   # remove it (e.g. before re-running with a new image)
```

After a Mac reboot the VM is not running: run `podman machine start`, then
`podman start transcription-agent` (or `podman compose up -d`).
To change the model, add `TA_MODEL=/models/ggml-small.bin` to `.env` (compose)
or change the `-e TA_MODEL=...` option (plain Podman).

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
| `HOST_DATA_DIR` | `$HOME/docker-host-containers/transcription-agent` | Host folder holding the three subfolders (mounted at `/data`) |
| `HOST_MODELS_DIR` | `./models` | Host folder with the ggml models (mounted read-only at `/models`) |
| `TA_MODEL` | `/models/ggml-base.bin` | Model file (e.g. `ggml-small.bin` for better accuracy) |
| `TA_LANGUAGE` | `auto` | Spoken language: `auto`, one code (`it`), or a list (`it,en`) to detect only among those |
| `TA_SCAN_INTERVAL_SECS` | `5` | How often the input folder is rescanned |
| `RUST_LOG` | `info` | Log level |
| `TA_LLM` | `none` | Meeting report backend: `none`, `openai` (private server), `ollama`, `anthropic` |
| `TA_LLM_DEFAULT` | `none` | For files in the root of `new-audio-files/`: `none` or `report` |
| `TA_LLM_MODEL` | `claude-opus-5-5` (anthropic) | Model name (required for ollama) |
| `TA_LLM_URL` | per backend | Ollama default `http://host.containers.internal:11434` |
| `TA_LLM_NUM_CTX` | `32768` | Ollama context window |
| `TA_REPORT_LANGUAGE` | transcript's | e.g. `Italian` |
| `TA_LLM_API_KEY` | | Bearer key for `openai` |
| `TA_LLM_CA_CERT` | | PEM of a private CA (e.g. `/models/intranet-ca.pem`) |
| `ANTHROPIC_API_KEY` | | API key for `anthropic` |

**Language.** `auto` lets Whisper guess from the first 30 seconds only. A silent
or noisy intro, music, or a small model can make it pick the wrong language (we
saw a recording detected as Vietnamese), and then the whole transcript is wrong.
If you know which languages you record, set them: `TA_LANGUAGE=it` forces one
language, and `TA_LANGUAGE=it,en` detects the language **only among those**,
sampling 5 points spread over the recording (the log line `language detection
among candidates ...` shows the probability of each). The detected language is
written in the front matter.

To use another model: `scripts/download-model.sh small`, then set
`TA_MODEL=/models/ggml-small.bin` in `.env`.

## Meeting reports (optional)

**Choosing per recording.** Drop each audio file in the folder that says what
you want:

```
new-audio-files/
  (file here)       -> what TA_LLM_DEFAULT says (default: transcription only)
  transcribe-only/  -> transcription only, never sent to the LLM
  with-report/      -> transcription + AI report
```

The agent creates the two subfolders at startup. The choice is written in the
transcription's front matter as `report: requested` or `report: skipped`. At
startup the agent writes missing reports **only** for `report: requested`;
transcriptions made before this feature (no `report:` field) or marked
`skipped` are never sent to the LLM. If you want a report for one of them later,
add `report: requested` to its front matter and restart the container.

With an LLM enabled, each new transcription also gets a report,
`transcriptions/<name>-report.md`: summary, participants (a best guess, the
transcript has no speaker labels), topics with timestamps, decisions, action
items, open questions. It is **off by default**. The transcription and the audio
are saved first, so an LLM failure only logs an error and never loses anything.
At startup the agent also writes reports for existing transcriptions that have
none (this is also how a failed one is retried: restart the container).

**Private server on the intranet / VPN (OpenAI-compatible API: Open WebUI, vLLM,
LiteLLM, llama.cpp server, ...):**

```
TA_LLM=openai
TA_LLM_URL=https://llm.intranet/v1          # base URL; /chat/completions is appended
                                            # Open WebUI: https://llm.intranet/api
TA_LLM_MODEL=<model name the server exposes>
TA_LLM_API_KEY=...                          # sent as "Authorization: Bearer ..."
TA_LLM_CA_CERT=/models/intranet-ca.pem      # only if the server uses a private CA
```

The container reaches the server through the Mac's network, so the VPN must be
up on the Mac. With Podman, the VM follows the Mac's routes but may not use the
VPN's DNS: if the name does not resolve, use the server's IP. To test from the
same network namespace as the agent:

```
podman run --rm docker.io/curlimages/curl -sS -H "Authorization: Bearer $KEY" https://llm.intranet/v1/models
```

The answer must be JSON listing the models (use those names for `TA_LLM_MODEL`).
If you get an HTML page (`<title>Open WebUI</title>`), the path is wrong: that
is the web interface. **Open WebUI** serves the API under `/api`
(`/api/models`, `/api/chat/completions`), so use `TA_LLM_URL=https://host/api`,
and create the key in *Settings > Account > API keys* (an admin may need to
enable API keys).

A private CA certificate (PEM) can be put in the `models/` folder, which is
already mounted at `/models`, and referenced by `TA_LLM_CA_CERT`. It replaces
the public roots for this connection only. Models that print their reasoning
inside `<think>...</think>` are cleaned automatically.

**Local, private (Ollama on the Mac).** Nothing leaves the machine:

```
TA_LLM=ollama
TA_LLM_MODEL=qwen3:14b      # any model you pulled with `ollama pull`
```

Ollama must listen where the container can reach it. From a Podman container the
Mac is `host.containers.internal` (the default `TA_LLM_URL` is
`http://host.containers.internal:11434`); with Docker use
`TA_LLM_URL=http://host.docker.internal:11434`. Ollama binds to 127.0.0.1 by
default, which the VM cannot reach: start it with `OLLAMA_HOST=0.0.0.0` (or
`launchctl setenv OLLAMA_HOST 0.0.0.0` and restart the app). `TA_LLM_NUM_CTX`
(default 32768) matters: Ollama silently truncates anything beyond its context
window, and an hour of speech is roughly 15-20k tokens.

**Claude API (better reports, the transcript is sent to Anthropic):**

```
TA_LLM=anthropic
ANTHROPIC_API_KEY=sk-ant-...
TA_LLM_MODEL=claude-opus-5-5    # default; e.g. claude-sonnet-5-5 is cheaper
```

`TA_REPORT_LANGUAGE=Italian` forces the report language (default: the
transcript's). Only the transcription Markdown is sent, never the audio.

## Troubleshooting

- **Podman: `permission denied` writing to `/data` or the folders appear
  owned by a strange uid**: `.env` must come from `scripts/setup.sh --podman`
  (`HOST_UID=0`), then recreate the container. Also check that the folder is
  inside your home folder (the only one `podman machine` shares by default) and
  that `podman machine` is running (`podman machine list`).
- **Podman: `podman compose` fails on a compose key**: use the plain
  `podman build` / `podman run` commands above instead.
- **Very slow, or `exec format error`**: the image was built for the wrong
  architecture (e.g. with `--platform linux/amd64`). Check with
  `podman image inspect transcription-agent:local --format '{{.Architecture}}'`
  (it must be `arm64` on Apple Silicon), then rebuild without `--platform`.

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
