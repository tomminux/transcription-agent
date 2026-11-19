#!/usr/bin/env bash
# Creates .env (gitignored) so the container runs with the right user and the
# files it writes to the mounted folders belong to you. Safe to re-run.
#
# Usage: scripts/setup.sh [--podman] [data-dir]
#   data-dir  defaults to /Users/paolo/docker-host-containers/transcription-agent
#   --podman  rootless Podman (macOS): run as container "root" (uid 0), which
#             Podman maps to your own user, so files in the mounted folders end
#             up owned by you. With Docker the container runs as your uid/gid.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
podman=false
data_dir="/Users/paolo/docker-host-containers/transcription-agent"
for arg in "$@"; do
  case "$arg" in
    --podman) podman=true ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) data_dir="$arg" ;;
  esac
done

if $podman; then uid=0; gid=0; else uid=$(id -u); gid=$(id -g); fi

cat > "$root/.env" <<ENV
HOST_UID=$uid
HOST_GID=$gid
HOST_DATA_DIR=$data_dir
HOST_MODELS_DIR=$root/models
# TA_MODEL=/models/ggml-small.bin
# TA_LANGUAGE=it,en   # auto | it | it,en (detect only among these)
# TA_SCAN_INTERVAL_SECS=5
# RUST_LOG=info
# Meeting report from each transcription (off by default):
# TA_LLM_DEFAULT=none      # files dropped in new-audio-files/ itself: none (transcription only) or report
# TA_LLM=openai            # private OpenAI-compatible server (intranet/VPN)
# TA_LLM_URL=https://llm.intranet/v1
# TA_LLM_MODEL=<model name the server exposes>
# TA_LLM_API_KEY=...
# TA_LLM_CA_CERT=/models/intranet-ca.pem   # only if the server uses a private CA
# TA_LLM=ollama            # local model, nothing leaves the Mac
# TA_LLM_MODEL=qwen3:14b   # a model you have pulled with `ollama pull`
# TA_LLM=anthropic         # Claude API: the transcript is SENT to Anthropic
# ANTHROPIC_API_KEY=sk-ant-...
# TA_REPORT_LANGUAGE=Italian
ENV

mkdir -p "$data_dir/new-audio-files" "$data_dir/transcriptions" "$data_dir/transcribed-audio-files"
echo "Wrote $root/.env (uid=$uid gid=$gid, data=$data_dir)"
