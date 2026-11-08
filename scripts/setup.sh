#!/usr/bin/env bash
# Creates .env (gitignored) so the container runs as your user and the files
# it writes to the mounted folders belong to you. Safe to re-run.
# Usage: scripts/setup.sh [data-dir]
#   data-dir defaults to /Users/paolo/docker-host-containers/transcription-agent
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
data_dir="${1:-/Users/paolo/docker-host-containers/transcription-agent}"

cat > "$root/.env" <<ENV
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_DATA_DIR=$data_dir
HOST_MODELS_DIR=$root/models
# TA_MODEL=/models/ggml-small.bin
# TA_SCAN_INTERVAL_SECS=5
# RUST_LOG=info
ENV

mkdir -p "$data_dir/new-audio-files" "$data_dir/transcriptions" "$data_dir/transcribed-audio-files"
echo "Wrote $root/.env (uid=$(id -u) gid=$(id -g), data=$data_dir)"
