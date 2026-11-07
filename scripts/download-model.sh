#!/usr/bin/env bash
# Downloads a whisper.cpp ggml model into ./models (gitignored).
# Usage: scripts/download-model.sh [base|small|medium|tiny|...]   (default: base)
set -euo pipefail

name="${1:-base}"
dir="$(cd "$(dirname "$0")/.." && pwd)/models"
file="ggml-${name}.bin"
url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${file}"

mkdir -p "$dir"
if [[ -s "$dir/$file" ]]; then
  echo "$dir/$file already exists"
  exit 0
fi
echo "Downloading $file ..."
curl -fL --progress-bar -o "$dir/$file.part" "$url"
mv "$dir/$file.part" "$dir/$file"
echo "Saved $dir/$file"
