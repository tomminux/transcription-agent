# syntax=docker/dockerfile:1
# Builds natively for the machine's architecture: arm64 on Apple Silicon (podman/docker
# machine VM), amd64 on x86 Linux. Do not force --platform linux/amd64 on a Mac
# with Apple Silicon: it would run under slow emulation.

FROM rust:slim-bookworm AS builder
# whisper-rs builds whisper.cpp with cmake + a C++ toolchain, and generates
# bindings with bindgen (needs clang/libclang).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake pkg-config clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Portable CPU baselines (not -march=native), so the image runs on any CPU of
# the same architecture whatever machine built it. GGML_* env vars are forwarded
# to cmake by whisper-rs-sys; the flags are written to a file because they
# depend on the architecture and must apply to both cargo builds below.
#   amd64: AVX2 + FMA + F16C (Intel Macs from 2013, any modern x86 PC)
#   arm64: armv8.2-a + dotprod + fp16 (every Apple M1..M5; NEON, no Metal in a VM)
RUN case "$(uname -m)" in \
      x86_64)  echo 'export GGML_NATIVE=OFF GGML_AVX=ON GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON' ;; \
      aarch64) echo 'export GGML_NATIVE=OFF GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16' ;; \
      *)       echo 'export GGML_NATIVE=OFF' ;; \
    esac > /etc/ggml-env.sh

# Build dependencies first so they are cached until Cargo.toml/Cargo.lock change.
COPY Cargo.toml Cargo.lock ./
RUN . /etc/ggml-env.sh \
    && mkdir src && echo 'fn main() {}' > src/main.rs \
    && cargo build --release \
    && rm -rf src target/release/transcription-agent* target/release/deps/transcription_agent* \
              target/release/.fingerprint/transcription-agent-* target/release/incremental

COPY src ./src
# Cargo decides what to rebuild by file mtimes. Sources copied from the host
# can be OLDER than the dummy build above, so Cargo would keep the empty
# `fn main() {}` binary (container starts, prints nothing, exits 0). The
# fingerprint removal above and this touch guarantee the real build.
RUN . /etc/ggml-env.sh && touch src/*.rs && cargo build --release \
    && test "$(target/release/transcription-agent --version)" != "" 

FROM debian:bookworm-slim AS runtime
# ffmpeg: fallback decoder for AAC variants the built-in decoder rejects (HE-AAC, multichannel...).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libstdc++6 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home agent
COPY --from=builder /app/target/release/transcription-agent /usr/local/bin/transcription-agent

# Container defaults: data under /data (one bind mount), model under /models.
ENV TA_INPUT_DIR=/data/new-audio-files \
    TA_OUTPUT_DIR=/data/transcriptions \
    TA_DONE_DIR=/data/transcribed-audio-files \
    TA_MODEL=/models/ggml-base.bin \
    RUST_LOG=info
USER agent
ENTRYPOINT ["transcription-agent"]
