# syntax=docker/dockerfile:1
# Build for linux/amd64 (Intel Mac): docker compose build

FROM rust:slim-bookworm AS builder
# cmake + C++ toolchain are needed later by whisper-rs (whisper.cpp).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Build dependencies first so they are cached until Cargo.toml/Cargo.lock change.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
    && cargo build --release \
    && rm -rf src target/release/transcription-agent* target/release/deps/transcription_agent*

COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
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
