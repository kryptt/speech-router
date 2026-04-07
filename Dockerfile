# syntax=docker/dockerfile:1
FROM rust:1-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    libclang-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git \
    cargo build --release

FROM builder AS test

RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git \
    cargo test -- --test-threads=1

FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libavformat61 \
    libavcodec61 \
    libavutil59 \
    libswresample5 \
    libswscale8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/speech-router /speech-router

EXPOSE 8000 10300 9090

ENTRYPOINT ["/speech-router"]
