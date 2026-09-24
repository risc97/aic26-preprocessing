#!/usr/bin/env bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEOS_DIR="$ROOT_DIR/data/videos"
AUDIOS_DIR="$ROOT_DIR/data/audios"

mkdir -p "$AUDIOS_DIR"
parallel ffmpeg -y -v error -i {} -vn -c:a copy "$AUDIOS_DIR/{/.}.ogg" ::: "$VIDEOS_DIR"/*.webm
parallel ffmpeg -y -v error -i {} -vn -c:a libopus -b:a 64k "$AUDIOS_DIR/{/.}.ogg" ::: "$VIDEOS_DIR"/*.mp4