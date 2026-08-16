#!/usr/bin/env python3
"""
Gipformer ONNX Inference - Vietnamese ASR
Uses sherpa-onnx for fast, cross-platform speech recognition.

Usage:
    python aic26-preprocessing/extract_transcript.py --audio data/audios/L21_V001.ogg
    python aic26-preprocessing/extract_transcript.py --audio-dir data/audios/ --quantize int8
    python infer_onnx.py --audio data/audio.wav
    python infer_onnx.py --audio data/audio.wav --quantize int8
    python infer_onnx.py --audio file1.wav file2.wav --num-threads 4
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from config import TRANSCRIPTS_DIR

try:
    import sherpa_onnx
except ImportError as e:
    print(f"Error: could not import sherpa-onnx ({e}).")
    print("Install it with: pip install sherpa-onnx sherpa-onnx-core")
    print(
        "Note: as of sherpa-onnx 1.13, the native libs (libonnxruntime.so) live "
        "in the separate 'sherpa-onnx-core' package — both must be installed."
    )
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Install it with: pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "g-group-ai-lab/gipformer-65M-rnnt"
SAMPLE_RATE = 16000
FEATURE_DIM = 80

DEFAULT_CHUNK_SECONDS = 20.0
DEFAULT_BATCH_SIZE = 8
SILENCE_SEARCH_SECONDS = 1.0
SILENCE_FRAME_SECONDS = 0.02

MIN_CHUNK_SECONDS = 1.0
AUDIO_EXTENSIONS = {".wav", ".ogg", ".flac", ".mp3", ".m4a"}

ONNX_FILES = {
    "fp32": {
        "encoder": "encoder-epoch-35-avg-6.onnx",
        "decoder": "decoder-epoch-35-avg-6.onnx",
        "joiner": "joiner-epoch-35-avg-6.onnx",
    },
    "int8": {
        "encoder": "encoder-epoch-35-avg-6.int8.onnx",
        "decoder": "decoder-epoch-35-avg-6.int8.onnx",
        "joiner": "joiner-epoch-35-avg-6.int8.onnx",
    },
}


def download_model(quantize: str = "int8") -> dict:
    """Download ONNX model files from HuggingFace.

    Args:
        quantize: "fp32" for full precision, "int8" for quantized (smaller, faster).

    Returns:
        Dict with local paths to encoder, decoder, joiner, and tokens files.
    """
    files = ONNX_FILES[quantize]
    print(f"Downloading {quantize} model from {REPO_ID}...")

    paths = {}
    for key, filename in files.items():
        paths[key] = hf_hub_download(repo_id=REPO_ID, filename=filename)

    paths["tokens"] = hf_hub_download(repo_id=REPO_ID, filename="tokens.txt")

    print("Model downloaded successfully.")
    return paths


def read_audio(filename: str) -> tuple:
    """Read an audio file and return float32 samples and sample rate.

    Supports WAV, FLAC, OGG, and other formats via soundfile.
    """
    samples, sample_rate = sf.read(filename, dtype="float32")

    # Convert stereo to mono if needed
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    return samples, sample_rate


def create_recognizer(
    model_paths: dict,
    num_threads: int = 4,
    decoding_method: str = "greedy_search",
) -> sherpa_onnx.OfflineRecognizer:
    """Create a sherpa-onnx OfflineRecognizer for transducer decoding."""
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=model_paths["encoder"],
        decoder=model_paths["decoder"],
        joiner=model_paths["joiner"],
        tokens=model_paths["tokens"],
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=FEATURE_DIM,
        decoding_method=decoding_method,
        debug=False
    )


def split_on_silence(
    samples: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    search_seconds: float = SILENCE_SEARCH_SECONDS,
    frame_seconds: float = SILENCE_FRAME_SECONDS,
) -> list:
    """Split samples into chunks of roughly `chunk_seconds`, breaking at the
    quietest point within `search_seconds` of each boundary to avoid cutting
    off a word mid-syllable.

    Returns a list of (start_sample, end_sample, chunk) tuples.
    """
    chunk_len = int(chunk_seconds * sample_rate)
    search_len = int(search_seconds * sample_rate)
    frame_len = max(1, int(frame_seconds * sample_rate))
    min_chunk_len = int(MIN_CHUNK_SECONDS * sample_rate)
    total = len(samples)

    if total <= chunk_len:
        return [(0, total, samples)]

    chunks = []
    start = 0
    while start < total:
        boundary = start + chunk_len
        if boundary >= total or (total - boundary) < min_chunk_len:
            if chunks and (total - start) < min_chunk_len:
                # fold it into the previous chunk instead of emitting a near-empty one.
                prev_start, _, _ = chunks[-1]
                chunks[-1] = (prev_start, total, samples[prev_start:])
            else:
                chunks.append((start, total, samples[start:]))
            break

        lo = max(start + 1, boundary - search_len)
        hi = min(total, boundary + search_len)
        window = samples[lo:hi]

        n_frames = max(1, len(window) // frame_len)
        frame_rms = [
            np.sqrt(np.mean(window[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ]
        quietest_frame = int(np.argmin(frame_rms))
        split = lo + quietest_frame * frame_len + frame_len // 2

        chunks.append((start, split, samples[start:split]))
        start = split

    return chunks


def transcribe(
    recognizer: sherpa_onnx.OfflineRecognizer,
    audio_path: str,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list:
    """Transcribe a single audio file, chunking long audio to keep the
    encoder's self-attention memory usage bounded.

    Chunks are decoded in batches of `batch_size` rather than all at once,
    since decode_streams holds every stream's encoder/beam-search state in
    memory for the duration of the call — batching the whole file at once
    makes peak memory scale with file length instead of a fixed cap.

    Returns a list of {"time_start_ms", "time_end_ms", "text"} dicts, one per chunk.
    """
    try:
        samples, sample_rate = read_audio(audio_path)
    except Exception as e:
        print(f"  Warning: Failed to read {audio_path}: {e}")
        return []

    min_samples = int(MIN_CHUNK_SECONDS * sample_rate)
    if samples.size < min_samples:
        print(f"  Warning: Audio file too short or empty: {audio_path}")
        return []

    chunks = [c for c in split_on_silence(samples, sample_rate, chunk_seconds=chunk_seconds) if c[2].size > 0]

    rows = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        streams = []
        spans_ms = []
        for start, end, chunk in batch:
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, chunk)
            streams.append(stream)
            spans_ms.append(
                (round(start / sample_rate * 1000), round(end / sample_rate * 1000))
            )

        recognizer.decode_streams(streams)

        rows.extend(
            {
                "time_start_ms": start_ms,
                "time_end_ms": end_ms,
                "text": stream.result.text.strip().lower(),
            }
            for (start_ms, end_ms), stream in zip(spans_ms, streams)
        )

    return rows


def save_transcript_csv(rows: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_start_ms", "time_end_ms", "text"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Gipformer ONNX Inference - Vietnamese ASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python infer_onnx.py --audio data/audio.wav\n"
        "  python infer_onnx.py --audio data/audio.wav --quantize fp32\n"
        "  python infer_onnx.py --audio f1.wav f2.wav --num-threads 4\n",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        default=[],
        help="Path(s) to audio file(s) to transcribe",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory containing audio files to transcribe",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["fp32", "int8"],
        default="fp32",
        help="Model precision: fp32 (full, default) or int8 (quantized)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for inference (default: 4)",
    )
    parser.add_argument(
        "--decoding-method",
        type=str,
        choices=["greedy_search", "modified_beam_search"],
        default="modified_beam_search",
        help="Decoding method (default: modified_beam_search)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help=(
            "Split audio into chunks of about this many seconds before "
            f"transcribing (default: {DEFAULT_CHUNK_SECONDS}). The encoder's "
            "self-attention is O(T^2) in audio length, so long files must be "
            "chunked or they can exhaust memory."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of chunks to decode per decode_streams() call "
            f"(default: {DEFAULT_BATCH_SIZE}). decode_streams keeps every "
            "stream's encoder/beam-search state in memory for the call's "
            "duration, so lower this if you're still hitting memory pressure "
            "on long audio files."
        ),
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=TRANSCRIPTS_DIR,
        help=f"Directory to save transcript CSVs to (default: {TRANSCRIPTS_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcript CSV files instead of skipping them",
    )
    args = parser.parse_args()

    audio_files = [Path(p) for p in args.audio]

    if args.audio_dir:
        if not args.audio_dir.is_dir():
            print(f"Error: {args.audio_dir} is not a valid directory.")
            sys.exit(1)
        dir_files = [
            p
            for p in args.audio_dir.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ]
        audio_files.extend(dir_files)

    if not audio_files:
        print("Error: No audio files found. Provide --audio or --audio-dir.")
        sys.exit(1)

    # Download model
    model_paths = download_model(args.quantize)

    # Create recognizer
    recognizer = create_recognizer(
        model_paths,
        num_threads=args.num_threads,
        decoding_method=args.decoding_method,
    )

    # Transcribe each audio file
    for audio_path in sorted(audio_files):
        output_path = args.transcripts_dir / f"{audio_path.stem}.csv"
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            print(f"  Skipping {audio_path.name} (already processed -> {output_path.name})")
            continue

        audio_path_str = str(audio_path)
        start = time.time()
        rows = transcribe(
            recognizer,
            audio_path_str,
            chunk_seconds=args.chunk_seconds,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - start

        audio_info = sf.info(audio_path_str)
        duration = audio_info.duration
        rtf = elapsed / duration if duration > 0 else 0

        output_path = args.transcripts_dir / f"{audio_path.stem}.csv"
        save_transcript_csv(rows, output_path)

        print(f"\n  File: {audio_path_str}")
        print(f"  Saved: {output_path} ({len(rows)} chunks)")
        print(f"  Time: {elapsed:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()