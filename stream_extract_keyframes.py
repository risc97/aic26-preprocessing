#!/usr/bin/env python3
"""Stream keyframes out of the remote video zips without downloading them.

For each group zip URL: parse the central directory over HTTP Range requests,
then for each video fetch just its stored entry bytes into a /dev/shm scratch
file, run ffmpeg's select filter, and write JPEGs to <out>/<video_id>/NNNN.jpg.
No video is ever kept on disk; the mp4 scratch file lives only in RAM.

Entries are STORED (no deflate), so each entry's bytes are raw mp4 fetched with
a single Range request. Zips use ZIP64 beyond 4G — handled by zipfile.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import zipfile

from config import DATA_PATH, KEYFRAME_MODE
from media.preprocess import has_overlay, preprocess
from media.scenes import read_scenes, select_frames

URLS = [
    "https://aic-data.ledo.io.vn/Videos_L21_a.zip",
    "https://aic-data.ledo.io.vn/Videos_L22_a.zip",
    "https://aic-data.ledo.io.vn/Videos_L23_a.zip",
    "https://aic-data.ledo.io.vn/Videos_L24_a.zip",
    "https://aic-data.ledo.io.vn/Videos_L25_a.zip",
    "https://aic-data.ledo.io.vn/Videos_L26_a.zip",
    "https://aic-data.ledo.io.vn/Videos_L26_b.zip",
    "https://aic-data.ledo.io.vn/Videos_L26_c.zip",
    "https://aic-data.ledo.io.vn/Videos_L26_d.zip",
    "https://aic-data.ledo.io.vn/Videos_L26_e.zip",
    "https://aic-data.ledo.io.vn/Videos_L27_a.zip",
    "https://aic-data.ledo.io.vn/Videos_L28_a.zip",
    "https://aic-data.ledo.io.vn/Videos_L29_a.zip",
    "https://aic-data.ledo.io.vn/Videos_L30_a.zip",
]

SHM_DIR = Path("/dev/shm")


class HTTPFile:
    """Read-only seekable file over HTTP Range requests (curl-backed)."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self._size = None

    def size(self) -> int:
        if self._size is None:
            r = subprocess.run(["curl", "-sI", "--max-time", "30", self.url],
                               capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if line.lower().startswith("content-length:"):
                    self._size = int(line.split(":")[1].strip())
                    break
            if self._size is None:
                raise RuntimeError(f"no content-length for {self.url}")
        return self._size

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size() + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            n = self.size() - self.pos
        if n <= 0:
            return b""
        start, end = self.pos, self.pos + n - 1
        r = subprocess.run(
            ["curl", "-s", "-r", f"{start}-{end}", "--max-time", "120",
             self.url],
            capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"curl failed for {self.url}")
        self.pos += len(r.stdout)
        return r.stdout

    def close(self) -> None:
        pass


def _entry_data_range(url: str, info: zipfile.ZipInfo) -> tuple[int, int]:
    """(start, end) of the stored entry data inside the remote zip."""
    hdr = subprocess.run(
        ["curl", "-s", "-r", f"{info.header_offset}-{info.header_offset + 99}",
         "--max-time", "30", url],
        capture_output=True,
    ).stdout
    if hdr[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {info.filename}")
    name_len, extra_len = _unpack_26(hdr)
    start = info.header_offset + 30 + name_len + extra_len
    return start, start + info.file_size - 1


def _unpack_26(hdr: bytes) -> tuple[int, int]:
    import struct
    return struct.unpack_from("<2H", hdr, 26)


def _stream_one_video(url: str, info: zipfile.ZipInfo, out_dir: Path,
                      mode: str) -> None:
    video_id = Path(info.filename).stem
    scenes_file = DATA_PATH / "staging" / f"{video_id}.scenes.txt"
    if not scenes_file.exists():
        print(f"[{video_id}] no scene file, skipping", flush=True)
        return
    video_out = out_dir / video_id
    if video_out.is_dir() and any(video_out.iterdir()):
        print(f"[{video_id}] already extracted, skipping", flush=True)
        return

    spans = read_scenes(scenes_file)
    frames = [f for _, f in select_frames(spans, mode)]
    select = "+".join(f"eq(n\\,{f})" for f in frames)

    clean_overlay = has_overlay(DATA_PATH / "media-info" / f"{video_id}.json")

    start, end = _entry_data_range(url, info)
    mp4 = SHM_DIR / f"{video_id}.mp4"
    try:
        r = subprocess.run(
            ["curl", "-s", "-r", f"{start}-{end}", "--max-time", "600",
             "-o", str(mp4), url],
            capture_output=True,
        )
        if r.returncode != 0 or mp4.stat().st_size != info.file_size:
            raise RuntimeError(f"fetch failed: got {mp4.stat().st_size}, "
                               f"expected {info.file_size}")

        tmp_out = SHM_DIR / f"kf_{video_id}"
        tmp_out.mkdir(exist_ok=True)
        r = subprocess.run(
            ["ffmpeg", "-i", str(mp4), "-vf", f"select='{select}'", "-vsync",
             "0", "-an", "-q:v", "2", "-y", "-loglevel", "error",
             str(tmp_out / "frame_%05d.jpg")],
            capture_output=True, text=True,
        )
        produced = sorted(tmp_out.glob("frame_*.jpg"))
        if len(produced) != len(frames):
            raise RuntimeError(f"ffmpeg produced {len(produced)} frames, "
                               f"expected {len(frames)}: {r.stderr[-200:]}")

        video_out.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(produced, 1):
            dst = video_out / f"{i:04}.jpg"
            shutil.move(str(src), dst)
            if clean_overlay:
                preprocess(dst)
        print(f"[{video_id}] saved {len(produced)} keyframes", flush=True)
    finally:
        shutil.rmtree(SHM_DIR / f"kf_{video_id}", ignore_errors=True)
        mp4.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", action="append",
                        help="process only this zip URL (repeatable)")
    parser.add_argument("--out-dir", type=Path,
                        default=DATA_PATH / "keyframes")
    parser.add_argument("--mode", default=KEYFRAME_MODE,
                        choices=("left", "mid", "right"))
    args = parser.parse_args()

    urls = args.url or URLS
    for url in urls:
        print(f"== {url} ==", flush=True)
        http = HTTPFile(url)
        with zipfile.ZipFile(http) as zf:
            infos = [i for i in zf.infolist() if i.filename.endswith(".mp4")]
        print(f"  {len(infos)} videos", flush=True)
        for info in infos:
            try:
                _stream_one_video(url, info, args.out_dir, args.mode)
            except Exception as e:
                print(f"[{Path(info.filename).stem}] FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
