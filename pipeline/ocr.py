from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

from tqdm import tqdm

from media.keyframes import find_keyframes
from pipeline.ocr_model import (MODEL_CHOICES, MODEL_NAME, OCRReader,
                                PPOCRReader, VinternReader)

__all__ = ["OCRReader", "PPOCRReader", "VinternReader", "CrossCheckReader",
           "MODEL_NAME", "MODEL_CHOICES", "ocr_video", "ocr_path",
           "check_complete_ocr", "reconcile_lines"]

_D_TRANSLATE = str.maketrans({"đ": "d", "Đ": "d"})

MAX_WORKER_RESTARTS = 3  # total Vintern worker restarts before giving up


def _base_form(word: str) -> str:
    """Lowercase word with Vietnamese diacritics stripped: CHỦ Y -> chu y."""
    word = word.translate(_D_TRANSLATE).lower()
    return "".join(c for c in unicodedata.normalize("NFD", word)
                   if not unicodedata.combining(c))


def _find_run(haystack: list[str], needle: list[str]) -> int | None:
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


def reconcile_lines(lines: list[str], vlm_text: str) -> list[str]:
    """Replace hybrid lines with the VLM's reading where both agree.

    The VLM reading wins only when a line's words appear verbatim in the
    VLM's text modulo case and diacritics; a reading that differs in its
    base words (e.g. a VLM hallucination) leaves the literal line alone.
    Matched lines keep the hybrid's all-caps style when it had one.
    """
    vlm_words = re.findall(r"\w+", vlm_text)
    vlm_base = [_base_form(w) for w in vlm_words]
    out: list[str] = []
    for line in lines:
        words = re.findall(r"\w+", line)
        if not words:
            out.append(line)
            continue
        idx = _find_run(vlm_base, [_base_form(w) for w in words])
        if idx is None:
            out.append(line)  # no agreement: keep the literal reading
            continue
        span = " ".join(vlm_words[idx:idx + len(words)])
        out.append(span.upper() if line.isupper() else span)
    return out


class CrossCheckReader:
    """Hybrid OCR anchored by a VLM cross-check.

    The hybrid (PP-OCRv6 det + VietOCR) is literal and cannot invent
    text; Vintern-1B-v3.5 has far better diacritics but occasionally
    hallucinates plausible-looking words. reconcile_lines() upgrades a
    line to the VLM's reading only when both agree on every base word,
    which fixes diacritic slips without ever admitting hallucinations.

    Vintern runs in a subprocess so the heavy VLM stack doesn't share an
    address space with the parent's paddle/VietOCR inference.
    """

    def __init__(self, model: str = MODEL_NAME, device: str = "cuda",
                 batch_size: int = 16, rec_thresh: float = 0.6):
        import atexit

        self.base = OCRReader(model=model, device=device, batch_size=batch_size,
                              rec_thresh=rec_thresh)
        self._worker_device = device
        self._worker = subprocess.Popen(
            [sys.executable, "-m", "pipeline.ocr_model.vintern",
             "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._degraded = False
        self._restarts = 0
        atexit.register(self.close)

    def read_text(self, image_path: Path) -> str:
        return self.read_texts([image_path])[0]

    def read_texts(self, image_paths: list[Path]) -> list[str]:
        base_texts = self.base.read_texts(image_paths)
        if self._degraded:
            return base_texts
        # only frames with detected text can gain from the cross-check;
        # skipping empty frames roughly halves the Vintern pass
        idxs = [i for i, t in enumerate(base_texts) if t.strip()]
        vlm_texts = [""] * len(image_paths)
        if idxs:
            sub_texts = self._vlm_read_texts([image_paths[i] for i in idxs])
            if sub_texts is None:
                print("WARNING: Vintern cross-check is degraded; using literal "
                      "OCR only.", flush=True)
                self._degraded = True
                return base_texts
            for i, t in zip(idxs, sub_texts):
                vlm_texts[i] = t
        return ["\n".join(reconcile_lines(text.split("\n"), vlm))
                for text, vlm in zip(base_texts, vlm_texts)]

    def _vlm_read_texts(self, image_paths: list[Path]) -> list[str] | None:
        """One request over the worker's JSON-lines protocol.

        A worker whose Vintern degrades (single-char output) is killed
        and replaced: each fresh process is an independent roll of the
        window's dice. Restarts are capped per run so a fully cursed
        window costs bounded time before giving up on upgrades.
        """
        for _ in range(2):
            try:
                req = json.dumps({"frames": [str(p) for p in image_paths]})
                self._worker.stdin.write(req + "\n")
                self._worker.stdin.flush()
                line = self._worker.stdout.readline()
                if not line:
                    raise BrokenPipeError("worker exited")
                resp = json.loads(line)
                if not resp.get("ok"):
                    raise ValueError(resp.get("error", "worker error"))
                texts = resp["texts"]
                if sum(len(t) <= 5 for t in texts) > len(texts) // 2:
                    raise ValueError("degenerate output")
                return texts
            except (BrokenPipeError, OSError, ValueError) as e:
                if self._restarts >= MAX_WORKER_RESTARTS:
                    print(f"[vintern worker] failed ({e}); giving up on "
                          f"cross-check", flush=True)
                    return None
                print(f"[vintern worker] failed ({e}); restarting...", flush=True)
                self._restarts += 1
                self._restart_worker()
        return None

    def _restart_worker(self) -> None:
        if self._worker.poll() is None:
            self._worker.terminate()
            try:
                self._worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._worker.kill()
        self._worker = subprocess.Popen(
            [sys.executable, "-m", "pipeline.ocr_model.vintern",
             "--device", self._worker_device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def close(self) -> None:
        """Terminate the VLM worker (call before exit, or atexit)."""
        if self._worker.poll() is None:
            self._worker.terminate()
            try:
                self._worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._worker.kill()


def ocr_path(ocr_dir: Path, video_id: str) -> Path:
    # <ocr_dir>/<video_id>.json   {frame_stem: text}
    return ocr_dir / f"{video_id}.json"


def check_complete_ocr(ocr_dir: Path, video_id: str, keyframes: list[Path]) -> bool:
    path = ocr_path(ocr_dir, video_id)
    if not path.is_file():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            texts = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return set(texts) == {p.stem for p in keyframes}


def ocr_video(reader: OCRReader, video_id: str, keyframes_dir: Path,
              ocr_dir: Path, force: bool, batch_size: int = 16) -> bool:
    """OCR one video's keyframes into a JSON file"""
    keyframes = find_keyframes(keyframes_dir, video_id)
    if not keyframes:
        print(f"[{video_id}] no keyframes in {keyframes_dir / video_id}, skipping")
        return False

    if not force and check_complete_ocr(ocr_dir, video_id, keyframes):
        print(f"[{video_id}] OCR already complete ({len(keyframes)} frames), skipping")
        return True

    print(f"[{video_id}] OCR-ing {len(keyframes)} keyframes...")
    texts: dict[str, str] = {}
    bar = tqdm(total=len(keyframes), desc=f"[{video_id}]", unit="frame",
               mininterval=0.5, leave=False)
    for i in range(0, len(keyframes), batch_size):
        chunk = keyframes[i:i + batch_size]
        for frame, text in zip(chunk, reader.read_texts(chunk)):
            texts[frame.stem] = text
        bar.update(len(chunk))
    bar.close()

    path = ocr_path(ocr_dir, video_id)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    # write to .tmp then rename so a crash never leaves a broken JSON
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

    print(f"[{video_id}] saved OCR for {len(texts)} frames to {path}")
    return True
