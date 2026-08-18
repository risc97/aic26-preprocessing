from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch  # noqa: F401  # import before paddle: paddle injects its own libs
# into the loader path, so torch must resolve its CUDA libs first
from PIL import Image
from paddleocr import PaddleOCR, TextDetection
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

from config import CKPT_PATH

# skip paddle's connectivity check at startup; the models are already cached
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

MODEL_NAME = "PP-OCRv6_medium"
MODEL_CHOICES = ("PP-OCRv6_tiny", "PP-OCRv6_small", "PP-OCRv6_medium")

# VietOCR weights live next to the other checkpoints (persistent volume)
REC_WEIGHTS_DIR = CKPT_PATH.parent
CROP_PAD = 10


class OCRReader:
    """OCR: PP-OCRv6 detection + VietOCR recognition.

    PP-OCRv6's detector is language-agnostic and locates the text boxes;
    VietOCR recognizes each cropped line, which handles Vietnamese
    diacritics far better than the bundled recognizer. Lines are joined
    with newlines in reading order (top-to-bottom, left-to-right); lines
    below the confidence threshold are dropped.

    rec_name selects the VietOCR head: vgg_seq2seq (default, ~80ms/line,
    attention decoder) or vgg_transformer (~330ms/line, slightly more
    accurate on long lines).
    """

    def __init__(self, model: str = MODEL_NAME, device: str = "cuda",
                 batch_size: int = 16, rec_thresh: float = 0.6,
                 rec_name: str = "vgg_seq2seq"):
        self.det = TextDetection(model_name=f"{model}_det",
                                 device="gpu" if device.startswith("cuda") else "cpu")
        config = Cfg.load_config_from_name(rec_name)
        config["device"] = device
        config["batch_size"] = batch_size
        config["weights"] = str(_ensure_rec_weights(config["weights"], rec_name))
        self.predictor = Predictor(config)
        self.rec_thresh = rec_thresh

    def read_text(self, image_path: Path) -> str:
        return self.read_texts([image_path])[0]

    def read_texts(self, image_paths: list[Path]) -> list[str]:
        det_results = self.det.predict([str(p) for p in image_paths])
        texts: list[str] = []
        for path, res in zip(image_paths, det_results):
            polys = res["dt_polys"]
            if not len(polys):
                texts.append("")
                continue
            img = cv2.imread(str(path))
            order = sorted(range(len(polys)),
                           key=lambda i: (polys[i][:, 1].mean(), polys[i][:, 0].mean()))
            crops = [_crop_box(img, polys[i]) for i in order]
            lines, probs = self.predictor.predict_batch(crops, return_prob=True)
            texts.append("\n".join(t for t, p in zip(lines, probs)
                                   if p >= self.rec_thresh))
        return texts


class PPOCRReader:
    """Pure PP-OCRv6 (detection + bundled recognition).

    About 10x faster than the VietOCR hybrid (~0.05s vs ~0.4s per frame)
    at the cost of somewhat weaker Vietnamese diacritics. Lines are
    joined with newlines in reading order; lines below the confidence
    threshold are dropped.
    """

    def __init__(self, model: str = MODEL_NAME, device: str = "cuda",
                 batch_size: int = 16, rec_thresh: float = 0.6):
        self.ocr = PaddleOCR(
            text_detection_model_name=f"{model}_det",
            text_recognition_model_name=f"{model}_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu" if device.startswith("cuda") else "cpu",
            text_recognition_batch_size=batch_size,
        )
        self.rec_thresh = rec_thresh

    def read_text(self, image_path: Path) -> str:
        return self.read_texts([image_path])[0]

    def read_texts(self, image_paths: list[Path]) -> list[str]:
        results = self.ocr.predict([str(p) for p in image_paths])
        texts: list[str] = []
        for r in results:
            lines = [t for t, s in zip(r["rec_texts"], r["rec_scores"])
                     if s >= self.rec_thresh]
            texts.append("\n".join(lines))
        return texts


def _ensure_rec_weights(weights_url: str, rec_name: str) -> Path:
    """VietOCR downloads its weights to /tmp by default; pin them next to
    the other checkpoints so they survive reboots and recycles."""
    from vietocr.tool.utils import download_weights
    local = REC_WEIGHTS_DIR / f"{rec_name}.pth"
    if not local.is_file():
        tmp = download_weights(weights_url)
        local.parent.mkdir(parents=True, exist_ok=True)
        # move (not replace): /tmp and the workspace may be on different
        # filesystems, where os.replace fails with a cross-device error
        shutil.move(tmp, local)
    return local


def _crop_box(img: np.ndarray, poly: np.ndarray, pad: int = CROP_PAD) -> Image.Image:
    """Perspective-warp the quad into an axis-aligned crop with a margin.

    Straightening the box and padding it a few pixels on each side makes
    VietOCR read first/last characters much more reliably.
    """
    poly = poly.astype(np.float32)
    w = max(np.linalg.norm(poly[1] - poly[0]), np.linalg.norm(poly[2] - poly[3]))
    h = max(np.linalg.norm(poly[3] - poly[0]), np.linalg.norm(poly[2] - poly[1]))
    w, h = int(w) + 2 * pad, int(h) + 2 * pad
    dst = np.array([[pad, pad], [w - pad, pad], [w - pad, h - pad], [pad, h - pad]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(poly, dst)
    warped = cv2.warpPerspective(img, m, (w, h))
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
