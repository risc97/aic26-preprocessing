from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

from tqdm import tqdm

from media.keyframes import list_keyframe_videos
from pipeline.ocr import (CrossCheckReader, OCRReader, PPOCRReader,
                          MODEL_CHOICES, MODEL_NAME, ocr_video)
from config import KEYFRAMES_DIR, OCR_DIR

REC_CHOICES = ("vietocr", "ppocr")

_READER = None


def _init_worker(model: str, device: str, batch_size: int, rec_thresh: float,
                 rec: str, cross_check: bool) -> None:
    """Pool initializer: every worker process loads its own models."""
    global _READER
    if rec == "ppocr":
        _READER = PPOCRReader(model=model, device=device, batch_size=batch_size,
                              rec_thresh=rec_thresh)
    elif cross_check:
        _READER = CrossCheckReader(model=model, device=device,
                                   batch_size=batch_size, rec_thresh=rec_thresh)
    else:
        _READER = OCRReader(model=model, device=device, batch_size=batch_size,
                            rec_thresh=rec_thresh)


def _ocr_one(video_id: str, keyframes_dir: Path, ocr_dir: Path, force: bool,
             batch_size: int) -> tuple[str, bool]:
    try:
        return video_id, ocr_video(_READER, video_id, keyframes_dir, ocr_dir,
                                   force, batch_size=batch_size)
    except Exception as e:
        print(f"[{video_id}] FAILED: {e}", flush=True)
        return video_id, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframes-dir", type=Path, default=KEYFRAMES_DIR)
    parser.add_argument("--ocr-dir", type=Path, default=OCR_DIR)
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        choices=list(MODEL_CHOICES),
                        help="PP-OCRv6 detector tier (default: "
                             f"{MODEL_NAME})")
    parser.add_argument("--rec", choices=REC_CHOICES, default="vietocr",
                        help="recognizer: vietocr vgg_transformer (quality, "
                             "~0.4s/frame) or PP-OCRv6's own rec (speed, "
                             "~0.05s/frame, weaker diacritics)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes, one GPU model set "
                             "each (default: 1)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="recognition batch size and frames per predict call")
    parser.add_argument("--rec-thresh", type=float, default=0.6,
                        help="drop detected lines whose recognition confidence "
                             "is below this (default: 0.6)")
    parser.add_argument("--cross-check", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="verify each line against Vintern-1B-v3.5 and "
                             "prefer its reading when both agree (vietocr rec "
                             "only; default: on)")
    parser.add_argument("--video", action="append", metavar="VIDEO_ID",
                        help="only OCR this video id (repeatable)")
    parser.add_argument("--limit", type=int,
                        help="stop after N videos (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="re-OCR videos that already have a complete OCR file")
    args = parser.parse_args()

    if args.rec == "ppocr" and args.cross_check:
        print("note: cross-check applies to the VietOCR path only; "
              "ignored with --rec ppocr")
        args.cross_check = False

    pool = mp.get_context("fork").Pool(
        args.workers, initializer=_init_worker,
        initargs=(args.model, args.device, args.batch_size, args.rec_thresh,
                  args.rec, args.cross_check))
    extra = " + Vintern cross-check" if args.cross_check else ""
    print(f"OCR workers: {args.workers} x ({args.model} + {args.rec}"
          f"{extra}) on {args.device}")

    wanted = set(args.video) if args.video else None

    video_ids = list_keyframe_videos(args.keyframes_dir)
    if wanted is not None:
        video_ids = [v for v in video_ids if v in wanted]
    if args.limit:
        video_ids = video_ids[:args.limit]
    if not video_ids:
        print(f"No keyframes to OCR in {args.keyframes_dir}")
        return 1

    done: set[str] = set()
    failed: set[str] = set()
    jobs: dict[str, object] = {}

    bar = tqdm(total=len(video_ids), desc="clips", unit="clip", mininterval=0.5)

    try:
        for video_id in video_ids:
            jobs[video_id] = pool.apply_async(
                _ocr_one, (video_id, args.keyframes_dir, args.ocr_dir,
                           args.force, args.batch_size))

        while jobs:
            for video_id in list(jobs):
                if not jobs[video_id].ready():
                    continue
                _, ok = jobs.pop(video_id).get()
                (done if ok else failed).add(video_id)
                bar.update(1)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pool.terminate()
        print(f"\ninterrupted: {len(done)} done, {len(failed)} failed")
    finally:
        bar.close()
        pool.terminate()
        pool.join()

    if failed:
        print(f"\n{len(failed)} video(s) failed: {', '.join(sorted(failed))}")
        return 1
    return 0


if __name__ == "__main__":
    main()
