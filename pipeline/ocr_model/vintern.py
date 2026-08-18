from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "5CD-AI/Vintern-1B-v3_5"
PROMPT = "<image>\nĐọc toàn bộ văn bản trong ảnh, chỉ trả về văn bản."
GENERATION_CONFIG = dict(max_new_tokens=1024, do_sample=False, num_beams=1)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VinternReader:
    """Vintern-1B-v3.5 (Vietnamese VLM) as a diacritic cross-check.

    Reads one image per chat() call: in multi-image prompts the model
    reads the text but mis-attributes it to the wrong image numbers, and
    compiled per-frame runs cost no more than a batch anyway. Note the
    model can hallucinate plausible-looking words; pipeline.ocr's
    reconcile_lines() only accepts its reading where the literal hybrid
    OCR agrees on every base word.
    """

    def __init__(self, model: str = MODEL_NAME, device: str = "cuda",
                 attn_impl: str = "sdpa", use_flash_attn: bool = True):
        """attn_impl: 'sdpa' (torch-native flash via cuDNN, default; ~26%
        faster LM decode than eager), 'eager', or 'flash_attention_2'
        (needs the flash-attn package). use_flash_attn switches the vision
        tower to flash-attn kernels when the package is installed (67 ms
        vs 109 ms per frame; degrades to naive attention without it)."""
        self.device = device
        if attn_impl is None:
            self.model = AutoModel.from_pretrained(
                model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                trust_remote_code=True, use_flash_attn=use_flash_attn
            ).eval().to(device)
        else:
            # build the LM ourselves with the chosen attention impl, then
            # let InternVLChatModel load its checkpoint around it (its
            # __init__ only supports eager/flash_attention_2 for the LM)
            from transformers import AutoConfig, Qwen2ForCausalLM

            cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
            lm_cfg = cfg.llm_config
            lm_cfg._attn_implementation = attn_impl
            lm = Qwen2ForCausalLM(lm_cfg).to(torch.bfloat16)
            self.model = AutoModel.from_pretrained(
                model, language_model=lm, torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True, trust_remote_code=True,
                use_flash_attn=use_flash_attn).eval().to(device)
        # torch.compile shaves ~35% off the LM's per-token decode overhead
        if device.startswith("cuda"):
            try:
                self.model.language_model = torch.compile(
                    self.model.language_model, mode="reduce-overhead")
            except Exception as e:
                print(f"[vintern] torch.compile unavailable, using eager "
                      f"({e})", file=sys.stderr, flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True,
                                                       use_fast=False)

    def read_text(self, image_path: Path) -> str:
        return self.read_texts([image_path])[0]

    def read_texts(self, image_paths: list[Path]) -> list[str]:
        texts: list[str] = []
        for path in image_paths:
            pixel_values = _load_image(path).to(torch.bfloat16).to(self.device)
            text = self.model.chat(self.tokenizer, pixel_values,
                                   PROMPT, GENERATION_CONFIG)
            if len(text.strip()) <= 5:
                # the model occasionally stops after a word fragment; retry
                # once. Even if the retry fails, the cross-check keeps the
                # literal reading.
                torch.cuda.empty_cache()
                text = self.model.chat(self.tokenizer, pixel_values,
                                       PROMPT, GENERATION_CONFIG)
            texts.append(text)
        return texts


def serve(device: str = "cuda") -> None:
    """Worker mode: OCR batches over a newline-JSON protocol on stdio.

    Vintern lives in its own process so the heavy VLM stack doesn't share
    an address space with the parent's paddle/VietOCR inference.
    Requests: {"frames": ["path", ...]} -> {"ok": true, "texts": [...]}.
    stdout carries protocol lines only; everything else goes to stderr.
    """
    # the remote code prints model-load chatter ("FlashAttention2 is not
    # installed.") to stdout; keep it off the protocol by sending stdout
    # to stderr until the protocol loop owns it
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    reader = VinternReader(device=device)
    sys.stdout = protocol_out
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            frames = json.loads(line)["frames"]
            texts = reader.read_texts([Path(p) for p in frames])
            resp = {"ok": True, "texts": texts}
        except Exception as e:  # never die mid-protocol
            resp = {"ok": False, "error": str(e)}
        print(json.dumps(resp, ensure_ascii=False), flush=True)


def _load_image(image_file: Path, input_size: int = 448, max_num: int = 6) -> torch.Tensor:
    """Standard InternVL2.5 tiling: 448x448 tiles + thumbnail, ImageNet norm.

    The model repo does not bundle this helper (its README assumes you
    bring it), so it lives here.
    """
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff, best_ratio, area = float("inf"), (1, 1), width * height
        for ratio in target_ratios:
            ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
            if ratio_diff < best_ratio_diff:
                best_ratio_diff, best_ratio = ratio_diff, ratio
            elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = sorted(
            {(i, j) for n in range(min_num, max_num + 1)
             for i in range(1, n + 1) for j in range(1, n + 1)
             if i * j <= max_num and i * j >= min_num},
            key=lambda x: x[0] * x[1])
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = ((i % (target_width // image_size)) * image_size,
                   (i // (target_width // image_size)) * image_size,
                   ((i % (target_width // image_size)) + 1) * image_size,
                   ((i // (target_width // image_size)) + 1) * image_size)
            processed_images.append(resized_img.crop(box))
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    image = Image.open(image_file).convert("RGB")
    transform = T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    images = dynamic_preprocess(image, image_size=input_size,
                                use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vintern OCR worker (JSON lines on stdio)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    serve(device=args.device)
