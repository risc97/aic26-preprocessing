from __future__ import annotations

import timm
import torch

from .base import Encoder

MODEL_NAME = "convnext_base.dinov3_lvd1689m"


class DINOv3(Encoder):
    """Image-only encoder; compiled by default since it only runs as a bulk GPU embedder."""

    MODEL_NAME = MODEL_NAME
    EMBED_DIM = 1024
    IMAGE_SIZE = 224

    def __init__(self, ckpt=None, device="cuda", amp=True, compile=True):
        torch.backends.cudnn.benchmark = True
        # default mode, not base's CUDA graphs: same speed at full batches, and graphs
        # would re-record for every distinct tail-batch size
        super().__init__(ckpt, device=device, amp=amp, compile=False)
        if compile:
            self.model = torch.compile(self.model)

    def _create_model_and_transforms(self, *, pretrained: bool = False):
        model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0)
        cfg = timm.data.resolve_data_config({}, model=model)
        # squash instead of center crop so the whole 16:9 keyframe is embedded
        return model, timm.data.create_transform(**{**cfg, "crop_mode": "squash"})

    def _forward_image(self, pixel_values):
        return self.model(pixel_values)
