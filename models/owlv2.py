from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_convert, nms
from transformers import Owlv2Processor, Owlv2ForObjectDetection

MODEL_REPOS = {
    "owlv2-base": "google/owlv2-base-patch16-ensemble",
    "owlv2-large": "google/owlv2-large-patch14-ensemble",
}
MODEL_NAME = "owlv2-base"
MODEL_CHOICES = tuple(MODEL_REPOS)
PAD_OBJECTNESS = -1e4

class Owlv2Detector:
    def __init__(self, model: str = MODEL_NAME, device: str = "cuda",
                 top_k: int = 32, pool: int = 300, nms_iou: float = 0.7,
                 amp: bool = True):
        self.model_name = model
        repo = MODEL_REPOS.get(model, model)
        self.processor = Owlv2Processor.from_pretrained(repo)
        self.model = Owlv2ForObjectDetection.from_pretrained(repo, attn_implementation="sdpa")
        self.model.to(device).eval()
        self.device = torch.device(device)
        # fp32 weights + autocast
        self.amp = amp and self.device.type == "cuda"
        self.top_k, self.pool, self.nms_iou = top_k, pool, nms_iou
        self.embed_dim = self.model.class_head.dense0.out_features

    def preprocess(self, image):
        return self.processor.image_processor(
            images=image, return_tensors="pt")["pixel_values"][0]

    @torch.inference_mode()
    def detect(self, pixel_values: torch.Tensor, sizes: torch.Tensor):
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        # forward pass through vision towers and heads
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            # extract image grid tokens from vision backbone
            feature_map = self.model.image_embedder(pixel_values=pixel_values)[0]
            b, h, w, d = feature_map.shape
            feats = feature_map.reshape(b, h * w, d)

            # predict bounding boxes and objectness
            pred_boxes = self.model.box_predictor(feats, feature_map)
            objectness = self.model.objectness_predictor(feats)

            # class embeddings, logit shifts, and positive scale factors
            head = self.model.class_head
            class_embeds = head.dense0(feats)
            shift = head.logit_shift(feats)
            scale = F.elu(head.logit_scale(feats)) + 1

        class_embeds = class_embeds.float()
        class_embeds = class_embeds / (class_embeds.norm(dim=-1, keepdim=True) + 1e-6)
        boxes = box_convert(pred_boxes.float(), "cxcywh", "xyxy").clamp(0, 1)
        objectness = objectness.float()
        aux = torch.cat([shift.float(), scale.float(), objectness[..., None]], dim=-1)

        K, N = self.top_k, feats.shape[1]
        out_e = torch.zeros(b, K, class_embeds.shape[-1])
        out_b = torch.zeros(b, K, 4)
        out_a = torch.full((b, K, 3), PAD_OBJECTNESS)

        for i in range(b):
            # filter top objectness and apply NMS to select top-k
            cand = objectness[i].topk(min(self.pool, N)).indices
            keep = nms(boxes[i][cand], objectness[i][cand], self.nms_iou)[:K]
            idx = cand[keep]
            n = len(idx)
            out_e[i, :n] = class_embeds[i][idx]
            out_a[i, :n] = aux[i][idx]

            # rescale normalized square boxes back to original ratio
            iw, ih = float(sizes[i, 0]), float(sizes[i, 1])
            s = max(iw, ih)
            box = boxes[i][idx].clone()
            box[:, [0, 2]] *= s / iw
            box[:, [1, 3]] *= s / ih
            out_b[i, :n] = box.clamp(0, 1)

        return (out_e.numpy().astype(np.float16),
                out_b.numpy().astype(np.float16),
                out_a.numpy().astype(np.float16))
