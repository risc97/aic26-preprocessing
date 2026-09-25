from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
MAX_LENGTH = 8192


class Gte:
    EMBED_DIM = 768

    def __init__(self, device: str = "cuda", amp: bool = True):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self._restore_buffers()
        self.model.to(self.device).eval()
        self.amp = amp and self.device.type == "cuda"

    @torch.no_grad()
    def _restore_buffers(self):
        # newer transformers inits on meta and skips non-persistent buffers
        # (position_ids, rotary caches), leaving them as garbage -> OOB index
        emb = self.model.embeddings
        fresh = type(emb)(self.model.config)
        for name, buf in fresh.named_buffers():
            emb.get_buffer(name).copy_(buf)


    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(
            texts, max_length=MAX_LENGTH, padding=True,
            truncation=True, return_tensors="pt",
        ).to(self.device)
        with self._autocast():
            embeddings = self.model(**tokens).last_hidden_state[:, 0]
        return F.normalize(embeddings.float(), dim=-1).cpu().numpy()
