"""clip_encoder.py — CLIP image & text embedding extractor.

Uses ``open_clip`` with a small ViT-B-32 backbone so everything fits
comfortably on an RTX 3060 (~600 MB VRAM at FP32).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """Batch encoder for images and texts using open_clip."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ):
        import open_clip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("Loading CLIP model %s (%s) on %s …", model_name, pretrained, device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        logger.info("CLIP model ready.")

    # ── images ───────────────────────────────────────────────────────────
    def encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        desc: str = "Encoding images",
    ) -> np.ndarray:
        """Return (N, D) float32 L2-normalised image embeddings."""
        all_feats: List[np.ndarray] = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc=desc):
            batch_paths = image_paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(self.preprocess(img))
            batch_tensor = torch.stack(imgs).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0).astype(np.float32)

    # ── texts ────────────────────────────────────────────────────────────
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        desc: str = "Encoding texts",
    ) -> np.ndarray:
        """Return (N, D) float32 L2-normalised text embeddings."""
        all_feats: List[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[i : i + batch_size]
            tokens = self.tokenizer(batch_texts).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0).astype(np.float32)
