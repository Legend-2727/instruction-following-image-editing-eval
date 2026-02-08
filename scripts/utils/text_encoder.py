"""text_encoder.py — Text-encoding interface with cross-lingual readiness.

Provides an abstract ``TextEncoder`` base class and a concrete
``CLIPTextEncoder`` that uses open_clip.  The interface accepts a ``lang``
parameter so we can later swap in multilingual encoders or translation
pipelines without changing calling code.
"""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Abstract interface ───────────────────────────────────────────────────────
class TextEncoder(ABC):
    """Encode a batch of texts into dense vectors."""

    @abstractmethod
    def encode(self, texts: List[str], lang: str = "en") -> np.ndarray:
        """Return an (N, D) float32 array of L2-normalised embeddings."""
        ...


# ── CLIP-based encoder (checkpoint 1 default) ───────────────────────────────
class CLIPTextEncoder(TextEncoder):
    """Thin wrapper around an ``open_clip`` text encoder.

    For checkpoint 1 this ignores ``lang`` (logs a warning for non-English).
    In later checkpoints, swap in a multilingual backbone or a translation
    step before encoding.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
    ):
        import open_clip
        import torch

        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._torch = torch

    def encode(self, texts: List[str], lang: str = "en") -> np.ndarray:
        """Tokenize and encode *texts*. Non-English *lang* is accepted but
        currently just logged as a warning (raw text is still passed to CLIP).
        """
        if lang != "en":
            logger.warning(
                "CLIPTextEncoder: lang='%s' received — multilingual encoding "
                "not yet supported; encoding raw text with CLIP.", lang
            )
        tokens = self.tokenizer(texts).to(self.device)
        with self._torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


# ── Translation helpers ─────────────────────────────────────────────────────
def load_translations(csv_path: str | Path) -> Dict[str, Dict[str, str]]:
    """Load ``translations.csv`` → ``{sample_id: {lang: prompt}}``.

    Expected CSV columns: ``id, lang, prompt``.
    """
    mapping: Dict[str, Dict[str, str]] = {}
    path = Path(csv_path)
    if not path.exists():
        logger.info("No translations file at %s — skipping.", path)
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["id"]
            lang = row["lang"]
            prompt = row["prompt"]
            mapping.setdefault(sid, {})[lang] = prompt
    logger.info("Loaded translations for %d samples from %s", len(mapping), path)
    return mapping
