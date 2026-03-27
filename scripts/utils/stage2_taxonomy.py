"""Shared Stage-2 multilingual taxonomy model utilities."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel, AutoTokenizer

from .io import load_jsonl, normalize_relpath
from .schema import ERROR_TYPES, ERROR_TO_IDX


LANGS: Tuple[str, str, str] = ("en", "hi", "bn")


def _clean_text(x: Any) -> str:
    return str(x or "").strip()


def select_prompt_with_fallback(row: Dict[str, Any], requested_lang: str) -> Dict[str, Any]:
    """Select the prompt for requested language with explicit fallback metadata."""
    requested_lang = _clean_text(requested_lang).lower()
    if requested_lang not in LANGS:
        raise ValueError(f"Unsupported language: {requested_lang}. Expected one of {LANGS}.")

    prompt_by_lang = {lang: _clean_text(row.get(f"instruction_{lang}")) for lang in LANGS}
    selected = prompt_by_lang.get(requested_lang, "")
    actual_lang = requested_lang
    fallback_used = False

    if not selected:
        fallback_used = True
        for lang in LANGS:
            if prompt_by_lang[lang]:
                selected = prompt_by_lang[lang]
                actual_lang = lang
                break

    return {
        "prompt_text": selected,
        "prompt_language_requested": requested_lang,
        "prompt_language_actual": actual_lang,
        "fallback_used": fallback_used,
    }


def token_count_from_attention_mask(attention_mask: torch.Tensor) -> int:
    if attention_mask.ndim == 2:
        return int(attention_mask[0].sum().item())
    return int(attention_mask.sum().item())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def labels_to_multihot(labels: Sequence[str]) -> List[float]:
    vec = [0.0] * len(ERROR_TYPES)
    for lbl in labels:
        if lbl in ERROR_TO_IDX:
            vec[ERROR_TO_IDX[lbl]] = 1.0
    return vec


def load_original_rows(jsonl_path: str) -> List[Dict[str, Any]]:
    rows = load_jsonl(jsonl_path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        sid = str(row.get("id") or "").strip()
        if not sid:
            continue
        out.append(row)
    return out


def expand_rows_by_language(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for row in rows:
        labels = [str(x) for x in row.get("taxonomy_labels", [])]
        target = labels_to_multihot(labels)
        for lang in LANGS:
            text = str(row.get(f"instruction_{lang}") or "").strip()
            if not text:
                continue
            expanded.append(
                {
                    "id": str(row["id"]),
                    "lang": lang,
                    "instruction": text,
                    "source_path": normalize_relpath(str(row.get("source_path") or "")),
                    "target_path": normalize_relpath(str(row.get("target_path") or "")),
                    "taxonomy_labels": labels,
                    "target_vec": target,
                    "edit_type": row.get("edit_type"),
                    "source_type": row.get("source_type"),
                }
            )
    return expanded


class TriInputTaxonomyDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        image_root: str,
        tokenizer: AutoTokenizer,
        max_text_len: int = 128,
        transform: Optional[transforms.Compose] = None,
    ):
        self.rows = list(rows)
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.transform = transform or build_image_transform()

        valid: List[Dict[str, Any]] = []
        for row in self.rows:
            src = os.path.join(self.image_root, row["source_path"])
            tgt = os.path.join(self.image_root, row["target_path"])
            if not (os.path.exists(src) and os.path.exists(tgt)):
                continue
            if not str(row.get("instruction") or "").strip():
                continue
            valid.append(row)
        self.rows = valid

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        src = os.path.join(self.image_root, row["source_path"])
        tgt = os.path.join(self.image_root, row["target_path"])

        src_img = Image.open(src).convert("RGB")
        tgt_img = Image.open(tgt).convert("RGB")
        src_tensor = self.transform(src_img)
        tgt_tensor = self.transform(tgt_img)

        toks = self.tokenizer(
            row["instruction"],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "id": row["id"],
            "lang": row["lang"],
            "instruction": row["instruction"],
            "source_path": row["source_path"],
            "target_path": row["target_path"],
            "src_img": src_tensor,
            "tgt_img": tgt_tensor,
            "input_ids": toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
            "target": torch.tensor(row["target_vec"], dtype=torch.float32),
        }


class TriInputTaxonomyClassifier(nn.Module):
    def __init__(self, text_model_name: str, freeze_vision: bool = False, freeze_text: bool = False):
        super().__init__()
        self.vision = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.vision.fc = nn.Identity()
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        self.text_enc = AutoModel.from_pretrained(text_model_name)
        if freeze_text:
            for p in self.text_enc.parameters():
                p.requires_grad = False

        fused_dim = (512 * 4) + int(self.text_enc.config.hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.taxonomy_head = nn.Linear(256, len(ERROR_TYPES))

    def forward(self, src_img, tgt_img, input_ids, attention_mask) -> torch.Tensor:
        f_src = self.vision(src_img)
        f_tgt = self.vision(tgt_img)
        f_diff = f_tgt - f_src
        f_abs_diff = torch.abs(f_diff)

        text_out = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)
        f_text = text_out.last_hidden_state[:, 0, :]

        fused = torch.cat([f_src, f_tgt, f_diff, f_abs_diff, f_text], dim=1)
        hidden = self.fusion(fused)
        return self.taxonomy_head(hidden)


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal = (1.0 - p_t).pow(self.gamma)
        loss = focal * bce
        if self.alpha is not None:
            loss = loss * self.alpha.view(1, -1)
        return loss.mean()


def compute_pos_weight(train_rows_expanded: Sequence[Dict[str, Any]]) -> torch.Tensor:
    arr = np.array([r["target_vec"] for r in train_rows_expanded], dtype=np.float32)
    if arr.size == 0:
        return torch.ones(len(ERROR_TYPES), dtype=torch.float32)
    pos = arr.sum(axis=0)
    neg = arr.shape[0] - pos
    # Avoid divide by zero; capped to keep optimization stable.
    pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    pos_weight = np.clip(pos_weight, 1.0, 20.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> List[float]:
    thresholds: List[float] = []
    grid = np.linspace(0.1, 0.9, 17)
    for c in range(y_true.shape[1]):
        best_t = 0.5
        best_f1 = -1.0
        yt = y_true[:, c]
        if yt.sum() == 0:
            thresholds.append(0.95)
            continue
        for t in grid:
            yp = (y_prob[:, c] >= t).astype(np.int32)
            f1 = f1_score(yt, yp, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds.append(best_t)
    return thresholds


def binarize_with_thresholds(y_prob: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    th = np.array(list(thresholds), dtype=np.float32).reshape(1, -1)
    return (y_prob >= th).astype(np.int32)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        metrics["mAP_macro"] = float(average_precision_score(y_true, y_prob, average="macro"))
    except ValueError:
        metrics["mAP_macro"] = 0.0
    try:
        metrics["mAP_micro"] = float(average_precision_score(y_true, y_prob, average="micro"))
    except ValueError:
        metrics["mAP_micro"] = 0.0
    return metrics


def load_stage1_compatible_weights(model: nn.Module, checkpoint_path: Optional[str]) -> Dict[str, Any]:
    if not checkpoint_path:
        return {"loaded": False, "reason": "no_checkpoint"}

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        return {"loaded": False, "reason": "not_found", "path": str(ckpt)}

    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state_dict = state["state_dict"]
    elif isinstance(state, dict):
        state_dict = state
    else:
        return {"loaded": False, "reason": "invalid_checkpoint_format", "path": str(ckpt)}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "loaded": True,
        "path": str(ckpt),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class EvalOutput:
    loss: float
    y_true: np.ndarray
    y_prob: np.ndarray
    ids: List[str]
    langs: List[str]
    source_paths: List[str]
    target_paths: List[str]
    instructions: List[str]
