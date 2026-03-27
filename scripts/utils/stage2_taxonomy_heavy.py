"""Shared utilities for the heavy Stage-2 multilingual taxonomy model."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16
from transformers import AutoModel, AutoTokenizer

from .io import load_jsonl, normalize_relpath
from .schema import ERROR_TO_IDX, ERROR_TYPES

LANGS: Tuple[str, str, str] = ("en", "hi", "bn")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _clean_text(x: Any) -> str:
    return str(x or "").strip()


def select_prompt_with_fallback(row: Dict[str, Any], requested_lang: str) -> Dict[str, Any]:
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
            text = _clean_text(row.get(f"instruction_{lang}"))
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


class TriInputTaxonomyHeavyDataset(Dataset):
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
            if not _clean_text(row.get("instruction")):
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


def build_vision_encoder(backbone: str) -> Tuple[nn.Module, int]:
    backbone = backbone.lower()
    if backbone == "vit_b16":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_dim = model.heads.head.in_features
        model.heads = nn.Identity()
        return model, int(in_dim)
    if backbone == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, int(in_dim)
    raise ValueError(f"Unsupported vision backbone: {backbone}")


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.ln(x))


class TriInputTaxonomyHeavyClassifier(nn.Module):
    def __init__(
        self,
        vision_backbone: str = "vit_b16",
        text_model_name: str = "xlm-roberta-base",
        freeze_vision: bool = False,
        freeze_text: bool = False,
        fusion_dim: int = 1024,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vision, vision_dim = build_vision_encoder(vision_backbone)
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        self.text_enc = AutoModel.from_pretrained(text_model_name)
        text_dim = int(self.text_enc.config.hidden_size)
        if freeze_text:
            for p in self.text_enc.parameters():
                p.requires_grad = False

        image_feats_dim = vision_dim * 4
        self.image_proj = nn.Sequential(
            nn.LayerNorm(image_feats_dim),
            nn.Linear(image_feats_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_gate = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, fusion_dim),
            nn.Sigmoid(),
        )

        fusion_in_dim = fusion_dim * 4
        self.fusion_in = nn.Sequential(
            nn.LayerNorm(fusion_in_dim),
            nn.Linear(fusion_in_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_blocks = nn.Sequential(
            ResidualMLPBlock(fusion_dim, dropout=dropout),
            ResidualMLPBlock(fusion_dim, dropout=dropout),
        )
        self.taxonomy_head = nn.Linear(fusion_dim, len(ERROR_TYPES))

    def forward(self, src_img, tgt_img, input_ids, attention_mask) -> torch.Tensor:
        f_src = self.vision(src_img)
        f_tgt = self.vision(tgt_img)
        f_diff = f_tgt - f_src
        f_abs_diff = torch.abs(f_diff)

        text_out = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)
        f_text = text_out.last_hidden_state[:, 0, :]

        image_feats = torch.cat([f_src, f_tgt, f_diff, f_abs_diff], dim=1)
        image_proj = self.image_proj(image_feats)
        text_proj = self.text_proj(f_text)
        gate = self.text_gate(f_text)
        gated_img = image_proj * gate

        fused = torch.cat([image_proj, text_proj, gated_img, torch.abs(image_proj - text_proj)], dim=1)
        hidden = self.fusion_in(fused)
        hidden = self.fusion_blocks(hidden)
        return self.taxonomy_head(hidden)

    def set_vision_trainable(self, trainable: bool) -> None:
        for p in self.vision.parameters():
            p.requires_grad = bool(trainable)

    def set_head_trainable(self, trainable: bool = True) -> None:
        modules = [self.image_proj, self.text_proj, self.text_gate, self.fusion_in, self.fusion_blocks, self.taxonomy_head]
        for module in modules:
            for p in module.parameters():
                p.requires_grad = bool(trainable)

    def configure_text_trainability(self, freeze_text: bool, unfreeze_last_n_layers: int = -1) -> Dict[str, Any]:
        for p in self.text_enc.parameters():
            p.requires_grad = False

        if freeze_text:
            return {
                "freeze_text": True,
                "unfreeze_last_n_layers": 0,
                "trainable_text_param_count": 0,
            }

        # Start with embedding/output blocks frozen, then optionally unfreeze top-N encoder layers.
        trainable = 0
        layers = None
        if hasattr(self.text_enc, "encoder") and hasattr(self.text_enc.encoder, "layer"):
            layers = self.text_enc.encoder.layer

        if layers is None:
            for p in self.text_enc.parameters():
                p.requires_grad = True
            trainable = int(sum(p.numel() for p in self.text_enc.parameters() if p.requires_grad))
            return {
                "freeze_text": False,
                "unfreeze_last_n_layers": -1,
                "trainable_text_param_count": trainable,
            }

        n_layers = len(layers)
        if unfreeze_last_n_layers is None or unfreeze_last_n_layers < 0:
            selected_layers = list(range(n_layers))
        else:
            k = max(0, min(int(unfreeze_last_n_layers), n_layers))
            selected_layers = list(range(n_layers - k, n_layers)) if k > 0 else []

        for idx in selected_layers:
            for p in layers[idx].parameters():
                p.requires_grad = True

        trainable = int(sum(p.numel() for p in self.text_enc.parameters() if p.requires_grad))
        return {
            "freeze_text": False,
            "unfreeze_last_n_layers": int(len(selected_layers)),
            "selected_text_layer_indices": selected_layers,
            "trainable_text_param_count": trainable,
        }


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
    pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    pos_weight = np.clip(pos_weight, 1.0, 30.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def build_sample_weights(rows: Sequence[Dict[str, Any]]) -> torch.Tensor:
    arr = np.array([r["target_vec"] for r in rows], dtype=np.float32)
    if arr.size == 0:
        return torch.ones(0, dtype=torch.float32)
    cls_counts = arr.sum(axis=0)
    inv = np.where(cls_counts > 0, 1.0 / cls_counts, 0.0)
    weights = []
    for i in range(arr.shape[0]):
        pos_idx = np.where(arr[i] > 0)[0]
        if pos_idx.size == 0:
            weights.append(0.2)
        else:
            weights.append(float(inv[pos_idx].sum() / max(pos_idx.size, 1)))
    w = np.array(weights, dtype=np.float32)
    w = np.clip(w, a_min=np.percentile(w, 5), a_max=np.percentile(w, 95))
    w = w / max(float(w.mean()), 1e-8)
    return torch.tensor(w, dtype=torch.float32)


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> List[Optional[float]]:
    thresholds: List[Optional[float]] = []
    grid = np.linspace(0.1, 0.9, 17)
    for c in range(y_true.shape[1]):
        best_t = 0.5
        best_f1 = -1.0
        yt = y_true[:, c]
        if yt.sum() == 0:
            thresholds.append(None)
            continue
        for t in grid:
            yp = (y_prob[:, c] >= t).astype(np.int32)
            f1 = f1_score(yt, yp, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds.append(best_t)
    return thresholds


def binarize_with_thresholds(y_prob: np.ndarray, thresholds: Sequence[Optional[float]], default_threshold: float = 0.5) -> np.ndarray:
    th_values = [float(t) if t is not None else float(default_threshold) for t in thresholds]
    th = np.array(th_values, dtype=np.float32).reshape(1, -1)
    return (y_prob >= th).astype(np.int32)


def compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = y_true.sum(axis=0)
    support_mask = support > 0

    macro_f1_all = float(np.mean(f1_per_class)) if f1_per_class.size else 0.0
    macro_f1_supported = float(np.mean(f1_per_class[support_mask])) if np.any(support_mask) else 0.0
    metrics["macro_f1"] = macro_f1_all
    metrics["macro_f1_all"] = macro_f1_all
    metrics["macro_f1_supported"] = macro_f1_supported

    try:
        metrics["mAP_micro"] = float(average_precision_score(y_true, y_prob, average="micro"))
    except ValueError:
        metrics["mAP_micro"] = 0.0

    ap_all: List[float] = []
    ap_supported: List[float] = []
    auroc_all: List[float] = []
    auroc_supported: List[float] = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if int(yt.sum()) > 0:
            try:
                ap_val = float(average_precision_score(yt, yp))
            except ValueError:
                ap_val = 0.0
            ap_supported.append(ap_val)
        else:
            ap_val = 0.0
        ap_all.append(ap_val)

        if np.unique(yt).size > 1:
            try:
                auroc_val = float(roc_auc_score(yt, yp))
            except ValueError:
                auroc_val = 0.0
        else:
            auroc_val = 0.0
        auroc_all.append(auroc_val)
        if int(yt.sum()) > 0:
            auroc_supported.append(auroc_val)

    metrics["mAP_macro"] = float(np.mean(ap_all)) if ap_all else 0.0
    metrics["mAP_macro_all"] = float(np.mean(ap_all)) if ap_all else 0.0
    metrics["mAP_macro_supported"] = float(np.mean(ap_supported)) if ap_supported else 0.0
    metrics["AUROC_macro_all"] = float(np.mean(auroc_all)) if auroc_all else 0.0
    metrics["AUROC_macro_supported"] = float(np.mean(auroc_supported)) if auroc_supported else 0.0
    return metrics


def compute_per_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    out: Dict[str, Dict[str, float]] = {}
    for i, lbl in enumerate(ERROR_TYPES):
        ap = 0.0
        auroc = 0.0
        try:
            ap = float(average_precision_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            ap = 0.0
        try:
            if np.unique(y_true[:, i]).size > 1:
                auroc = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
            else:
                auroc = 0.0
        except ValueError:
            auroc = 0.0

        out[lbl] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "ap": ap,
            "auroc": auroc,
            "support_positive": int(y_true[:, i].sum()),
        }
    return out


def metrics_by_language(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, langs: List[str]) -> Dict[str, Dict[str, float]]:
    if y_true.shape[0] != len(langs):
        raise ValueError(f"Language vector length mismatch: len(langs)={len(langs)} vs rows={y_true.shape[0]}")

    out: Dict[str, Dict[str, float]] = {}
    lang_arr = np.array([str(x).strip() for x in langs], dtype=object)
    for lang in LANGS:
        mask = lang_arr == lang
        if int(mask.sum()) == 0:
            out[lang] = {"micro_f1": 0.0, "macro_f1": 0.0, "mAP_macro": 0.0, "mAP_micro": 0.0}
            continue
        out[lang] = compute_multilabel_metrics(y_true[mask], y_prob[mask], y_pred[mask])
    out["overall"] = compute_multilabel_metrics(y_true, y_prob, y_pred)
    return out


def language_debug_summary(y_true: np.ndarray, langs: List[str]) -> Dict[str, Dict[str, Any]]:
    if y_true.shape[0] != len(langs):
        raise ValueError(f"Language vector length mismatch: len(langs)={len(langs)} vs rows={y_true.shape[0]}")

    lang_arr = np.array([str(x).strip() for x in langs], dtype=object)
    out: Dict[str, Dict[str, Any]] = {}
    for lang in LANGS:
        mask = lang_arr == lang
        rows = int(mask.sum())
        pos = int(y_true[mask].sum()) if rows > 0 else 0
        per_label = {ERROR_TYPES[i]: int(y_true[mask, i].sum()) if rows > 0 else 0 for i in range(len(ERROR_TYPES))}
        out[lang] = {
            "eval_rows": rows,
            "positive_label_count": pos,
            "positive_label_counts_by_class": per_label,
        }
    out["overall"] = {
        "eval_rows": int(y_true.shape[0]),
        "positive_label_count": int(y_true.sum()),
        "positive_label_counts_by_class": {ERROR_TYPES[i]: int(y_true[:, i].sum()) for i in range(len(ERROR_TYPES))},
    }
    return out


def prediction_diagnostics(y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if y_prob.size == 0:
        zero_map = {lbl: 0 for lbl in ERROR_TYPES}
        zero_float_map = {lbl: 0.0 for lbl in ERROR_TYPES}
        return {
            "avg_predicted_positives_per_sample": 0.0,
            "predicted_positive_count_per_class": zero_map,
            "mean_probability_per_class": zero_float_map,
            "max_probability_per_class": zero_float_map,
        }

    avg_pos = float(y_pred.sum(axis=1).mean())
    pred_pos_per_class = {ERROR_TYPES[i]: int(y_pred[:, i].sum()) for i in range(y_pred.shape[1])}
    mean_prob_per_class = {ERROR_TYPES[i]: float(y_prob[:, i].mean()) for i in range(y_prob.shape[1])}
    max_prob_per_class = {ERROR_TYPES[i]: float(y_prob[:, i].max()) for i in range(y_prob.shape[1])}
    return {
        "avg_predicted_positives_per_sample": avg_pos,
        "predicted_positive_count_per_class": pred_pos_per_class,
        "mean_probability_per_class": mean_prob_per_class,
        "max_probability_per_class": max_prob_per_class,
    }


def load_stage1_compatible_weights(model: nn.Module, checkpoint_path: Optional[str]) -> Dict[str, Any]:
    return load_checkpoint_partial(model, checkpoint_path)


def _extract_state_dict(state: Any) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    return None


def load_checkpoint_partial(model: nn.Module, checkpoint_path: Optional[str]) -> Dict[str, Any]:
    if not checkpoint_path:
        return {"loaded": False, "reason": "no_checkpoint"}

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        return {"loaded": False, "reason": "not_found", "path": str(ckpt)}

    state = torch.load(str(ckpt), map_location="cpu")
    state_dict = _extract_state_dict(state)
    if state_dict is None:
        return {"loaded": False, "reason": "invalid_checkpoint_format", "path": str(ckpt)}

    model_state = model.state_dict()
    loadable: Dict[str, torch.Tensor] = {}
    shape_mismatch: List[str] = []
    unexpected_keys: List[str] = []

    for k, v in state_dict.items():
        if k not in model_state:
            unexpected_keys.append(k)
            continue
        if tuple(model_state[k].shape) != tuple(v.shape):
            shape_mismatch.append(k)
            continue
        loadable[k] = v

    missing, _ = model.load_state_dict(loadable, strict=False)
    return {
        "loaded": True,
        "path": str(ckpt),
        "missing_keys": missing,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch_keys": shape_mismatch,
        "checkpoint_key_count": int(len(state_dict.keys())),
        "loaded_key_count": int(len(loadable.keys())),
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
