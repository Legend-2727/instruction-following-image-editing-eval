"""Recovered evidence-grounded taxonomy utilities.

This module is grounded in
``notebooks/evidence_grounded_taxonomy_eval_lineage_v1_to_v3.ipynb`` and keeps
the original EN/HI/BN single-prompt protocol while adapting paths to the
cleaned repo layout.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from scripts.utils.io import ensure_dirs, load_jsonl, normalize_relpath, save_jsonl
from scripts.utils.schema import ERROR_TO_IDX, ERROR_TYPES


NOTEBOOK_SOURCE = "notebooks/evidence_grounded_taxonomy_eval_lineage_v1_to_v3.ipynb"

LANGS: Tuple[str, ...] = ("en", "hi", "bn")
OP_TYPES: List[str] = [
    "remove",
    "add",
    "replace",
    "recolor",
    "style",
    "move",
    "background",
    "other",
]
OP_TO_IDX: Dict[str, int] = {op: i for i, op in enumerate(OP_TYPES)}

DEFAULT_VISION_MODEL_ID = "google/vit-base-patch16-224-in21k"
DEFAULT_TEXT_MODEL_ID = "xlm-roberta-base"
DEFAULT_IMG_SIZE = 224
DEFAULT_MAX_TEXT_LEN = 128
DEFAULT_HIDDEN_DIM = 256
DEFAULT_DROPOUT = 0.15

DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 0

DEFAULT_V1_REPORT_DIR = Path("reports/evidence_grounded/v1")
DEFAULT_V2_REPORT_DIR = Path("reports/evidence_grounded/v2")
DEFAULT_V3_REPORT_DIR = Path("reports/evidence_grounded/v3")

DEFAULT_V1_MODEL_OUT = Path("experiments/history/evidence_grounded_taxonomy_eval_v1/best_model.pt")
DEFAULT_V2_MODEL_OUT = Path("experiments/history/evidence_grounded_taxonomy_eval_v2/best_model.pt")
DEFAULT_V3_MODEL_OUT = Path("models/released/evidence_grounded_taxonomy_eval_v3/best_model.pt")

LOCAL_IMPORT_V1_MODEL = Path("evidence_grounded_taxonomy_eval/best_model.pt")
LOCAL_IMPORT_V2_MODEL = Path("evidence_grounded_taxonomy_eval_v2/best_model.pt")
LOCAL_IMPORT_V3_MODEL = Path("evidence_grounded_taxonomy_eval_v3/best_model.pt")

LOCAL_IMPORT_V1_THRESHOLDS = Path("evidence_grounded_taxonomy_eval/best_thresholds.json")
LOCAL_IMPORT_V2_THRESHOLDS = Path("evidence_grounded_taxonomy_eval_v2/best_thresholds.json")
LOCAL_IMPORT_V3_THRESHOLDS = Path("evidence_grounded_taxonomy_eval_v3/best_thresholds.json")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DatasetPaths:
    image_root: Path
    train_jsonl: Path
    val_jsonl: Path
    test_jsonl: Path


@dataclass
class ModelConfig:
    vision_model_id: str = DEFAULT_VISION_MODEL_ID
    text_model_id: str = DEFAULT_TEXT_MODEL_ID
    img_size: int = DEFAULT_IMG_SIZE
    max_text_len: int = DEFAULT_MAX_TEXT_LEN
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    dropout: float = DEFAULT_DROPOUT
    unfreeze_text_layers: int = 2
    unfreeze_vision_layers: int = 2
    local_files_only: bool = False


@dataclass
class V1TrainConfig:
    seed: int = 42
    train_neg_ratio: float = 0.35
    primary_eval_mode: str = "taxonomy_positive"
    smoke_test: bool = False
    smoke_limit_originals: int = 120
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    epochs: int = 8
    patience: int = 3
    head_lr: float = 3e-4
    encoder_lr: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_amp: bool = True
    focal_gamma: float = 1.5
    aux_op_loss_weight: float = 0.20


@dataclass
class V2FinetuneConfig:
    seed: int = 42
    train_neg_ratio: float = 0.35
    smoke_test: bool = False
    smoke_limit_originals: int = 120
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    epochs: int = 5
    patience: int = 3
    lr: float = 1e-5
    weight_decay: float = 5e-4
    grad_clip: float = 1.0
    use_amp: bool = True


@dataclass
class V3FinetuneConfig:
    seed: int = 42
    train_neg_ratio: float = 0.35
    smoke_test: bool = False
    smoke_limit_originals: int = 120
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    epochs: int = 6
    patience: int = 3
    lr: float = 5e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True
    alpha_plain: float = 0.70
    alpha_weighted: float = 0.30
    op_aux_weight: float = 0.10


@dataclass
class RowBundle:
    train_orig: List[Dict[str, Any]]
    val_orig: List[Dict[str, Any]]
    test_orig: List[Dict[str, Any]]
    train_rows: List[Dict[str, Any]]
    val_rows_full: List[Dict[str, Any]]
    test_rows_full: List[Dict[str, Any]]
    val_rows_benchmark: List[Dict[str, Any]]
    test_rows_benchmark: List[Dict[str, Any]]


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader_full: DataLoader
    test_loader_full: DataLoader
    val_loader_benchmark: DataLoader
    test_loader_benchmark: DataLoader


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: str | Path, obj: Dict[str, Any] | List[Any]) -> None:
    path = Path(path)
    ensure_dirs(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_lang(lang: str) -> str:
    if lang not in LANGS:
        raise ValueError(f"Unsupported language {lang!r}. Expected one of {LANGS}.")
    return lang


def labels_to_multihot(labels: Sequence[str]) -> List[float]:
    vec = [0.0] * len(ERROR_TYPES)
    for label in labels:
        if label in ERROR_TO_IDX:
            vec[ERROR_TO_IDX[label]] = 1.0
    return vec


def infer_prompt_operation(prompt: str) -> str:
    prompt_lower = (prompt or "").strip().lower()
    patterns = {
        "remove": [
            r"\bremove\b",
            r"\bdelete\b",
            r"\berase\b",
            r"\bwithout\b",
            r"हटा",
            r"निकाल",
            r"मिटा",
            r"মুছে",
            r"সরিয়ে",
            r"সরিয়ে",
            r"বাদ দাও",
        ],
        "add": [
            r"\badd\b",
            r"\binsert\b",
            r"\binclude\b",
            r"\bput\b",
            r"जोड़",
            r"डाल",
            r"যোগ",
            r"দাও",
        ],
        "replace": [
            r"\breplace\b",
            r"\bswap\b",
            r"\bchange the .* to\b",
            r"बदल",
            r"পরিবর্তন",
            r"বদল",
        ],
        "recolor": [
            r"\bcolor\b",
            r"\brecolor\b",
            r"\bturn .* (red|blue|green|black|white|yellow|pink|orange|purple|brown|gold|silver)\b",
            r"रंग",
            r"কালো",
            r"লাল",
            r"নীল",
            r"সাদা",
        ],
        "style": [
            r"\bstyle\b",
            r"\bmake it look like\b",
            r"\bcartoon\b",
            r"\boil painting\b",
            r"शैली",
            r"স্টাইল",
            r"ধাঁচ",
        ],
        "move": [
            r"\bmove\b",
            r"\bshift\b",
            r"\bplace\b",
            r"\bput .* on the left\b",
            r"\bput .* on the right\b",
            r"स्थान",
            r"बाएँ",
            r"दाएँ",
            r"সরে",
            r"বামে",
            r"ডানে",
        ],
        "background": [
            r"\bbackground\b",
            r"\bsky\b",
            r"\bwall\b",
            r"\bscene\b",
            r"पृष्ठभूमि",
            r"आसमान",
            r"ব্যাকগ্রাউন্ড",
            r"আকাশ",
        ],
    }
    for op_type, op_patterns in patterns.items():
        for pattern in op_patterns:
            if re.search(pattern, prompt_lower):
                return op_type
    return "other"


def expand_rows_by_language(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for row in rows:
        labels = [str(label) for label in row.get("taxonomy_labels", [])]
        for lang in LANGS:
            text = str(row.get(f"instruction_{lang}") or "").strip()
            if not text:
                continue
            op_type = infer_prompt_operation(text)
            expanded.append(
                {
                    "id": str(row["id"]),
                    "lang": lang,
                    "instruction": text,
                    "source_path": normalize_relpath(str(row.get("source_path") or "")),
                    "target_path": normalize_relpath(str(row.get("target_path") or "")),
                    "taxonomy_labels": labels,
                    "target_vec": labels_to_multihot(labels),
                    "has_error": 1 if labels else 0,
                    "op_type": op_type,
                    "op_type_id": OP_TO_IDX[op_type],
                }
            )
    return expanded


def filter_taxonomy_positive(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("taxonomy_labels")]


def build_train_rows(train_orig: Sequence[Dict[str, Any]], neg_ratio: float, seed: int) -> List[Dict[str, Any]]:
    positives = filter_taxonomy_positive(train_orig)
    negatives = [row for row in train_orig if not row.get("taxonomy_labels")]
    kept_negatives = negatives
    if neg_ratio > 0 and positives and negatives:
        keep_count = min(len(negatives), int(round(len(positives) * neg_ratio)))
        kept_negatives = random.Random(seed).sample(negatives, keep_count)
    return expand_rows_by_language(list(positives) + list(kept_negatives))


def load_row_bundle(paths: DatasetPaths, train_neg_ratio: float, seed: int, smoke_test: bool, smoke_limit_originals: int) -> RowBundle:
    train_orig = load_jsonl(paths.train_jsonl)
    val_orig = load_jsonl(paths.val_jsonl)
    test_orig = load_jsonl(paths.test_jsonl)

    if smoke_test:
        train_orig = train_orig[:smoke_limit_originals]
        val_orig = val_orig[: max(30, smoke_limit_originals // 8)]
        test_orig = test_orig[: max(30, smoke_limit_originals // 8)]

    return RowBundle(
        train_orig=train_orig,
        val_orig=val_orig,
        test_orig=test_orig,
        train_rows=build_train_rows(train_orig, train_neg_ratio, seed),
        val_rows_full=expand_rows_by_language(val_orig),
        test_rows_full=expand_rows_by_language(test_orig),
        val_rows_benchmark=expand_rows_by_language(filter_taxonomy_positive(val_orig)),
        test_rows_benchmark=expand_rows_by_language(filter_taxonomy_positive(test_orig)),
    )


def dataset_summary(row_bundle: RowBundle, train_neg_ratio: float) -> Dict[str, Any]:
    return {
        "train_originals": len(row_bundle.train_orig),
        "val_originals": len(row_bundle.val_orig),
        "test_originals": len(row_bundle.test_orig),
        "train_rows": len(row_bundle.train_rows),
        "val_rows_full": len(row_bundle.val_rows_full),
        "test_rows_full": len(row_bundle.test_rows_full),
        "val_rows_benchmark": len(row_bundle.val_rows_benchmark),
        "test_rows_benchmark": len(row_bundle.test_rows_benchmark),
        "languages": list(LANGS),
        "train_neg_ratio": train_neg_ratio,
    }


def build_eval_transform(img_size: int = DEFAULT_IMG_SIZE):
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def transform(image: Image.Image) -> torch.Tensor:
        resized = image.resize((img_size, img_size))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - mean) / std

    return transform


class EvidenceTaxonomyDataset(Dataset):
    """Multilingual single-prompt dataset grounded in the notebook implementation."""

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        image_root: str | Path,
        tokenizer,
        max_text_len: int = DEFAULT_MAX_TEXT_LEN,
        image_transform=None,
    ):
        self.rows = list(rows)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        image_path = self.image_root / Path(normalize_relpath(rel_path))
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            if self.image_transform is not None:
                return self.image_transform(rgb)
            array = np.asarray(rgb, dtype=np.float32) / 255.0
            return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        tokenized = self.tokenizer(
            row["instruction"],
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return {
            "id": row["id"],
            "lang": row["lang"],
            "instruction": row["instruction"],
            "source_path": row["source_path"],
            "target_path": row["target_path"],
            "src_img": self._load_image(row["source_path"]),
            "tgt_img": self._load_image(row["target_path"]),
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "target": torch.tensor(row["target_vec"], dtype=torch.float32),
            "has_error": torch.tensor(float(row["has_error"]), dtype=torch.float32),
            "op_type_id": torch.tensor(int(row["op_type_id"]), dtype=torch.long),
            "taxonomy_labels": list(row["taxonomy_labels"]),
        }


def build_tokenizer(model_config: ModelConfig):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_config.text_model_id,
        local_files_only=model_config.local_files_only,
    )


def build_dataloaders(
    row_bundle: RowBundle,
    dataset_paths: DatasetPaths,
    model_config: ModelConfig,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> LoaderBundle:
    tokenizer = build_tokenizer(model_config)
    transform = build_eval_transform(model_config.img_size)

    train_ds = EvidenceTaxonomyDataset(row_bundle.train_rows, dataset_paths.image_root, tokenizer, model_config.max_text_len, transform)
    val_ds_full = EvidenceTaxonomyDataset(row_bundle.val_rows_full, dataset_paths.image_root, tokenizer, model_config.max_text_len, transform)
    test_ds_full = EvidenceTaxonomyDataset(row_bundle.test_rows_full, dataset_paths.image_root, tokenizer, model_config.max_text_len, transform)
    val_ds_benchmark = EvidenceTaxonomyDataset(row_bundle.val_rows_benchmark, dataset_paths.image_root, tokenizer, model_config.max_text_len, transform)
    test_ds_benchmark = EvidenceTaxonomyDataset(row_bundle.test_rows_benchmark, dataset_paths.image_root, tokenizer, model_config.max_text_len, transform)

    pin_memory = device.type == "cuda"
    return LoaderBundle(
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        val_loader_full=DataLoader(val_ds_full, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        test_loader_full=DataLoader(test_ds_full, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        val_loader_benchmark=DataLoader(val_ds_benchmark, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        test_loader_benchmark=DataLoader(test_ds_benchmark, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    )


def freeze_all(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_last_n_layers(module: nn.Module, layer_count: int) -> None:
    if layer_count <= 0:
        return
    layers = None
    if hasattr(module, "encoder") and hasattr(module.encoder, "layer"):
        layers = module.encoder.layer
    if layers is None:
        return
    for layer in layers[-layer_count:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


class EvidenceGroundedTaxonomyModel(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        from transformers import ViTModel, XLMRobertaModel

        self.vision = ViTModel.from_pretrained(
            model_config.vision_model_id,
            local_files_only=model_config.local_files_only,
        )
        self.text = XLMRobertaModel.from_pretrained(
            model_config.text_model_id,
            local_files_only=model_config.local_files_only,
        )

        freeze_all(self.vision)
        freeze_all(self.text)
        unfreeze_last_n_layers(self.vision, model_config.unfreeze_vision_layers)
        unfreeze_last_n_layers(self.text, model_config.unfreeze_text_layers)

        vision_dim = self.vision.config.hidden_size
        text_dim = self.text.config.hidden_size
        hidden_dim = model_config.hidden_dim
        dropout = model_config.dropout

        self.patch_proj = nn.Linear(vision_dim, hidden_dim)
        self.vision_cls_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_cls_proj = nn.Linear(text_dim, hidden_dim)
        self.text_query_proj = nn.Linear(hidden_dim, hidden_dim)

        self.src_presence_head = MLP(hidden_dim * 3, hidden_dim, 1, dropout)
        self.tgt_presence_head = MLP(hidden_dim * 3, hidden_dim, 1, dropout)
        self.local_change_head = MLP(hidden_dim, hidden_dim, 1, dropout)
        self.global_change_head = MLP(hidden_dim, hidden_dim, 1, dropout)
        self.outside_change_head = MLP(hidden_dim, hidden_dim, 1, dropout)
        self.op_head = MLP(hidden_dim, hidden_dim, len(OP_TYPES), dropout)

        evidence_dim = hidden_dim * 8 + 7 + len(OP_TYPES)
        self.taxonomy_head = nn.Sequential(
            nn.Linear(evidence_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ERROR_TYPES)),
        )

    def forward(
        self,
        src_img: torch.Tensor,
        tgt_img: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        src_out = self.vision(pixel_values=src_img)
        tgt_out = self.vision(pixel_values=tgt_img)
        txt_out = self.text(input_ids=input_ids, attention_mask=attention_mask)

        src_seq = src_out.last_hidden_state
        tgt_seq = tgt_out.last_hidden_state
        txt_seq = txt_out.last_hidden_state

        src_cls = self.vision_cls_proj(src_seq[:, 0])
        tgt_cls = self.vision_cls_proj(tgt_seq[:, 0])
        src_patches = self.patch_proj(src_seq[:, 1:])
        tgt_patches = self.patch_proj(tgt_seq[:, 1:])
        txt_cls = self.text_cls_proj(txt_seq[:, 0])

        query = F.normalize(self.text_query_proj(txt_cls), dim=-1)
        src_norm = F.normalize(src_patches, dim=-1)
        tgt_norm = F.normalize(tgt_patches, dim=-1)

        src_scores = torch.einsum("bnd,bd->bn", src_norm, query) / math.sqrt(src_norm.shape[-1])
        tgt_scores = torch.einsum("bnd,bd->bn", tgt_norm, query) / math.sqrt(tgt_norm.shape[-1])

        src_attn = torch.softmax(src_scores, dim=-1)
        tgt_attn = torch.softmax(tgt_scores, dim=-1)

        src_local = torch.sum(src_patches * src_attn.unsqueeze(-1), dim=1)
        tgt_local = torch.sum(tgt_patches * tgt_attn.unsqueeze(-1), dim=1)

        patch_diff = torch.abs(src_patches - tgt_patches)
        local_diff = torch.sum(patch_diff * src_attn.unsqueeze(-1), dim=1)

        outside_weights = (1.0 - src_attn).clamp(min=1e-6)
        outside_weights = outside_weights / outside_weights.sum(dim=-1, keepdim=True)
        outside_diff = torch.sum(patch_diff * outside_weights.unsqueeze(-1), dim=1)

        corr_scores = torch.einsum(
            "bnd,bd->bn",
            F.normalize(tgt_patches, dim=-1),
            F.normalize(src_local, dim=-1),
        )
        corr_max = corr_scores.max(dim=-1).values.unsqueeze(-1)
        corr_mean = corr_scores.mean(dim=-1, keepdim=True)

        src_presence = torch.sigmoid(self.src_presence_head(torch.cat([txt_cls, src_local, src_cls], dim=-1)))
        tgt_presence = torch.sigmoid(self.tgt_presence_head(torch.cat([txt_cls, tgt_local, tgt_cls], dim=-1)))
        local_change_score = torch.sigmoid(self.local_change_head(local_diff))
        global_change_feat = torch.abs(src_cls - tgt_cls)
        global_change_score = torch.sigmoid(self.global_change_head(global_change_feat))
        outside_change_score = torch.sigmoid(self.outside_change_head(outside_diff))
        op_logits = self.op_head(txt_cls)
        op_probs = torch.softmax(op_logits, dim=-1)

        evidence_vec = torch.cat(
            [
                txt_cls,
                src_cls,
                tgt_cls,
                src_local,
                tgt_local,
                global_change_feat,
                local_diff,
                outside_diff,
                src_presence,
                tgt_presence,
                corr_max,
                corr_mean,
                local_change_score,
                global_change_score,
                outside_change_score,
                op_probs,
            ],
            dim=-1,
        )
        taxonomy_logits = self.taxonomy_head(evidence_vec)
        evidence = {
            "src_attn": src_attn,
            "tgt_attn": tgt_attn,
            "src_presence": src_presence.squeeze(-1),
            "tgt_presence": tgt_presence.squeeze(-1),
            "correspondence_max": corr_max.squeeze(-1),
            "correspondence_mean": corr_mean.squeeze(-1),
            "local_change_score": local_change_score.squeeze(-1),
            "global_change_score": global_change_score.squeeze(-1),
            "outside_change_score": outside_change_score.squeeze(-1),
            "op_logits": op_logits,
            "op_probs": op_probs,
        }
        return taxonomy_logits, evidence


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 1.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal = (1 - pt).pow(self.gamma)
        return (focal * bce).mean()


def compute_pos_weight(rows: Sequence[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    labels = np.array([row["target_vec"] for row in rows], dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    pos_weight = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 25.0)
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    support_mask = y_true.sum(axis=0) > 0
    metrics["macro_f1_supported"] = float(per_class_f1[support_mask].mean()) if support_mask.any() else 0.0
    try:
        metrics["mAP_micro"] = float(average_precision_score(y_true, y_prob, average="micro"))
    except Exception:
        metrics["mAP_micro"] = 0.0
    try:
        ap_per_class = []
        for index in range(y_true.shape[1]):
            if y_true[:, index].sum() > 0:
                ap_per_class.append(average_precision_score(y_true[:, index], y_prob[:, index]))
        metrics["mAP_macro_supported"] = float(np.mean(ap_per_class)) if ap_per_class else 0.0
    except Exception:
        metrics["mAP_macro_supported"] = 0.0
    return metrics


def compute_per_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    del y_prob
    rows: List[Dict[str, Any]] = []
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    support = y_true.sum(axis=0)
    for index, label in enumerate(ERROR_TYPES):
        rows.append(
            {
                "class": label,
                "f1": float(per_f1[index]),
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "support": int(support[index]),
            }
        )
    return pd.DataFrame(rows)


def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: Optional[Sequence[float]] = None,
) -> List[float]:
    thresholds: List[float] = []
    search_grid = list(grid) if grid is not None else list(np.round(np.arange(0.15, 0.86, 0.05), 2))
    for class_index in range(y_true.shape[1]):
        if y_true[:, class_index].sum() == 0:
            thresholds.append(0.50)
            continue
        best_threshold = 0.50
        best_f1 = -1.0
        for threshold in search_grid:
            predicted = (y_prob[:, class_index] >= threshold).astype(int)
            metric = f1_score(y_true[:, class_index], predicted, zero_division=0)
            if metric > best_f1:
                best_f1 = metric
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return thresholds


def thresholds_to_list(thresholds: Any) -> List[float]:
    if isinstance(thresholds, dict):
        return [float(thresholds[label]) for label in ERROR_TYPES]
    if isinstance(thresholds, list):
        if len(thresholds) != len(ERROR_TYPES):
            raise ValueError("Threshold list length does not match canonical taxonomy length.")
        return [float(value) for value in thresholds]
    raise TypeError(f"Unsupported thresholds payload: {type(thresholds)!r}")


def thresholds_to_label_map(thresholds: Sequence[float]) -> Dict[str, float]:
    values = [float(value) for value in thresholds]
    if len(values) != len(ERROR_TYPES):
        raise ValueError("Threshold count does not match taxonomy length.")
    return {label: value for label, value in zip(ERROR_TYPES, values)}


def load_thresholds(path: str | Path) -> List[float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return thresholds_to_list(payload)


def select_labels_from_probabilities(probabilities: Sequence[float], thresholds: Sequence[float]) -> List[str]:
    return [
        label
        for label, probability, threshold in zip(ERROR_TYPES, probabilities, thresholds)
        if float(probability) >= float(threshold)
    ]


def evidence_to_reason(evidence_row: Dict[str, Any], predicted_labels: Sequence[str]) -> str:
    parts = [
        f"pred_op={evidence_row['pred_op_type']}",
        f"src_presence={evidence_row['src_presence']:.2f}",
        f"tgt_presence={evidence_row['tgt_presence']:.2f}",
        f"corr_max={evidence_row['correspondence_max']:.2f}",
        f"local_change={evidence_row['local_change_score']:.2f}",
        f"outside_change={evidence_row['outside_change_score']:.2f}",
    ]
    if predicted_labels:
        parts.append("labels=" + ", ".join(predicted_labels))
    else:
        parts.append("labels=[]")
    return " | ".join(parts)


def collect_trainable_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    head_params: List[nn.Parameter] = []
    encoder_params: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("vision.") or name.startswith("text."):
            encoder_params.append(parameter)
        else:
            head_params.append(parameter)
    return head_params, encoder_params


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src = batch["src_img"].to(device, non_blocking=True)
    tgt = batch["tgt_img"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True).float()
    return src, tgt, input_ids, attention_mask, targets


def evaluate_with_evidence(
    model: nn.Module,
    loader: DataLoader,
    criterion_tax: nn.Module,
    criterion_op: nn.Module,
    device: torch.device,
    aux_op_loss_weight: float,
    use_amp: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], pd.DataFrame]:
    model.eval()
    probabilities, golds, logits_out = [], [], []
    langs: List[str] = []
    ids: List[str] = []
    evidence_rows: List[Dict[str, Any]] = []
    total_loss = 0.0
    example_count = 0

    with torch.no_grad():
        for batch in loader:
            src, tgt, input_ids, attention_mask, targets = _batch_to_device(batch, device)
            op_ids = batch["op_type_id"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                logits, evidence = model(src, tgt, input_ids, attention_mask)
                loss_tax = criterion_tax(logits, targets)
                loss_op = criterion_op(evidence["op_logits"], op_ids)
                loss = loss_tax + aux_op_loss_weight * loss_op

            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
            probabilities.append(batch_probs)
            golds.append(targets.detach().cpu().numpy())
            logits_out.append(logits.detach().cpu().numpy())

            total_loss += float(loss.item()) * targets.shape[0]
            example_count += int(targets.shape[0])
            langs.extend(batch["lang"])
            ids.extend(batch["id"])

            for index in range(len(batch["id"])):
                evidence_rows.append(
                    {
                        "id": batch["id"][index],
                        "lang": batch["lang"][index],
                        "src_presence": float(evidence["src_presence"][index].detach().cpu()),
                        "tgt_presence": float(evidence["tgt_presence"][index].detach().cpu()),
                        "correspondence_max": float(evidence["correspondence_max"][index].detach().cpu()),
                        "correspondence_mean": float(evidence["correspondence_mean"][index].detach().cpu()),
                        "local_change_score": float(evidence["local_change_score"][index].detach().cpu()),
                        "global_change_score": float(evidence["global_change_score"][index].detach().cpu()),
                        "outside_change_score": float(evidence["outside_change_score"][index].detach().cpu()),
                        "pred_op_type": OP_TYPES[int(torch.argmax(evidence["op_probs"][index]).detach().cpu())],
                    }
                )

    y_prob = np.concatenate(probabilities, axis=0)
    y_true = np.concatenate(golds, axis=0)
    y_logits = np.concatenate(logits_out, axis=0)
    return total_loss / max(example_count, 1), y_true, y_prob, y_logits, langs, ids, pd.DataFrame(evidence_rows)


def evaluate_taxonomy_only(
    model: nn.Module,
    loader: DataLoader,
    criterion_tax: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    alpha_plain: Optional[float] = None,
    alpha_weighted: Optional[float] = None,
    criterion_tax_weighted: Optional[nn.Module] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_prob = [], []
    total_loss = 0.0
    example_count = 0

    with torch.no_grad():
        for batch in loader:
            src, tgt, input_ids, attention_mask, targets = _batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                logits, _ = model(src, tgt, input_ids, attention_mask)
                loss = criterion_tax(logits, targets)
                if alpha_plain is not None and alpha_weighted is not None and criterion_tax_weighted is not None:
                    weighted = criterion_tax_weighted(logits, targets)
                    loss = alpha_plain * loss + alpha_weighted * weighted

            y_prob.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_true.append(targets.detach().cpu().numpy())
            total_loss += float(loss.item()) * targets.shape[0]
            example_count += int(targets.shape[0])

    return (
        total_loss / max(example_count, 1),
        np.concatenate(y_true, axis=0),
        np.concatenate(y_prob, axis=0),
    )


def _collect_train_targets(loader: DataLoader) -> np.ndarray:
    targets = []
    for batch in loader:
        targets.append(batch["target"].cpu().numpy())
    return np.concatenate(targets, axis=0)


def _get_optional_op_ids(batch: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    for key in ["op_id", "op_ids", "operation_id", "operation_ids", "op_type_id"]:
        if key in batch:
            return batch[key].to(device, non_blocking=True).long()
    return None


def build_prediction_rows(
    ids: Sequence[str],
    langs: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index in range(len(ids)):
        rows.append(
            {
                "id": ids[index],
                "lang": langs[index],
                "gold_labels": [ERROR_TYPES[j] for j, value in enumerate(y_true[index]) if int(value) == 1],
                "pred_labels": [ERROR_TYPES[j] for j, value in enumerate(y_pred[index]) if int(value) == 1],
                "probs": {ERROR_TYPES[j]: float(y_prob[index, j]) for j in range(len(ERROR_TYPES))},
            }
        )
    return rows


def build_vector_prediction_rows(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index in range(y_true.shape[0]):
        rows.append(
            {
                "gold_vec": y_true[index].astype(int).tolist(),
                "pred_vec": y_pred[index].astype(int).tolist(),
                "prob": [float(value) for value in y_prob[index].tolist()],
            }
        )
    return rows


def resolve_best_existing_path(primary: str | Path, fallbacks: Sequence[str | Path]) -> Path:
    candidates = [Path(primary)] + [Path(path) for path in fallbacks]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(primary)


def build_model(model_config: ModelConfig, device: torch.device) -> EvidenceGroundedTaxonomyModel:
    return EvidenceGroundedTaxonomyModel(model_config).to(device)


def save_threshold_artifacts(report_dir: str | Path, thresholds: Sequence[float] | Dict[str, float], prefer_array: bool) -> List[float]:
    report_dir = Path(report_dir)
    ensure_dirs(report_dir)
    threshold_list = thresholds_to_list(thresholds)
    best_thresholds_payload: Any = threshold_list if prefer_array else thresholds_to_label_map(threshold_list)
    save_json(report_dir / "best_thresholds.json", best_thresholds_payload)
    save_json(report_dir / "best_thresholds_by_label.json", thresholds_to_label_map(threshold_list))
    save_json(report_dir / "label_map.json", {str(index): label for index, label in enumerate(ERROR_TYPES)})
    return threshold_list


def train_v1(
    dataset_paths: DatasetPaths,
    report_dir: str | Path,
    model_out: str | Path,
    model_config: ModelConfig,
    train_config: V1TrainConfig,
) -> Dict[str, Any]:
    device = resolve_device()
    seed_everything(train_config.seed)
    report_dir = Path(report_dir)
    model_out = Path(model_out)
    ensure_dirs(report_dir, model_out.parent)

    row_bundle = load_row_bundle(
        dataset_paths,
        train_neg_ratio=train_config.train_neg_ratio,
        seed=train_config.seed,
        smoke_test=train_config.smoke_test,
        smoke_limit_originals=train_config.smoke_limit_originals,
    )
    save_json(report_dir / "dataset_summary.json", dataset_summary(row_bundle, train_config.train_neg_ratio))

    loaders = build_dataloaders(
        row_bundle,
        dataset_paths,
        model_config,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        device=device,
    )
    model = build_model(model_config, device)

    pos_weight = compute_pos_weight(row_bundle.train_rows, device)
    criterion_tax = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=train_config.focal_gamma)
    criterion_op = nn.CrossEntropyLoss()
    head_params, encoder_params = collect_trainable_params(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": train_config.head_lr},
            {"params": encoder_params, "lr": train_config.encoder_lr},
        ],
        weight_decay=train_config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(train_config.use_amp and device.type == "cuda"))

    best_score = -1.0
    best_state = None
    best_thresholds = [0.5] * len(ERROR_TYPES)
    history: List[Dict[str, Any]] = []
    no_improve = 0

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        running_loss = 0.0
        example_count = 0
        for batch in loaders.train_loader:
            src, tgt, input_ids, attention_mask, targets = _batch_to_device(batch, device)
            op_ids = batch["op_type_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(train_config.use_amp and device.type == "cuda")):
                logits, evidence = model(src, tgt, input_ids, attention_mask)
                loss_tax = criterion_tax(logits, targets)
                loss_op = criterion_op(evidence["op_logits"], op_ids)
                loss = loss_tax + train_config.aux_op_loss_weight * loss_op

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * targets.shape[0]
            example_count += int(targets.shape[0])

        val_loss, y_true_v, y_prob_v, _, _, _, _ = evaluate_with_evidence(
            model,
            loaders.val_loader_benchmark,
            criterion_tax,
            criterion_op,
            device,
            train_config.aux_op_loss_weight,
            use_amp=train_config.use_amp,
        )
        thresholds_v = tune_thresholds(y_true_v, y_prob_v)
        y_pred_v = (y_prob_v >= np.array(thresholds_v).reshape(1, -1)).astype(int)
        metrics_v = compute_multilabel_metrics(y_true_v, y_prob_v, y_pred_v)
        score = metrics_v["macro_f1_supported"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(example_count, 1),
                "val_loss_benchmark": val_loss,
                **{f"val_benchmark_{key}": value for key, value in metrics_v.items()},
            }
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_thresholds = thresholds_v
            no_improve = 0
            torch.save(best_state, model_out)
            save_json(report_dir / "best_thresholds.json", thresholds_to_label_map(best_thresholds))
        else:
            no_improve += 1
            if no_improve >= train_config.patience and not train_config.smoke_test:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(torch.load(model_out, map_location=device))

    _, y_true_tb, y_prob_tb, _, langs_tb, ids_tb, evidence_tb = evaluate_with_evidence(
        model,
        loaders.test_loader_benchmark,
        criterion_tax,
        criterion_op,
        device,
        train_config.aux_op_loss_weight,
        use_amp=train_config.use_amp,
    )
    y_pred_tb = (y_prob_tb >= np.array(best_thresholds).reshape(1, -1)).astype(int)
    metrics_tb = compute_multilabel_metrics(y_true_tb, y_prob_tb, y_pred_tb)
    per_class_tb = compute_per_class_metrics(y_true_tb, y_prob_tb, y_pred_tb)

    _, y_true_tf, y_prob_tf, _, _, _, _ = evaluate_with_evidence(
        model,
        loaders.test_loader_full,
        criterion_tax,
        criterion_op,
        device,
        train_config.aux_op_loss_weight,
        use_amp=train_config.use_amp,
    )
    y_pred_tf = (y_prob_tf >= np.array(best_thresholds).reshape(1, -1)).astype(int)
    metrics_tf = compute_multilabel_metrics(y_true_tf, y_prob_tf, y_pred_tf)

    prediction_rows = build_prediction_rows(ids_tb, langs_tb, y_true_tb, y_pred_tb, y_prob_tb)
    evidence_preview = evidence_tb.copy()
    pred_lookup = {(row["id"], row["lang"]): row["pred_labels"] for row in prediction_rows}
    evidence_preview["pred_labels"] = evidence_preview.apply(
        lambda row: pred_lookup.get((row["id"], row["lang"]), []),
        axis=1,
    )
    evidence_preview["reason"] = evidence_preview.apply(
        lambda row: evidence_to_reason(row, row["pred_labels"]),
        axis=1,
    )

    save_json(report_dir / "train_history.json", {"history": history})
    save_json(
        report_dir / "metrics.json",
        {
            "primary_eval_mode": train_config.primary_eval_mode,
            "val_best_macro_f1_supported": float(best_score),
            "test_benchmark": metrics_tb,
            "test_full": metrics_tf,
            "best_thresholds": thresholds_to_label_map(best_thresholds),
            "config": {
                "vision_model_id": model_config.vision_model_id,
                "text_model_id": model_config.text_model_id,
                "hidden_dim": model_config.hidden_dim,
                "train_neg_ratio": train_config.train_neg_ratio,
                "batch_size": train_config.batch_size,
                "epochs": train_config.epochs,
                "aux_op_loss_weight": train_config.aux_op_loss_weight,
                "unfreeze_text_layers": model_config.unfreeze_text_layers,
                "unfreeze_vision_layers": model_config.unfreeze_vision_layers,
            },
        },
    )
    save_json(report_dir / "label_map.json", {str(index): label for index, label in enumerate(ERROR_TYPES)})
    per_class_tb.to_csv(report_dir / "per_class_test_benchmark.csv", index=False)
    save_jsonl(prediction_rows, report_dir / "predictions_test_benchmark.jsonl")
    evidence_tb.to_csv(report_dir / "evidence_test_benchmark.csv", index=False)
    evidence_preview.head(50).to_csv(report_dir / "evidence_preview.csv", index=False)

    return {
        "report_dir": str(report_dir),
        "model_out": str(model_out),
        "best_score": float(best_score),
        "benchmark_metrics": metrics_tb,
        "full_metrics": metrics_tf,
    }


def finetune_v2(
    dataset_paths: DatasetPaths,
    report_dir: str | Path,
    model_in: str | Path,
    model_out: str | Path,
    model_config: ModelConfig,
    finetune_config: V2FinetuneConfig,
) -> Dict[str, Any]:
    device = resolve_device()
    seed_everything(finetune_config.seed)
    report_dir = Path(report_dir)
    model_in = Path(model_in)
    model_out = Path(model_out)
    ensure_dirs(report_dir, model_out.parent)

    row_bundle = load_row_bundle(
        dataset_paths,
        train_neg_ratio=finetune_config.train_neg_ratio,
        seed=finetune_config.seed,
        smoke_test=finetune_config.smoke_test,
        smoke_limit_originals=finetune_config.smoke_limit_originals,
    )
    loaders = build_dataloaders(
        row_bundle,
        dataset_paths,
        model_config,
        batch_size=finetune_config.batch_size,
        num_workers=finetune_config.num_workers,
        device=device,
    )
    model = build_model(model_config, device)
    model.load_state_dict(torch.load(model_in, map_location=device))

    train_targets = _collect_train_targets(loaders.train_loader)
    positives = train_targets.sum(axis=0)
    negatives = train_targets.shape[0] - positives
    pos_weight = (negatives / np.clip(positives, 1, None)).astype(np.float32)
    pos_weight = np.clip(pos_weight, 1.0, 8.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    criterion_tax = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=finetune_config.lr,
        weight_decay=finetune_config.weight_decay,
    )

    best_score = -1.0
    best_thresholds = [0.5] * len(ERROR_TYPES)
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, finetune_config.epochs + 1):
        model.train()
        running_loss = 0.0
        example_count = 0

        for batch in loaders.train_loader:
            src, tgt, input_ids, attention_mask, targets = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(finetune_config.use_amp and device.type == "cuda")):
                logits, _ = model(src, tgt, input_ids, attention_mask)
                loss = criterion_tax(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), finetune_config.grad_clip)
            optimizer.step()

            running_loss += float(loss.item()) * targets.shape[0]
            example_count += int(targets.shape[0])

        val_loss, y_true_v, y_prob_v = evaluate_taxonomy_only(
            model,
            loaders.val_loader_benchmark,
            criterion_tax,
            device,
            use_amp=finetune_config.use_amp,
        )
        thresholds_v = tune_thresholds(y_true_v, y_prob_v, grid=np.arange(0.10, 0.91, 0.05))
        y_pred_v = (y_prob_v >= np.array(thresholds_v).reshape(1, -1)).astype(int)
        metrics_v = compute_multilabel_metrics(y_true_v, y_prob_v, y_pred_v)
        score = metrics_v["macro_f1_supported"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(example_count, 1),
                "val_loss": val_loss,
                **{f"val_{key}": value for key, value in metrics_v.items()},
            }
        )

        if score > best_score:
            best_score = score
            best_thresholds = thresholds_v
            torch.save(model.state_dict(), model_out)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= finetune_config.patience:
                break

    model.load_state_dict(torch.load(model_out, map_location=device))
    _, y_true_b, y_prob_b = evaluate_taxonomy_only(
        model,
        loaders.test_loader_benchmark,
        criterion_tax,
        device,
        use_amp=finetune_config.use_amp,
    )
    y_pred_b = (y_prob_b >= np.array(best_thresholds).reshape(1, -1)).astype(int)
    metrics_b = compute_multilabel_metrics(y_true_b, y_prob_b, y_pred_b)
    per_class_b = compute_per_class_metrics(y_true_b, y_prob_b, y_pred_b)

    _, y_true_f, y_prob_f = evaluate_taxonomy_only(
        model,
        loaders.test_loader_full,
        criterion_tax,
        device,
        use_amp=finetune_config.use_amp,
    )
    y_pred_f = (y_prob_f >= np.array(best_thresholds).reshape(1, -1)).astype(int)
    metrics_f = compute_multilabel_metrics(y_true_f, y_prob_f, y_pred_f)

    save_json(report_dir / "train_history_v2.json", history)
    save_threshold_artifacts(report_dir, best_thresholds, prefer_array=True)
    save_json(
        report_dir / "metrics.json",
        {
            "v2_best_val_macro_f1_supported": float(best_score),
            "v2_benchmark_metrics": metrics_b,
            "v2_full_test_metrics": metrics_f,
            "best_thresholds": [float(value) for value in best_thresholds],
            "pos_weight": pos_weight.tolist(),
        },
    )
    per_class_b.to_csv(report_dir / "per_class_test_benchmark.csv", index=False)
    save_jsonl(build_vector_prediction_rows(y_true_b, y_pred_b, y_prob_b), report_dir / "predictions_test_benchmark.jsonl")

    return {
        "report_dir": str(report_dir),
        "model_out": str(model_out),
        "best_score": float(best_score),
        "benchmark_metrics": metrics_b,
        "full_metrics": metrics_f,
    }


def finetune_v3(
    dataset_paths: DatasetPaths,
    report_dir: str | Path,
    model_in: str | Path,
    model_out: str | Path,
    model_config: ModelConfig,
    finetune_config: V3FinetuneConfig,
) -> Dict[str, Any]:
    device = resolve_device()
    seed_everything(finetune_config.seed)
    report_dir = Path(report_dir)
    model_in = Path(model_in)
    model_out = Path(model_out)
    ensure_dirs(report_dir, model_out.parent)

    row_bundle = load_row_bundle(
        dataset_paths,
        train_neg_ratio=finetune_config.train_neg_ratio,
        seed=finetune_config.seed,
        smoke_test=finetune_config.smoke_test,
        smoke_limit_originals=finetune_config.smoke_limit_originals,
    )
    loaders = build_dataloaders(
        row_bundle,
        dataset_paths,
        model_config,
        batch_size=finetune_config.batch_size,
        num_workers=finetune_config.num_workers,
        device=device,
    )
    model = build_model(model_config, device)
    model.load_state_dict(torch.load(model_in, map_location=device))
    val_loader = loaders.val_loader_benchmark

    train_targets = _collect_train_targets(loaders.train_loader)
    positives = train_targets.sum(axis=0)
    negatives = train_targets.shape[0] - positives
    pos_weight = np.sqrt(negatives / np.clip(positives, 1, None)).astype(np.float32)
    pos_weight = np.clip(pos_weight, 1.0, 4.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    criterion_plain = nn.BCEWithLogitsLoss()
    criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion_op = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=finetune_config.lr,
        weight_decay=finetune_config.weight_decay,
    )

    best_score = -1.0
    best_thresholds = [0.5] * len(ERROR_TYPES)
    no_improve = 0
    history: List[Dict[str, Any]] = []

    op_probe = next(iter(loaders.train_loader))
    has_op_labels = _get_optional_op_ids(op_probe, device) is not None

    for epoch in range(1, finetune_config.epochs + 1):
        model.train()
        running_loss = 0.0
        example_count = 0

        for batch in loaders.train_loader:
            src, tgt, input_ids, attention_mask, targets = _batch_to_device(batch, device)
            op_ids = _get_optional_op_ids(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=(finetune_config.use_amp and device.type == "cuda")):
                logits, evidence = model(src, tgt, input_ids, attention_mask)
                loss_plain = criterion_plain(logits, targets)
                loss_weighted = criterion_weighted(logits, targets)
                loss = finetune_config.alpha_plain * loss_plain + finetune_config.alpha_weighted * loss_weighted
                if has_op_labels and op_ids is not None and "op_logits" in evidence:
                    loss = loss + finetune_config.op_aux_weight * criterion_op(evidence["op_logits"], op_ids)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), finetune_config.grad_clip)
            optimizer.step()

            running_loss += float(loss.item()) * targets.shape[0]
            example_count += int(targets.shape[0])

        val_loss, y_true_v, y_prob_v = evaluate_taxonomy_only(
            model,
            val_loader,
            criterion_plain,
            device,
            use_amp=finetune_config.use_amp,
            alpha_plain=finetune_config.alpha_plain,
            alpha_weighted=finetune_config.alpha_weighted,
            criterion_tax_weighted=criterion_weighted,
        )
        thresholds_v = tune_thresholds(y_true_v, y_prob_v, grid=np.arange(0.10, 0.91, 0.05))
        y_pred_v = (y_prob_v >= np.array(thresholds_v).reshape(1, -1)).astype(int)
        metrics_v = compute_multilabel_metrics(y_true_v, y_prob_v, y_pred_v)
        score = 0.70 * metrics_v["macro_f1_supported"] + 0.30 * metrics_v["mAP_micro"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(example_count, 1),
                "val_loss": val_loss,
                "val_score": score,
                **{f"val_{key}": value for key, value in metrics_v.items()},
            }
        )

        if score > best_score:
            best_score = score
            best_thresholds = thresholds_v
            torch.save(model.state_dict(), model_out)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= finetune_config.patience:
                break

    model.load_state_dict(torch.load(model_out, map_location=device))
    _, y_true_b, y_prob_b = evaluate_taxonomy_only(
        model,
        loaders.test_loader_benchmark,
        criterion_plain,
        device,
        use_amp=finetune_config.use_amp,
        alpha_plain=finetune_config.alpha_plain,
        alpha_weighted=finetune_config.alpha_weighted,
        criterion_tax_weighted=criterion_weighted,
    )
    y_pred_b = (y_prob_b >= np.array(best_thresholds).reshape(1, -1)).astype(int)
    metrics_b = compute_multilabel_metrics(y_true_b, y_prob_b, y_pred_b)
    per_class_b = compute_per_class_metrics(y_true_b, y_prob_b, y_pred_b)

    _, y_true_f, y_prob_f = evaluate_taxonomy_only(
        model,
        loaders.test_loader_full,
        criterion_plain,
        device,
        use_amp=finetune_config.use_amp,
        alpha_plain=finetune_config.alpha_plain,
        alpha_weighted=finetune_config.alpha_weighted,
        criterion_tax_weighted=criterion_weighted,
    )
    y_pred_f = (y_prob_f >= np.array(best_thresholds).reshape(1, -1)).astype(int)
    metrics_f = compute_multilabel_metrics(y_true_f, y_prob_f, y_pred_f)

    save_json(report_dir / "train_history_v3.json", history)
    save_threshold_artifacts(report_dir, best_thresholds, prefer_array=True)
    save_json(
        report_dir / "metrics.json",
        {
            "v3_best_val_composite_score": float(best_score),
            "v3_benchmark_metrics": metrics_b,
            "v3_full_test_metrics": metrics_f,
            "best_thresholds": [float(value) for value in best_thresholds],
            "pos_weight": pos_weight.tolist(),
            "loss_blend": {
                "plain_bce": finetune_config.alpha_plain,
                "weighted_bce": finetune_config.alpha_weighted,
                "op_aux_weight": finetune_config.op_aux_weight if has_op_labels else 0.0,
            },
        },
    )
    per_class_b.to_csv(report_dir / "per_class_test_benchmark.csv", index=False)
    save_jsonl(build_vector_prediction_rows(y_true_b, y_pred_b, y_prob_b), report_dir / "predictions_test_benchmark.jsonl")

    return {
        "report_dir": str(report_dir),
        "model_out": str(model_out),
        "best_score": float(best_score),
        "benchmark_metrics": metrics_b,
        "full_metrics": metrics_f,
    }


def build_inference_rows_from_originals(
    rows: Sequence[Dict[str, Any]],
    lang: str,
) -> List[Dict[str, Any]]:
    validate_lang(lang)
    inference_rows: List[Dict[str, Any]] = []
    for row in rows:
        instruction = str(row.get(f"instruction_{lang}") or "").strip()
        if not instruction:
            continue
        op_type = infer_prompt_operation(instruction)
        inference_rows.append(
            {
                "id": str(row["id"]),
                "lang": lang,
                "instruction": instruction,
                "source_path": normalize_relpath(str(row["source_path"])),
                "target_path": normalize_relpath(str(row["target_path"])),
                "taxonomy_labels": [str(label) for label in row.get("taxonomy_labels", [])],
                "target_vec": labels_to_multihot(row.get("taxonomy_labels", [])),
                "has_error": 1 if row.get("taxonomy_labels") else 0,
                "op_type": op_type,
                "op_type_id": OP_TO_IDX[op_type],
            }
        )
    return inference_rows


def run_inference(
    rows: Sequence[Dict[str, Any]],
    image_root: str | Path,
    checkpoint_path: str | Path,
    thresholds_path: str | Path,
    model_config: ModelConfig,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    use_amp: bool = True,
) -> List[Dict[str, Any]]:
    device = resolve_device()
    tokenizer = build_tokenizer(model_config)
    transform = build_eval_transform(model_config.img_size)
    dataset = EvidenceTaxonomyDataset(rows, image_root, tokenizer, model_config.max_text_len, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    thresholds = load_thresholds(thresholds_path)

    model = build_model(model_config, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            src, tgt, input_ids, attention_mask, targets = _batch_to_device(batch, device)
            del targets
            with torch.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                logits, evidence = model(src, tgt, input_ids, attention_mask)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            op_indices = torch.argmax(evidence["op_probs"], dim=-1).detach().cpu().tolist()
            for index in range(len(batch["id"])):
                prob_vector = [float(value) for value in probs[index].tolist()]
                pred_labels = select_labels_from_probabilities(prob_vector, thresholds)
                evidence_row = {
                    "src_presence": float(evidence["src_presence"][index].detach().cpu()),
                    "tgt_presence": float(evidence["tgt_presence"][index].detach().cpu()),
                    "correspondence_max": float(evidence["correspondence_max"][index].detach().cpu()),
                    "correspondence_mean": float(evidence["correspondence_mean"][index].detach().cpu()),
                    "local_change_score": float(evidence["local_change_score"][index].detach().cpu()),
                    "global_change_score": float(evidence["global_change_score"][index].detach().cpu()),
                    "outside_change_score": float(evidence["outside_change_score"][index].detach().cpu()),
                    "pred_op_type": OP_TYPES[int(op_indices[index])],
                }
                records.append(
                    {
                        "id": batch["id"][index],
                        "lang": batch["lang"][index],
                        "instruction": batch["instruction"][index],
                        "source_path": batch["source_path"][index],
                        "target_path": batch["target_path"][index],
                        "pred_labels": pred_labels,
                        "probabilities": thresholds_to_label_map(prob_vector),
                        "thresholds": thresholds_to_label_map(thresholds),
                        "evidence": evidence_row,
                        "reason": evidence_to_reason(evidence_row, pred_labels),
                    }
                )
    return records


def replay_benchmark_predictions(
    original_rows: Sequence[Dict[str, Any]],
    prediction_rows: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    benchmark_rows = expand_rows_by_language(filter_taxonomy_positive(original_rows))
    if len(benchmark_rows) != len(prediction_rows):
        raise ValueError(
            "Prediction row count does not match benchmark row count: "
            f"{len(prediction_rows)} vs {len(benchmark_rows)}"
        )
    outputs: List[Dict[str, Any]] = []
    for row, prediction in zip(benchmark_rows, prediction_rows):
        prob_vector = prediction.get("prob")
        if prob_vector is None and "probs" in prediction:
            prob_vector = [float(prediction["probs"][label]) for label in ERROR_TYPES]
        if prob_vector is None:
            raise ValueError("Prediction rows must contain either `prob` or `probs`.")
        outputs.append(
            {
                "id": row["id"],
                "lang": row["lang"],
                "instruction": row["instruction"],
                "source_path": row["source_path"],
                "target_path": row["target_path"],
                "gold_labels": row["taxonomy_labels"],
                "pred_labels": select_labels_from_probabilities(prob_vector, thresholds),
                "probabilities": thresholds_to_label_map(prob_vector),
                "thresholds": thresholds_to_label_map(thresholds),
            }
        )
    return outputs
