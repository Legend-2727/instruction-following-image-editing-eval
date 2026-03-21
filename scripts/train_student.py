#!/usr/bin/env python
"""Train a minimal multilingual tri-input student classifier (smoke-test friendly)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import ensure_dirs, load_labels, load_metadata, normalize_relpath
from utils.schema import (
    ADHERENCE_LABELS,
    ADHERENCE_TO_IDX,
    ERROR_TYPES,
    LabelRecord,
)
from utils.text_encoder import load_translations


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train compact multilingual student classifier (adherence + taxonomy)."
    )
    parser.add_argument("--metadata", type=str, default="data/sample/metadata.jsonl")
    parser.add_argument("--labels", type=str, default="data/annotations/labels.jsonl")
    parser.add_argument("--translations_csv", type=str, default="data/sample/translations.csv")
    parser.add_argument("--out", type=str, default="runs/student_smoke")
    parser.add_argument("--text_mode", choices=["english_only", "original_only", "both"], default="english_only")
    parser.add_argument("--text_model", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_text_len", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0, help="Use first N matched samples (0 means all).")
    parser.add_argument("--overfit_n", type=int, default=0, help="Overfit on first N train samples only (0 means use all).")
    parser.add_argument("--taxonomy_threshold", type=float, default=0.5, help="Threshold for taxonomy label prediction.")
    parser.add_argument("--eval_split", choices=["train", "val"], default="val", help="Evaluate on train or val split (for memorization checks).")
    parser.add_argument("--require_original_text", action="store_true", help="Keep only samples with original-language text available.")
    parser.add_argument("--smoke", action="store_true", help="Small deterministic run: epochs=1, limit<=24, batch_size<=4.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VisionEncoder(nn.Module):
    """Frozen compact vision encoder based on ResNet18 global pooled features."""

    def __init__(self) -> None:
        super().__init__()
        try:
            weights = ResNet18_Weights.DEFAULT
            self.preprocess = weights.transforms()
        except Exception:
            weights = None
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = int(backbone.fc.in_features)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return feats.flatten(1)


class MultilingualTextEncoder(nn.Module):
    """Frozen multilingual text encoder with mean pooling."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.out_dim = int(self.model.config.hidden_size)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, texts: Sequence[str], device: torch.device, max_len: int) -> torch.Tensor:
        toks = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.model(**toks)
        hidden = out.last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return pooled


class StudentModel(nn.Module):
    """Tri-input student with fused source/edit/diff/text features and two heads."""

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int = 512,
        num_adherence: int = 3,
        num_errors: int = 11,
    ) -> None:
        super().__init__()
        fused_dim = (vision_dim * 4) + text_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.adherence_head = nn.Linear(hidden_dim // 2, num_adherence)
        self.taxonomy_head = nn.Linear(hidden_dim // 2, num_errors)

    def forward(
        self,
        src_feat: torch.Tensor,
        edit_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        diff = edit_feat - src_feat
        abs_diff = torch.abs(diff)
        fused = torch.cat([src_feat, edit_feat, diff, abs_diff, text_feat], dim=1)
        shared = self.fusion(fused)
        return {
            "adherence_logits": self.adherence_head(shared),
            "taxonomy_logits": self.taxonomy_head(shared),
        }


class StudentDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], image_transform: transforms.Compose):
        self.records = records
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        src_img = Image.open(row["source_image"]).convert("RGB")
        edit_img = Image.open(row["edited_image"]).convert("RGB")
        return {
            "id": row["id"],
            "source": self.image_transform(src_img),
            "edited": self.image_transform(edit_img),
            "text": row["text"],
            "adherence": int(row["adherence"]),
            "errors": torch.tensor(row["errors"], dtype=torch.float32),
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": [b["id"] for b in batch],
        "source": torch.stack([b["source"] for b in batch], dim=0),
        "edited": torch.stack([b["edited"] for b in batch], dim=0),
        "text": [b["text"] for b in batch],
        "adherence": torch.tensor([b["adherence"] for b in batch], dtype=torch.long),
        "errors": torch.stack([b["errors"] for b in batch], dim=0),
    }


def _first_nonempty(values: Sequence[Optional[str]]) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _pick_text(
    row: Dict[str, Any],
    text_mode: str,
    translations: Dict[str, Dict[str, str]],
) -> str:
    sid = str(row.get("id", ""))
    lang = str(row.get("lang", "en") or "en")
    trans_by_lang = translations.get(sid, {})

    english_text = _first_nonempty(
        [
            row.get("instruction_en"),
            row.get("prompt_en"),
            trans_by_lang.get("en"),
            row.get("prompt"),
            row.get("instruction"),
        ]
    )
    original_text = _first_nonempty(
        [
            row.get("instruction_original"),
            row.get("original_translation"),
            trans_by_lang.get(lang),
            row.get("prompt"),
            row.get("instruction"),
            english_text,
        ]
    )

    if text_mode == "english_only":
        return english_text
    if text_mode == "original_only":
        return original_text

    if english_text and original_text and english_text != original_text:
        return f"{original_text} [SEP] {english_text}"
    return english_text or original_text


def _get_text_source(
    row: Dict[str, Any],
    text_mode: str,
    translations: Dict[str, Dict[str, str]],
) -> str:
    """Detect which source was used for the final text in text_mode.
    Returns: 'has_english', 'has_original', or 'fallback_to_english'.
    """
    sid = str(row.get("id", ""))
    lang = str(row.get("lang", "en") or "en")
    trans_by_lang = translations.get(sid, {})

    english_text = _first_nonempty(
        [
            row.get("instruction_en"),
            row.get("prompt_en"),
            trans_by_lang.get("en"),
            row.get("prompt"),
            row.get("instruction"),
        ]
    )
    original_text = _first_nonempty(
        [
            row.get("instruction_original"),
            row.get("original_translation"),
            trans_by_lang.get(lang),
            row.get("prompt"),
            row.get("instruction"),
            english_text,
        ]
    )

    # Check if we have distinct original text
    has_original = original_text and original_text != english_text
    has_english = english_text != ""

    if text_mode == "english_only":
        return "has_english" if has_english else "fallback_to_english"
    if text_mode == "original_only":
        return "has_original" if has_original else "fallback_to_english"
    # text_mode == "both"
    if has_original and has_english:
        return "has_original"
    elif has_english:
        return "has_english"
    else:
        return "fallback_to_english"


def build_records(
    metadata_path: Path,
    labels_path: Path,
    text_mode: str,
    translations_csv: Optional[Path],
    limit: int,
    require_original_text: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    metadata_rows = load_metadata(metadata_path)
    labels = load_labels(labels_path)

    label_map: Dict[str, Dict[str, Any]] = {}
    for rec in labels:
        label_map[str(rec["id"])] = rec

    translations: Dict[str, Dict[str, str]] = {}
    if translations_csv and translations_csv.exists():
        translations = load_translations(translations_csv)

    data_root = metadata_path.parent
    records: List[Dict[str, Any]] = []
    text_source_counts: Dict[str, int] = {
        "has_english": 0,
        "has_original": 0,
        "fallback_to_english": 0,
    }

    for row in metadata_rows:
        sid = str(row.get("id", ""))
        if sid not in label_map:
            continue
        label = label_map[sid]

        orig_rel = normalize_relpath(str(row.get("orig_path", "")))
        edit_rel = normalize_relpath(str(row.get("edited_path", "")))
        source_image = (data_root / orig_rel).resolve()
        edited_image = (data_root / edit_rel).resolve()
        if not source_image.exists() or not edited_image.exists():
            continue

        adherence = label.get("adherence", "")
        if adherence not in ADHERENCE_TO_IDX:
            continue

        text = _pick_text(row=row, text_mode=text_mode, translations=translations)
        if not text:
            continue

        text_source = _get_text_source(row=row, text_mode=text_mode, translations=translations)

        # Filter if --require_original_text is set
        if require_original_text and text_source != "has_original":
            continue

        text_source_counts[text_source] += 1

        record = {
            "id": sid,
            "source_image": str(source_image),
            "edited_image": str(edited_image),
            "text": text,
            "adherence": ADHERENCE_TO_IDX[adherence],
            "errors": LabelRecord.from_dict(label).error_vector(),
            "lang": row.get("lang", "en"),
        }
        records.append(record)

        if limit > 0 and len(records) >= limit:
            break

    return records, text_source_counts


def split_records(
    records: List[Dict[str, Any]],
    val_size: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(records) < 2:
        raise ValueError("Need at least 2 samples after filtering.")

    y = [r["adherence"] for r in records]
    idxs = list(range(len(records)))

    try:
        tr_idx, va_idx = train_test_split(
            idxs,
            test_size=val_size,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        tr_idx, va_idx = train_test_split(
            idxs,
            test_size=val_size,
            random_state=seed,
            stratify=None,
        )

    train_records = [records[i] for i in tr_idx]
    val_records = [records[i] for i in va_idx]
    return train_records, val_records


def run_epoch(
    loader: DataLoader,
    model: StudentModel,
    vision_encoder: VisionEncoder,
    text_encoder: MultilingualTextEncoder,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_text_len: int,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    for batch in loader:
        src = batch["source"].to(device)
        edit = batch["edited"].to(device)
        y_adh = batch["adherence"].to(device)
        y_err = batch["errors"].to(device)

        with torch.no_grad():
            src_feat = vision_encoder(src)
            edit_feat = vision_encoder(edit)
            txt_feat = text_encoder(batch["text"], device=device, max_len=max_text_len)

        out = model(src_feat, edit_feat, txt_feat)
        loss_adh = ce_loss(out["adherence_logits"], y_adh)
        loss_err = bce_loss(out["taxonomy_logits"], y_err)
        loss = loss_adh + loss_err

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(steps, 1)


def evaluate(
    loader: DataLoader,
    model: StudentModel,
    vision_encoder: VisionEncoder,
    text_encoder: MultilingualTextEncoder,
    device: torch.device,
    max_text_len: int,
    taxonomy_threshold: float = 0.5,
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    all_ids: List[str] = []
    y_true_adh: List[int] = []
    y_pred_adh: List[int] = []
    y_true_err: List[np.ndarray] = []
    y_pred_err: List[np.ndarray] = []
    y_prob_err: List[np.ndarray] = []
    y_logits_err: List[np.ndarray] = []  # For debug stats

    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            src = batch["source"].to(device)
            edit = batch["edited"].to(device)
            y_adh = batch["adherence"].to(device)
            y_err = batch["errors"].to(device)

            src_feat = vision_encoder(src)
            edit_feat = vision_encoder(edit)
            txt_feat = text_encoder(batch["text"], device=device, max_len=max_text_len)

            out = model(src_feat, edit_feat, txt_feat)

            loss = ce_loss(out["adherence_logits"], y_adh) + bce_loss(out["taxonomy_logits"], y_err)
            total_loss += float(loss.item())
            steps += 1

            adh_pred = out["adherence_logits"].argmax(dim=1)
            tax_logits = out["taxonomy_logits"]
            err_prob = torch.sigmoid(tax_logits)
            err_pred = (err_prob >= taxonomy_threshold).int()

            all_ids.extend(batch["id"])
            y_true_adh.extend(y_adh.cpu().tolist())
            y_pred_adh.extend(adh_pred.cpu().tolist())
            y_true_err.extend(y_err.cpu().numpy())
            y_pred_err.extend(err_pred.cpu().numpy())
            y_prob_err.extend(err_prob.cpu().numpy())
            y_logits_err.extend(tax_logits.cpu().numpy())

    y_true_err_arr = np.array(y_true_err)
    y_pred_err_arr = np.array(y_pred_err)
    y_prob_err_arr = np.array(y_prob_err)
    y_logits_err_arr = np.array(y_logits_err)

    # Compute debug stats for taxonomy calibration
    mean_tax_logit = float(np.mean(y_logits_err_arr))
    mean_tax_sigmoid = float(np.mean(y_prob_err_arr))
    avg_pred_labels_per_sample = float(np.mean(y_pred_err_arr.sum(axis=1)))

    metrics = {
        "val_loss": total_loss / max(steps, 1),
        "adherence_accuracy": float(accuracy_score(y_true_adh, y_pred_adh)),
        "adherence_macro_f1": float(f1_score(y_true_adh, y_pred_adh, average="macro", zero_division=0)),
        "taxonomy_micro_f1": float(f1_score(y_true_err_arr, y_pred_err_arr, average="micro", zero_division=0)),
        "taxonomy_macro_f1": float(f1_score(y_true_err_arr, y_pred_err_arr, average="macro", zero_division=0)),
        "mean_taxonomy_logit": mean_tax_logit,
        "mean_taxonomy_sigmoid": mean_tax_sigmoid,
        "avg_predicted_labels_per_sample": avg_pred_labels_per_sample,
    }

    cols_with_pos = [c for c in range(y_true_err_arr.shape[1]) if y_true_err_arr[:, c].sum() > 0]
    if cols_with_pos:
        ap_vals = [
            average_precision_score(y_true_err_arr[:, c], y_prob_err_arr[:, c])
            for c in cols_with_pos
        ]
        metrics["taxonomy_mAP"] = float(np.mean(ap_vals))
    else:
        metrics["taxonomy_mAP"] = 0.0

    predictions: List[Dict[str, Any]] = []
    for i, sid in enumerate(all_ids):
        true_err_names = [ERROR_TYPES[j] for j, v in enumerate(y_true_err_arr[i].tolist()) if int(v) == 1]
        pred_err_names = [ERROR_TYPES[j] for j, v in enumerate(y_pred_err_arr[i].tolist()) if int(v) == 1]
        predictions.append(
            {
                "id": sid,
                "adherence_true": ADHERENCE_LABELS[y_true_adh[i]],
                "adherence_pred": ADHERENCE_LABELS[y_pred_adh[i]],
                "taxonomy_true": true_err_names,
                "taxonomy_pred": pred_err_names,
            }
        )

    return metrics, predictions


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        if args.limit <= 0:
            args.limit = 24
        else:
            args.limit = min(args.limit, 24)

    set_seed(args.seed)

    metadata_path = Path(args.metadata)
    labels_path = Path(args.labels)
    translations_csv = Path(args.translations_csv) if args.translations_csv else None
    out_dir = Path(args.out)
    ensure_dirs(out_dir)

    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata file: {metadata_path}")
    if not labels_path.exists():
        raise SystemExit(f"Missing labels file: {labels_path}")

    records, text_source_counts = build_records(
        metadata_path=metadata_path,
        labels_path=labels_path,
        text_mode=args.text_mode,
        translations_csv=translations_csv,
        limit=args.limit,
        require_original_text=args.require_original_text,
    )
    if len(records) < 4:
        raise SystemExit(f"Not enough matched samples after filtering: {len(records)}")

    if args.require_original_text:
        logger.info(
            "Text source filtering: has_original=%d has_english=%d fallback=%d",
            text_source_counts["has_original"],
            text_source_counts["has_english"],
            text_source_counts["fallback_to_english"],
        )

    train_records, val_records = split_records(
        records=records,
        val_size=args.val_size,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Samples: total=%d train=%d val=%d", len(records), len(train_records), len(val_records))

    vision_encoder = VisionEncoder().to(device)
    text_encoder = MultilingualTextEncoder(model_name=args.text_model).to(device)
    vision_encoder.eval()
    text_encoder.eval()

    model = StudentModel(
        vision_dim=vision_encoder.out_dim,
        text_dim=text_encoder.out_dim,
        num_adherence=len(ADHERENCE_LABELS),
        num_errors=len(ERROR_TYPES),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Apply overfit_n to restrict training to first N samples
    if args.overfit_n > 0 and args.overfit_n < len(train_records):
        logger.info("Overfitting on first %d train samples.", args.overfit_n)
        train_records = train_records[:args.overfit_n]

    train_ds = StudentDataset(records=train_records, image_transform=vision_encoder.preprocess)
    val_ds = StudentDataset(records=val_records, image_transform=vision_encoder.preprocess)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    best_metrics: Dict[str, float] = {}
    best_predictions: List[Dict[str, Any]] = []

    # Select evaluation loader based on --eval_split
    eval_loader = train_loader if args.eval_split == "train" else val_loader
    eval_split_name = args.eval_split

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            loader=train_loader,
            model=model,
            vision_encoder=vision_encoder,
            text_encoder=text_encoder,
            optimizer=optimizer,
            device=device,
            max_text_len=args.max_text_len,
        )

        eval_metrics, eval_predictions = evaluate(
            loader=eval_loader,
            model=model,
            vision_encoder=vision_encoder,
            text_encoder=text_encoder,
            device=device,
            max_text_len=args.max_text_len,
            taxonomy_threshold=args.taxonomy_threshold,
        )

        logger.info(
            "Epoch %d/%d | train_loss=%.4f %s_loss=%.4f adh_acc=%.4f tax_micro_f1=%.4f",
            epoch,
            args.epochs,
            train_loss,
            eval_split_name,
            eval_metrics["val_loss"],
            eval_metrics["adherence_accuracy"],
            eval_metrics["taxonomy_micro_f1"],
        )

        best_metrics = {"train_loss": float(train_loss), **eval_metrics}
        best_predictions = eval_predictions

    torch.save(model.state_dict(), out_dir / "student_model.pt")

    # Add text source counts to metrics for auditability
    best_metrics["text_source_counts"] = text_source_counts

    config_snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "adherence_labels": ADHERENCE_LABELS,
        "error_types": ERROR_TYPES,
        "n_total": len(records),
        "n_train": len(train_records),
        "n_val": len(val_records),
        "eval_split": eval_split_name,
        "text_source_counts": text_source_counts,
        "vision_encoder": "resnet18_frozen",
        "text_encoder": args.text_model,
    }

    save_json(out_dir / "config_snapshot.json", config_snapshot)
    save_json(out_dir / "metrics.json", best_metrics)
    save_jsonl(out_dir / "predictions.jsonl", best_predictions)

    logger.info("Saved model -> %s", out_dir / "student_model.pt")
    logger.info("Saved metrics -> %s", out_dir / "metrics.json")
    logger.info("Saved predictions -> %s", out_dir / "predictions.jsonl")
    logger.info("Saved config -> %s", out_dir / "config_snapshot.json")


if __name__ == "__main__":
    main()
