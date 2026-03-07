#!/usr/bin/env python
"""step5_train_classifier.py — Train CLIP + XLM-RoBERTa multilingual error classifier.

Architecture:
  - CLIP ViT-B/32 (frozen) for dual image encoding (source + target)
  - XLM-RoBERTa-base for multilingual text encoding (supports ne/bn/hi natively)
  - Cross-attention fusion between image pair difference and text
  - 2-layer MLP classifier → 11-dim multi-hot output

This lightweight architecture is practical for low-resource deployment.

Input:  source_img + instruction (any language) + target_img
Output: 11-dim binary vector (multi-label error classification)

Usage
-----
    python scripts/step5_train_classifier.py --data data/magicbrush --epochs 15
    nohup python scripts/step5_train_classifier.py --data data/magicbrush --epochs 15 > logs/step5.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_jsonl, ensure_dirs
from utils.schema import ERROR_TYPES, NUM_ERROR_TYPES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# Dataset
class EditErrorDataset(Dataset):
    def __init__(self, records, data_dir, clip_preprocess, tokenizer, lang="en", max_text_len=128, augment=False):
        self.records = records
        self.data_dir = data_dir
        self.clip_preprocess = clip_preprocess
        self.tokenizer = tokenizer
        self.lang = lang
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        src_img = Image.open(self.data_dir / rec["source_path"]).convert("RGB")
        tgt_img = Image.open(self.data_dir / rec["target_path"]).convert("RGB")
        src_tensor = self.clip_preprocess(src_img)
        tgt_tensor = self.clip_preprocess(tgt_img)
        lang_key = f"instruction_{self.lang}" if self.lang != "en" else "instruction_en"
        instruction = rec.get(lang_key, rec.get("instruction_en", ""))
        text_inputs = self.tokenizer(instruction, max_length=self.max_text_len,
                                      padding="max_length", truncation=True, return_tensors="pt")
        error_vec = rec.get("error_label_vector", [0] * NUM_ERROR_TYPES)
        if len(error_vec) != NUM_ERROR_TYPES:
            error_vec = [0] * NUM_ERROR_TYPES
        return {
            "src_pixel_values": src_tensor,
            "tgt_pixel_values": tgt_tensor,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(error_vec, dtype=torch.float32),
            "id": rec["id"],
        }


# Model
class DualImageCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_feat, tgt_feat):
        src = src_feat.unsqueeze(1)
        tgt = tgt_feat.unsqueeze(1)
        attn_out, _ = self.cross_attn(tgt, src, src)
        out = self.norm(tgt + self.dropout(attn_out))
        return out.squeeze(1)


class ImageTextFusion(nn.Module):
    def __init__(self, img_dim, text_dim, hidden_dim=512, num_heads=8):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, img_feat, text_feat):
        img = self.img_proj(img_feat).unsqueeze(1)
        text = self.text_proj(text_feat).unsqueeze(1)
        fused, _ = self.cross_attn(img, text, text)
        out = self.norm(img + fused)
        return out.squeeze(1)


class MultilingualEditErrorClassifier(nn.Module):
    """CLIP ViT-B/32 (frozen) + XLM-RoBERTa + cross-attention + MLP."""

    def __init__(self, clip_model_name="ViT-B-32", clip_pretrained="laion2b_s34b_b79k",
                 xlmr_model_name="xlm-roberta-base", num_classes=NUM_ERROR_TYPES,
                 hidden_dim=512, dropout=0.3, freeze_vision=True, freeze_text=True):
        super().__init__()
        import open_clip
        from transformers import XLMRobertaModel

        logger.info("Loading CLIP: %s (%s)", clip_model_name, clip_pretrained)
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
        self.clip_visual = clip_model.visual
        self.vision_dim = clip_model.visual.output_dim

        if freeze_vision:
            for param in self.clip_visual.parameters():
                param.requires_grad = False
            logger.info("CLIP frozen (%d params)", sum(p.numel() for p in self.clip_visual.parameters()))

        logger.info("Loading XLM-RoBERTa: %s", xlmr_model_name)
        self.xlmr = XLMRobertaModel.from_pretrained(xlmr_model_name)
        self.text_dim = self.xlmr.config.hidden_size

        if freeze_text:
            for param in self.xlmr.parameters():
                param.requires_grad = False
            logger.info("XLM-R frozen (%d params)", sum(p.numel() for p in self.xlmr.parameters()))
        else:
            # Partial unfreezing: freeze all but last 2 encoder layers + pooler
            for param in self.xlmr.parameters():
                param.requires_grad = False
            for param in self.xlmr.encoder.layer[-2:].parameters():
                param.requires_grad = True
            if hasattr(self.xlmr, 'pooler') and self.xlmr.pooler is not None:
                for param in self.xlmr.pooler.parameters():
                    param.requires_grad = True
            unfrozen = sum(p.numel() for p in self.xlmr.parameters() if p.requires_grad)
            logger.info("XLM-R partial unfreeze: last 2 layers (%d params trainable)", unfrozen)

        self.dual_img_attn = DualImageCrossAttention(dim=self.vision_dim, num_heads=8, dropout=dropout)
        self.fusion = ImageTextFusion(img_dim=self.vision_dim * 3, text_dim=self.text_dim, hidden_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info("Model: vision_dim=%d text_dim=%d hidden=%d | trainable=%d/%d (%.1f%%)",
                    self.vision_dim, self.text_dim, hidden_dim, trainable, total, 100*trainable/total)

    def encode_image(self, pixel_values):
        with torch.no_grad():
            return self.clip_visual(pixel_values).float()

    def encode_text(self, input_ids, attention_mask):
        output = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]

    def forward(self, src_pixel_values, tgt_pixel_values, input_ids, attention_mask):
        src_feat = self.encode_image(src_pixel_values)
        tgt_feat = self.encode_image(tgt_pixel_values)
        cross_feat = self.dual_img_attn(src_feat, tgt_feat)
        diff_feat = tgt_feat - src_feat
        img_combined = torch.cat([src_feat, cross_feat, diff_feat], dim=-1)
        text_feat = self.encode_text(input_ids, attention_mask)
        fused = self.fusion(img_combined, text_feat)
        return self.classifier(fused)


# Focal loss for class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# Training utils
def compute_class_weights(records, num_classes=NUM_ERROR_TYPES):
    counts = np.zeros(num_classes)
    for r in records:
        vec = r.get("error_label_vector", [0]*num_classes)
        for i, v in enumerate(vec):
            counts[i] += v
    total = len(records)
    weights = np.where(counts > 0, total / (num_classes * counts), 1.0)
    return torch.tensor(np.clip(weights, 0.5, 10.0), dtype=torch.float32)


def evaluate(model, dataloader, device, class_weights=None):
    from sklearn.metrics import f1_score, hamming_loss
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0, 0
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device) if class_weights is not None else None)

    with torch.no_grad():
        for batch in dataloader:
            src_pv = batch["src_pixel_values"].to(device)
            tgt_pv = batch["tgt_pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(src_pv, tgt_pv, ids, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1
            preds = (torch.sigmoid(logits) > 0.3).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return {
        "loss": total_loss / max(n_batches, 1),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(all_labels, all_preds, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "hamming_loss": float(hamming_loss(all_labels, all_preds)),
        "exact_match": float(np.all(all_preds == all_labels, axis=1).mean()),
        "per_class_f1": {ERROR_TYPES[i]: float(per_class_f1[i]) for i in range(min(len(per_class_f1), len(ERROR_TYPES)))},
    }


def train(model, train_loader, val_loader, device, epochs=15, lr=1e-4,
          warmup_steps=100, patience=5, out_dir=Path("runs/classifier"),
          class_weights=None, grad_accum_steps=2):
    ensure_dirs(out_dir)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("Trainable params: %d", sum(p.numel() for p in trainable_params))

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs // grad_accum_steps
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    criterion = FocalLoss(alpha=1.0, gamma=2.0, pos_weight=class_weights.to(device) if class_weights is not None else None)
    scaler = torch.amp.GradScaler("cuda")
    best_f1, patience_counter = -1.0, 0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            src_pv = batch["src_pixel_values"].to(device)
            tgt_pv = batch["tgt_pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                logits = model(src_pv, tgt_pv, ids, mask)
                loss = criterion(logits, labels) / grad_accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * grad_accum_steps
            n_batches += 1
            pbar.set_postfix({"loss": f"{epoch_loss/n_batches:.4f}"})

        avg_train_loss = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device, class_weights)

        logger.info("Epoch %d/%d — train=%.4f val=%.4f macro_f1=%.4f w_f1=%.4f em=%.4f",
                    epoch+1, epochs, avg_train_loss, val_metrics["loss"],
                    val_metrics["macro_f1"], val_metrics["weighted_f1"], val_metrics["exact_match"])
        for et, f1 in val_metrics["per_class_f1"].items():
            logger.info("  %-30s F1=%.4f", et, f1)

        history.append({"epoch": epoch+1, "train_loss": avg_train_loss, **val_metrics})

        # Always save last checkpoint
        torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1, "metrics": val_metrics}, out_dir / "last_model.pt")

        if val_metrics["weighted_f1"] > best_f1:
            best_f1 = val_metrics["weighted_f1"]
            patience_counter = 0
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_f1": best_f1, "metrics": val_metrics}, out_dir / "best_model.pt")
            logger.info("Saved best model (w_f1=%.4f)", best_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch+1)
                break

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training done. Best w_f1=%.4f", best_f1)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/magicbrush")
    parser.add_argument("--out", default="runs/classifier")
    parser.add_argument("--clip_model", default="ViT-B-32")
    parser.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--xlmr_model", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train_lang", default="en")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_vision", action="store_true", default=True)
    parser.add_argument("--freeze_text", action="store_true", default=True)
    parser.add_argument("--unfreeze_text", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    ensure_dirs(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load annotations
    final_path = data_dir / "annotations_final.jsonl"
    vlm_path = data_dir / "vlm_annotations.jsonl"
    meta_path = data_dir / "metadata.jsonl"

    if final_path.exists():
        annotations_raw = load_jsonl(final_path)
        logger.info("Using final annotations: %d", len(annotations_raw))
    elif vlm_path.exists():
        annotations_raw = load_jsonl(vlm_path)
        logger.info("Using VLM annotations: %d", len(annotations_raw))
    else:
        logger.error("No annotations found. Run step3 first.")
        sys.exit(1)

    metadata = {m["id"]: m for m in load_jsonl(meta_path)}
    records = []
    for ann in annotations_raw:
        uid = ann["id"]
        if uid in metadata:
            records.append({**metadata[uid], **ann})

    records = [r for r in records if "error_label_vector" in r and len(r["error_label_vector"]) == NUM_ERROR_TYPES]
    logger.info("Records with valid labels: %d", len(records))

    if len(records) < 20:
        logger.error("Not enough records (need >= 20)")
        sys.exit(1)

    from sklearn.model_selection import train_test_split
    train_recs, test_recs = train_test_split(records, test_size=0.2, random_state=args.seed)
    train_recs, val_recs = train_test_split(train_recs, test_size=0.125, random_state=args.seed)
    logger.info("Split: train=%d val=%d test=%d", len(train_recs), len(val_recs), len(test_recs))

    from utils.io import save_jsonl
    save_jsonl(test_recs, out_dir / "test_set.jsonl")
    class_weights = compute_class_weights(train_recs)
    logger.info("Class weights: %s", [round(w, 2) for w in class_weights.tolist()])

    import open_clip
    from transformers import AutoTokenizer
    logger.info("Loading CLIP preprocessor...")
    _, clip_preprocess, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained)
    logger.info("Loading XLM-R tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.xlmr_model)

    lang = args.train_lang
    train_ds = EditErrorDataset(train_recs, data_dir, clip_preprocess, tokenizer, lang=lang)
    val_ds = EditErrorDataset(val_recs, data_dir, clip_preprocess, tokenizer, lang=lang)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    freeze_text = not args.unfreeze_text
    model = MultilingualEditErrorClassifier(
        clip_model_name=args.clip_model, clip_pretrained=args.clip_pretrained,
        xlmr_model_name=args.xlmr_model, num_classes=NUM_ERROR_TYPES,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        freeze_vision=args.freeze_vision, freeze_text=freeze_text,
    ).to(device)

    history = train(model, train_loader, val_loader, device, epochs=args.epochs,
                    lr=args.lr, patience=args.patience, out_dir=out_dir,
                    class_weights=class_weights, grad_accum_steps=args.grad_accum)

    # Test evaluation
    logger.info("=" * 60)
    logger.info("Test evaluation...")
    test_ds = EditErrorDataset(test_recs, data_dir, clip_preprocess, tokenizer, lang=lang)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    ckpt_path = out_dir / "best_model.pt" if (out_dir / "best_model.pt").exists() else out_dir / "last_model.pt"
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, class_weights)

    logger.info("Test: macro_f1=%.4f w_f1=%.4f em=%.4f hl=%.4f",
                test_metrics["macro_f1"], test_metrics["weighted_f1"],
                test_metrics["exact_match"], test_metrics["hamming_loss"])
    for et, f1 in test_metrics["per_class_f1"].items():
        logger.info("  %-30s F1=%.4f", et, f1)

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    config = vars(args)
    config.update({"num_train": len(train_recs), "num_val": len(val_recs),
                   "num_test": len(test_recs), "best_val_f1": float(ckpt["best_f1"])})
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("All saved -> %s", out_dir)


if __name__ == "__main__":
    main()
