#!/usr/bin/env python
"""
Stage 1 Pretraining: Multilingual Binary Classifier for Image-Edit Evaluation.
Predicts edit success (sft=1) vs reject (preference_rejected=0) from:
- source image
- edited image
- multilingual prompt (en, hi, bn)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        logger.error(f"Metadata file {path} not found.")
        sys.exit(1)
        
    data = []
    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

class TriInputBinaryDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_root: str,
        tokenizer: AutoTokenizer,
        max_text_len: int = 128,
        transform: Optional[transforms.Compose] = None
    ):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.valid_samples = []
        missing_images = 0
        missing_text = 0
        
        for s in samples:
            src_path = os.path.join(self.image_root, s['source_path'])
            tgt_path = os.path.join(self.image_root, s['target_path'])
            
            if not s.get('instruction'):
                missing_text += 1
                logger.warning(f"Missing prompt text for sample {s['id']} lang={s['lang']}")
                continue
                
            if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                missing_images += 1
                logger.warning(f"Missing images for {s['id']}: src={src_path}, tgt={tgt_path}")
                continue
                
            self.valid_samples.append(s)
            
        if missing_images > 0:
            logger.warning(f"Total skipped due to missing images: {missing_images}")
        if missing_text > 0:
            logger.warning(f"Total skipped due to missing text: {missing_text}")
            
    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.valid_samples[idx]
        
        src_path = os.path.join(self.image_root, s['source_path'])
        tgt_path = os.path.join(self.image_root, s['target_path'])
        
        # Load images
        try:
            src_img = Image.open(src_path).convert('RGB')
            tgt_img = Image.open(tgt_path).convert('RGB')
        except Exception as e:
            # Fallback for corrupted images in training
            logger.error(f"Error loading images for {s['id']}: {e}")
            src_img = Image.new('RGB', (224, 224))
            tgt_img = Image.new('RGB', (224, 224))
        
        if self.transform:
            src_tensor = self.transform(src_img)
            tgt_tensor = self.transform(tgt_img)
            
        enc_text = self.tokenizer(
            s['instruction'],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "id": s["id"],
            "lang": s["lang"],
            "src_img": src_tensor,
            "tgt_img": tgt_tensor,
            "input_ids": enc_text["input_ids"].squeeze(0),
            "attention_mask": enc_text["attention_mask"].squeeze(0),
            "label": torch.tensor(s["label"], dtype=torch.float32)
        }

class TriInputBinaryClassifier(nn.Module):
    def __init__(self, text_model_name: str, freeze_vision: bool = True, freeze_text: bool = False):
        super().__init__()
        
        self.vision = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.vision.fc = nn.Identity()
        
        if freeze_vision:
            for param in self.vision.parameters():
                param.requires_grad = False
                
        self.text_enc = AutoModel.from_pretrained(text_model_name)
        if freeze_text:
            for param in self.text_enc.parameters():
                param.requires_grad = False
                
        concat_dim = (512 * 4) + self.text_enc.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, src_img, tgt_img, input_ids, attention_mask):
        f_src = self.vision(src_img)
        f_tgt = self.vision(tgt_img)
        f_diff = f_tgt - f_src
        f_abs_diff = torch.abs(f_diff)
        
        text_out = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)
        f_text = text_out.last_hidden_state[:, 0, :]
        
        f_fusion = torch.cat([f_src, f_tgt, f_diff, f_abs_diff, f_text], dim=1)
        return self.classifier(f_fusion).squeeze(-1)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in loader:
        src = batch["src_img"].to(device)
        tgt = batch["tgt_img"].to(device)
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(src, tgt, input_ids, mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = torch.sigmoid(logits).detach().cpu().numpy() >= 0.5
        all_preds.extend(preds.astype(int))
        all_labels.extend(labels.cpu().numpy().astype(int))
        
    return total_loss / max(len(loader.dataset), 1), all_preds, all_labels

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    res = {"preds": [], "probs": [], "labels": [], "langs": [], "ids": []}
    
    with torch.no_grad():
        for batch in loader:
            src = batch["src_img"].to(device)
            tgt = batch["tgt_img"].to(device)
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(src, tgt, input_ids, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            res["preds"].extend(preds)
            res["probs"].extend(probs)
            res["labels"].extend(labels.cpu().numpy().astype(int))
            res["langs"].extend(batch["lang"])
            res["ids"].extend(batch["id"])
            
    return total_loss / max(len(loader.dataset), 1), res

def compute_metrics(res: Dict[str, Any]) -> Dict[str, float]:
    y_true = np.array(res["labels"])
    y_pred = np.array(res["preds"])
    y_prob = np.array(res["probs"])
    langs = np.array(res["langs"])
    
    def safe_roc_auc(t, p):
        try: return float(roc_auc_score(t, p))
        except ValueError: return 0.0
            
    metrics = {
        "overall_acc": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "overall_f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "overall_prec": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "overall_rec": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "overall_roc_auc": safe_roc_auc(y_true, y_prob)
    }
    
    for l in ["en", "hi", "bn"]:
        m_idx = langs == l
        y_t, y_p = y_true[m_idx], y_pred[m_idx]
        metrics[f"lang_acc_{l}"] = float(accuracy_score(y_t, y_p)) if len(y_t) > 0 else 0.0
        metrics[f"lang_f1_{l}"] = float(f1_score(y_t, y_p, zero_division=0)) if len(y_t) > 0 else 0.0
        
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--text_model", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")
    parser.add_argument("--exclude_ids_json", type=str, default=None, help="JSON file containing list of IDs to exclude")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    args = parser.parse_args()

    if args.smoke:
        logger.info("SMOKE MODE ENABLED. Overriding args.")
        args.limit = 300
        args.epochs = 1
        args.batch_size = 8

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    excluded_ids = set()
    if args.exclude_ids_json:
        with open(args.exclude_ids_json, "r") as f:
            excluded_ids = set(json.load(f))
        logger.info(f"Loaded {len(excluded_ids)} IDs to exclude from training")

    logger.info("Loading metadata...")
    raw_data = load_jsonl(args.metadata)
        
    samples_by_id = {}
    for row in raw_data:
        if row["id"] in excluded_ids:
            continue
            
        stype = row.get("source_type")
        # Ensure only valid source types are used (Check #1)
        if stype not in ["sft", "preference_rejected"]:
            continue
            
        label = 1 if stype == "sft" else 0
        samples_by_id[row["id"]] = {
            "id": row["id"],
            "source_type": stype,
            "label": label,
            "source_path": row.get("source_path", ""),
            "target_path": row.get("target_path", ""),
            "instruction_en": row.get("instruction_en", ""),
            "instruction_hi": row.get("instruction_hi", ""),
            "instruction_bn": row.get("instruction_bn", "")
        }
        
    unique_ids = list(samples_by_id.keys())
    
    if args.limit:
        unique_ids = unique_ids[:args.limit]
    
    # Split by original ID to avoid leakage across language variations (Check #2)
    train_ids, temp_ids = train_test_split(unique_ids, test_size=(args.val_size + args.test_size), random_state=args.seed)
    # Rebalance validation and test
    val_test_ratio = args.test_size / (args.val_size + args.test_size)
    if len(temp_ids) > 1:
        val_ids, test_ids = train_test_split(temp_ids, test_size=val_test_ratio, random_state=args.seed)
    else:
        val_ids, test_ids = temp_ids, []
    
    logger.info(f"Split {len(unique_ids)} IDs -> train:{len(train_ids)} val:{len(val_ids)} test:{len(test_ids)}")

    def expand_multilingual(id_list: List[str]) -> List[Dict[str, Any]]:
        expanded = []
        for _id in id_list:
            s_data = samples_by_id[_id]
            for lang, text_key in [("en", "instruction_en"), ("hi", "instruction_hi"), ("bn", "instruction_bn")]:
                if s_data.get(text_key):
                    expanded.append({
                        "id": s_data["id"],
                        "lang": lang,
                        "instruction": s_data[text_key],
                        "source_path": s_data["source_path"],
                        "target_path": s_data["target_path"],
                        "label": s_data["label"],
                        "source_type": s_data["source_type"]
                    })
        return expanded
        
    train_data = expand_multilingual(train_ids)
    val_data = expand_multilingual(val_ids)
    test_data = expand_multilingual(test_ids)

    # Class imbalance weight
    train_labels = [x["label"] for x in train_data]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)] if n_pos > 0 else [1.0], dtype=torch.float32)

    # Compile dataset summary (Check #4)
    all_expanded = train_data + val_data + test_data
    lang_counts = {"en": 0, "hi": 0, "bn": 0}
    for r in all_expanded:
        if r["lang"] in lang_counts:
            lang_counts[r["lang"]] += 1
            
    # Original valid IDs considered
    original_samples = len(unique_ids)
    total_expanded = len(all_expanded)
    
    classes_cnt = {
        "sft (1)": sum(1 for v in samples_by_id.values() if v["label"] == 1 and v["id"] in unique_ids),
        "preference_rejected (0)": sum(1 for v in samples_by_id.values() if v["label"] == 0 and v["id"] in unique_ids),
    }

    summary = {
        "original_samples_count": original_samples,
        "total_expanded_rows_count": total_expanded,
        "splits_expanded_rows": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        },
        "classes_original_count": classes_cnt,
        "per_language_expanded_rows": lang_counts,
        "pos_weight_for_loss": float(pos_weight.item())
    }
    
    logger.info("Dataset Summary:")
    logger.info(json.dumps(summary, indent=2))
    with open(out_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if len(train_data) == 0:
        logger.error("No valid training data after filtering. Check metadata and limit flags.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    model = TriInputBinaryClassifier(args.text_model, args.freeze_vision, args.freeze_text)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pos_weight = pos_weight.to(device)

    train_ds = TriInputBinaryDataset(train_data, args.image_root, tokenizer, args.max_text_len)
    val_ds = TriInputBinaryDataset(val_data, args.image_root, tokenizer, args.max_text_len)
    test_ds = TriInputBinaryDataset(test_data, args.image_root, tokenizer, args.max_text_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # Using batch size of 1 for val/test when very little data due to smoke test
    val_loader = DataLoader(val_ds, batch_size=min(args.batch_size, len(val_ds)) if len(val_ds) > 0 else 1, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=min(args.batch_size, len(test_ds)) if len(test_ds) > 0 else 1, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = -1.0
    
    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_preds, tr_lbls = train_epoch(model, train_loader, optimizer, criterion, device)
        
        if len(val_ds) > 0:
            val_loss, val_res = evaluate(model, val_loader, criterion, device)
            val_metrics = compute_metrics(val_res)
            val_f1 = val_metrics["overall_f1"]
            logger.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_metrics['overall_acc']:.4f}")
        else:
            val_f1 = float(epoch)
            val_metrics = {}
        
        if val_f1 > best_val_f1 or epoch == 1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            
            if len(val_ds) > 0:
                with open(out_dir / "predictions_val.jsonl", "w") as f:
                    for idx in range(len(val_res["ids"])):
                        f.write(json.dumps({
                            "id": val_res["ids"][idx],
                            "lang": val_res["langs"][idx],
                            "true": int(val_res["labels"][idx]),
                            "pred": int(val_res["preds"][idx]),
                            "prob": float(val_res["probs"][idx])
                        }) + "\n")
            logger.info("Saved new best model checkpoint.")

    # Evaluate test dataset
    if len(test_ds) > 0:
        logger.info("Training complete. Evaluating Test Split...")
        model.load_state_dict(torch.load(out_dir / "best_model.pt"))
        test_loss, test_res = evaluate(model, test_loader, criterion, device)
        test_metrics = compute_metrics(test_res)
        
        logger.info(f"Test F1: {test_metrics['overall_f1']:.4f} | Test Acc: {test_metrics['overall_acc']:.4f}")
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
            
        with open(out_dir / "predictions_test.jsonl", "w") as f:
            for idx in range(len(test_res["ids"])):
                f.write(json.dumps({
                    "id": test_res["ids"][idx],
                    "lang": test_res["langs"][idx],
                    "true": int(test_res["labels"][idx]),
                    "pred": int(test_res["preds"][idx]),
                    "prob": float(test_res["probs"][idx])
                }) + "\n")
    else:
        logger.warning("No test data available for final evaluation.")

if __name__ == "__main__":
    main()
