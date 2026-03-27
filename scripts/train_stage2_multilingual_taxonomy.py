#!/usr/bin/env python3
"""Train Stage-2 multilingual tri-input taxonomy model.

Inputs per training row:
- source image
- edited image
- one prompt text (en/hi/bn)

Original-level split JSONL files are expanded into single-language rows at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils.schema import ERROR_TYPES
from scripts.utils.stage2_taxonomy import (
    EvalOutput,
    FocalBCEWithLogitsLoss,
    TriInputTaxonomyClassifier,
    TriInputTaxonomyDataset,
    binarize_with_thresholds,
    build_image_transform,
    compute_multilabel_metrics,
    compute_pos_weight,
    expand_rows_by_language,
    load_original_rows,
    load_stage1_compatible_weights,
    save_json,
    set_seed,
    tune_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 multilingual taxonomy model.")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, default="")
    parser.add_argument("--text_model", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on expanded rows per split for smoke checks.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run only val/test evaluation from a checkpoint.")
    parser.add_argument("--eval_checkpoint", type=str, default="", help="Checkpoint path for --eval_only (defaults to <out>/best_model.pt).")
    parser.add_argument("--thresholds_json", type=str, default="", help="Optional thresholds JSON for eval (defaults to <out>/best_thresholds.json).")
    return parser.parse_args()


def _load_thresholds(path: Path) -> List[float]:
    if not path.exists():
        return [0.5] * len(ERROR_TYPES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [float(payload.get(lbl, 0.5)) for lbl in ERROR_TYPES]
    return [0.5] * len(ERROR_TYPES)


def _as_str_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, tuple):
        return [str(v) for v in x]
    return [str(x)]


def run_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
) -> EvalOutput:
    model.eval()
    total_loss = 0.0
    n = 0
    y_true: List[np.ndarray] = []
    y_prob: List[np.ndarray] = []
    ids: List[str] = []
    langs: List[str] = []
    source_paths: List[str] = []
    target_paths: List[str] = []
    instructions: List[str] = []

    with torch.no_grad():
        for batch in loader:
            src = batch["src_img"].to(device, non_blocking=True)
            tgt = batch["tgt_img"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
                logits = model(src, tgt, input_ids, mask)
                loss = criterion(logits, target)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            total_loss += float(loss.item()) * target.shape[0]
            n += int(target.shape[0])
            y_true.append(target.detach().cpu().numpy())
            y_prob.append(probs)

            ids.extend(_as_str_list(batch["id"]))
            langs.extend(_as_str_list(batch["lang"]))
            source_paths.extend(_as_str_list(batch["source_path"]))
            target_paths.extend(_as_str_list(batch["target_path"]))
            instructions.extend(_as_str_list(batch["instruction"]))

    arr_true = np.concatenate(y_true, axis=0) if y_true else np.zeros((0, len(ERROR_TYPES)), dtype=np.float32)
    arr_prob = np.concatenate(y_prob, axis=0) if y_prob else np.zeros((0, len(ERROR_TYPES)), dtype=np.float32)
    return EvalOutput(
        loss=(total_loss / max(n, 1)),
        y_true=arr_true,
        y_prob=arr_prob,
        ids=ids,
        langs=langs,
        source_paths=source_paths,
        target_paths=target_paths,
        instructions=instructions,
    )


def _metrics_by_language(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, langs: List[str]) -> Dict[str, Dict[str, float]]:
    if y_true.shape[0] != len(langs):
        raise ValueError(f"Language vector length mismatch: len(langs)={len(langs)} vs rows={y_true.shape[0]}")

    out: Dict[str, Dict[str, float]] = {}
    lang_arr = np.array([str(x).strip() for x in langs], dtype=object)
    for lang in ("en", "hi", "bn"):
        mask = lang_arr == lang
        if int(mask.sum()) == 0:
            out[lang] = {"micro_f1": 0.0, "macro_f1": 0.0, "mAP_macro": 0.0, "mAP_micro": 0.0}
            continue
        out[lang] = compute_multilabel_metrics(y_true[mask], y_prob[mask], y_pred[mask])
    out["overall"] = compute_multilabel_metrics(y_true, y_prob, y_pred)
    return out


def _language_debug_summary(y_true: np.ndarray, langs: List[str]) -> Dict[str, Dict[str, Any]]:
    if y_true.shape[0] != len(langs):
        raise ValueError(f"Language vector length mismatch: len(langs)={len(langs)} vs rows={y_true.shape[0]}")

    lang_arr = np.array([str(x).strip() for x in langs], dtype=object)
    out: Dict[str, Dict[str, Any]] = {}
    for lang in ("en", "hi", "bn"):
        mask = lang_arr == lang
        rows = int(mask.sum())
        pos = int(y_true[mask].sum()) if rows > 0 else 0
        per_label = {
            ERROR_TYPES[i]: int(y_true[mask, i].sum()) if rows > 0 else 0
            for i in range(len(ERROR_TYPES))
        }
        out[lang] = {
            "eval_rows": rows,
            "positive_label_count": pos,
            "positive_label_counts_by_class": per_label,
        }
    overall_per_label = {ERROR_TYPES[i]: int(y_true[:, i].sum()) for i in range(len(ERROR_TYPES))}
    out["overall"] = {
        "eval_rows": int(y_true.shape[0]),
        "positive_label_count": int(y_true.sum()),
        "positive_label_counts_by_class": overall_per_label,
    }
    return out


def _save_predictions(path: Path, eval_out: EvalOutput, y_pred: np.ndarray, thresholds: List[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(len(eval_out.ids)):
            probs = eval_out.y_prob[i].tolist()
            gold = [ERROR_TYPES[j] for j, v in enumerate(eval_out.y_true[i].tolist()) if int(v) == 1]
            pred = [ERROR_TYPES[j] for j, v in enumerate(y_pred[i].tolist()) if int(v) == 1]
            rec = {
                "id": eval_out.ids[i],
                "lang": eval_out.langs[i],
                "source_path": eval_out.source_paths[i],
                "target_path": eval_out.target_paths[i],
                "instruction": eval_out.instructions[i],
                "gold_labels": gold,
                "pred_labels": pred,
                "probs": {ERROR_TYPES[j]: float(probs[j]) for j in range(len(ERROR_TYPES))},
                "thresholds": {ERROR_TYPES[j]: float(thresholds[j]) for j in range(len(ERROR_TYPES))},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", vars(args))

    train_originals = load_original_rows(args.train_jsonl)
    val_originals = load_original_rows(args.val_jsonl)
    test_originals = load_original_rows(args.test_jsonl)

    train_rows = expand_rows_by_language(train_originals)
    val_rows = expand_rows_by_language(val_originals)
    test_rows = expand_rows_by_language(test_originals)

    if args.limit > 0:
        train_rows = train_rows[: args.limit]
        val_rows = val_rows[: args.limit]
        test_rows = test_rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    transform = build_image_transform()

    train_ds = TriInputTaxonomyDataset(
        train_rows,
        image_root=args.image_root,
        tokenizer=tokenizer,
        max_text_len=args.max_text_len,
        transform=transform,
    )
    val_ds = TriInputTaxonomyDataset(
        val_rows,
        image_root=args.image_root,
        tokenizer=tokenizer,
        max_text_len=args.max_text_len,
        transform=transform,
    )
    test_ds = TriInputTaxonomyDataset(
        test_rows,
        image_root=args.image_root,
        tokenizer=tokenizer,
        max_text_len=args.max_text_len,
        transform=transform,
    )

    if len(train_ds) == 0:
        raise RuntimeError("No valid training rows after expansion/filtering.")

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0),
    )

    model = TriInputTaxonomyClassifier(
        text_model_name=args.text_model,
        freeze_vision=args.freeze_vision,
        freeze_text=args.freeze_text,
    )
    stage1_load_info = load_stage1_compatible_weights(model, args.stage1_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pos_weight = compute_pos_weight(train_rows).to(device)
    alpha = torch.clamp(pos_weight / torch.clamp(pos_weight.max(), min=1.0), min=0.1, max=1.0)

    if args.use_focal_loss:
        criterion = FocalBCEWithLogitsLoss(alpha=alpha, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_score = -1.0
    best_thresholds = [0.5] * len(ERROR_TYPES)
    history: List[Dict[str, Any]] = []

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            model.train()
            run_loss = 0.0
            n = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
            for batch in pbar:
                src = batch["src_img"].to(device, non_blocking=True)
                tgt = batch["tgt_img"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                    logits = model(src, tgt, input_ids, mask)
                    loss = criterion(logits, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                run_loss += float(loss.item()) * target.shape[0]
                n += int(target.shape[0])
                pbar.set_postfix({"train_loss": f"{(run_loss / max(n, 1)):.4f}"})

            scheduler.step()
            train_loss = run_loss / max(n, 1)

            val_eval = run_eval(model, val_loader, criterion, device, args.amp)
            val_thresholds = tune_thresholds(val_eval.y_true, val_eval.y_prob)
            val_pred = binarize_with_thresholds(val_eval.y_prob, val_thresholds)
            val_metrics = compute_multilabel_metrics(val_eval.y_true, val_eval.y_prob, val_pred)
            val_lang_metrics = _metrics_by_language(val_eval.y_true, val_eval.y_prob, val_pred, val_eval.langs)

            # Primary selection uses macro F1; tie-break with micro F1 + mAP.
            composite = (
                float(val_metrics.get("macro_f1", 0.0))
                + 0.5 * float(val_metrics.get("micro_f1", 0.0))
                + 0.25 * float(val_metrics.get("mAP_macro", 0.0))
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_eval.loss,
                "val_metrics": val_metrics,
                "val_metrics_by_lang": val_lang_metrics,
                "composite_score": composite,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(epoch_record)
            print(json.dumps(epoch_record, ensure_ascii=False))

            if args.save_every_epoch:
                torch.save(model.state_dict(), out_dir / f"checkpoint_epoch_{epoch}.pt")

            if composite > best_score:
                best_score = composite
                best_thresholds = val_thresholds
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                save_json(out_dir / "best_thresholds.json", {ERROR_TYPES[i]: float(best_thresholds[i]) for i in range(len(ERROR_TYPES))})
                _save_predictions(out_dir / "predictions_val.jsonl", val_eval, val_pred, best_thresholds)

        torch.save(model.state_dict(), out_dir / "final_model.pt")
        save_json(out_dir / "train_history.json", {"history": history, "best_composite": best_score})

        # Evaluate best checkpoint on val and test with fixed thresholds.
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    else:
        eval_ckpt = Path(args.eval_checkpoint) if args.eval_checkpoint else (out_dir / "best_model.pt")
        if not eval_ckpt.exists():
            raise FileNotFoundError(f"--eval_only requested but checkpoint not found: {eval_ckpt}")
        model.load_state_dict(torch.load(eval_ckpt, map_location=device))
        thresholds_path = Path(args.thresholds_json) if args.thresholds_json else (out_dir / "best_thresholds.json")
        best_thresholds = _load_thresholds(thresholds_path)
        if not thresholds_path.exists():
            # Fallback to validation-based tuning if threshold file is unavailable.
            val_eval_tmp = run_eval(model, val_loader, criterion, device, args.amp)
            best_thresholds = tune_thresholds(val_eval_tmp.y_true, val_eval_tmp.y_prob)

    val_eval = run_eval(model, val_loader, criterion, device, args.amp)
    val_pred = binarize_with_thresholds(val_eval.y_prob, best_thresholds)
    val_metrics = compute_multilabel_metrics(val_eval.y_true, val_eval.y_prob, val_pred)
    val_lang_metrics = _metrics_by_language(val_eval.y_true, val_eval.y_prob, val_pred, val_eval.langs)
    val_lang_debug = _language_debug_summary(val_eval.y_true, val_eval.langs)
    _save_predictions(out_dir / "predictions_val.jsonl", val_eval, val_pred, best_thresholds)

    test_eval = run_eval(model, test_loader, criterion, device, args.amp)
    test_pred = binarize_with_thresholds(test_eval.y_prob, best_thresholds)
    test_metrics = compute_multilabel_metrics(test_eval.y_true, test_eval.y_prob, test_pred)
    test_lang_metrics = _metrics_by_language(test_eval.y_true, test_eval.y_prob, test_pred, test_eval.langs)
    test_lang_debug = _language_debug_summary(test_eval.y_true, test_eval.langs)
    _save_predictions(out_dir / "predictions_test.jsonl", test_eval, test_pred, best_thresholds)

    label_map = {
        "labels": ERROR_TYPES,
        "index": {str(i): ERROR_TYPES[i] for i in range(len(ERROR_TYPES))},
        "name_to_index": {ERROR_TYPES[i]: i for i in range(len(ERROR_TYPES))},
    }
    save_json(out_dir / "label_map.json", label_map)

    metrics_payload = {
        "stage1_checkpoint_load": stage1_load_info,
        "dataset_sizes": {
            "train_expanded": len(train_ds),
            "val_expanded": len(val_ds),
            "test_expanded": len(test_ds),
        },
        "best_thresholds": {ERROR_TYPES[i]: float(best_thresholds[i]) for i in range(len(ERROR_TYPES))},
        "val": {
            "loss": val_eval.loss,
            "overall": val_metrics,
            "by_lang": val_lang_metrics,
            "by_lang_debug": val_lang_debug,
        },
        "test": {
            "loss": test_eval.loss,
            "overall": test_metrics,
            "by_lang": test_lang_metrics,
            "by_lang_debug": test_lang_debug,
        },
    }
    save_json(out_dir / "metrics.json", metrics_payload)
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
