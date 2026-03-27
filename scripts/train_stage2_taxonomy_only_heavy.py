#!/usr/bin/env python3
"""Train taxonomy-only heavy Stage-2 model on error-positive samples.

Pipeline intent:
- Stage-1 handles clean/no-error gating.
- This script trains heavy taxonomy classifier for error classes only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils.schema import ERROR_TYPES
from scripts.utils.stage2_taxonomy_heavy import (
    EvalOutput,
    FocalBCEWithLogitsLoss,
    TriInputTaxonomyHeavyClassifier,
    TriInputTaxonomyHeavyDataset,
    binarize_with_thresholds,
    build_image_transform,
    build_sample_weights,
    compute_multilabel_metrics,
    compute_per_class_metrics,
    compute_pos_weight,
    expand_rows_by_language,
    language_debug_summary,
    load_checkpoint_partial,
    load_original_rows,
    load_stage1_compatible_weights,
    metrics_by_language,
    prediction_diagnostics,
    save_json,
    set_seed,
    tune_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train taxonomy-only heavy Stage-2 model.")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, default="")

    parser.add_argument("--vision_backbone", type=str, default="vit_b16", choices=["vit_b16", "resnet50"])
    parser.add_argument("--text_model", type=str, default="xlm-roberta-base")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4, help="Legacy fallback LR.")
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--lr_text", type=float, default=2e-5)
    parser.add_argument("--lr_vision", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--freeze_vision", dest="freeze_vision", action="store_true", default=True)
    parser.add_argument("--unfreeze_vision", dest="freeze_vision", action="store_false")
    parser.add_argument("--freeze_text", action="store_true")
    parser.add_argument("--unfreeze_text_last_n_layers", type=int, default=2)
    parser.add_argument("--warmup_epochs", type=int, default=1)

    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--use_weighted_sampler", action="store_true")

    parser.add_argument("--include_clean_negative_ratio", type=float, default=0.0)

    parser.add_argument("--tune_thresholds", action="store_true")
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="macro_f1_supported",
        choices=["macro_f1", "macro_f1_supported", "mAP_macro", "mAP_macro_supported"],
    )
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_checkpoint", type=str, default="")
    parser.add_argument("--thresholds_json", type=str, default="")
    return parser.parse_args()


def _is_taxonomy_positive(row: Dict[str, Any]) -> bool:
    labels = row.get("taxonomy_labels", [])
    return isinstance(labels, list) and len(labels) > 0


def _filter_for_taxonomy_training(
    rows: Sequence[Dict[str, Any]],
    include_clean_negative_ratio: float,
    seed: int,
    split_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pos = [r for r in rows if _is_taxonomy_positive(r)]
    neg = [r for r in rows if not _is_taxonomy_positive(r)]

    selected_neg: List[Dict[str, Any]] = []
    if split_name == "train" and include_clean_negative_ratio > 0 and len(pos) > 0 and len(neg) > 0:
        k = min(len(neg), int(round(len(pos) * include_clean_negative_ratio)))
        rng = random.Random(seed)
        selected_neg = rng.sample(neg, k)

    out = pos + selected_neg
    rng2 = random.Random(seed + 17)
    rng2.shuffle(out)

    stats = {
        "split": split_name,
        "input_rows": len(rows),
        "positive_rows": len(pos),
        "clean_negative_rows": len(neg),
        "selected_negative_rows": len(selected_neg),
        "output_rows": len(out),
        "include_clean_negative_ratio": include_clean_negative_ratio if split_name == "train" else 0.0,
    }
    return out, stats


def _class_support_from_originals(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {lbl: 0 for lbl in ERROR_TYPES}
    for row in rows:
        for lbl in row.get("taxonomy_labels", []):
            if lbl in counts:
                counts[lbl] += 1
    return counts


def _split_support_report(train_rows: List[Dict[str, Any]], val_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_support = _class_support_from_originals(train_rows)
    val_support = _class_support_from_originals(val_rows)
    test_support = _class_support_from_originals(test_rows)

    val_zero = [k for k, v in val_support.items() if v == 0]
    test_zero = [k for k, v in test_support.items() if v == 0]

    warnings: List[str] = []
    if val_zero:
        warnings.append(f"Validation has zero support for classes: {val_zero}")
    if test_zero:
        warnings.append(f"Test has zero support for classes: {test_zero}")

    return {
        "train_support": train_support,
        "val_support": val_support,
        "test_support": test_support,
        "warnings": warnings,
    }


def _load_thresholds(path: Path) -> List[Optional[float]]:
    if not path.exists():
        return [0.5] * len(ERROR_TYPES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        out: List[Optional[float]] = []
        for lbl in ERROR_TYPES:
            v = payload.get(lbl, 0.5)
            out.append(None if v is None else float(v))
        return out
    return [0.5] * len(ERROR_TYPES)


def _serialize_thresholds(thresholds: List[Optional[float]]) -> Dict[str, Optional[float]]:
    return {ERROR_TYPES[i]: (None if thresholds[i] is None else float(thresholds[i])) for i in range(len(ERROR_TYPES))}


def _as_str_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, tuple):
        return [str(v) for v in x]
    return [str(x)]


def _configure_trainable_phase(
    model: TriInputTaxonomyHeavyClassifier,
    warmup_active: bool,
    freeze_vision: bool,
    freeze_text: bool,
    unfreeze_text_last_n_layers: int,
) -> Dict[str, Any]:
    vision_trainable = (not freeze_vision) and (not warmup_active)
    model.set_vision_trainable(vision_trainable)
    text_info = model.configure_text_trainability(
        freeze_text=freeze_text,
        unfreeze_last_n_layers=unfreeze_text_last_n_layers,
    )
    model.set_head_trainable(True)
    return {
        "warmup_active": bool(warmup_active),
        "vision_trainable": bool(vision_trainable),
        "text_trainable": bool(not freeze_text),
        "text_policy": text_info,
    }


def _build_optimizer(
    model: TriInputTaxonomyHeavyClassifier,
    args: argparse.Namespace,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR, Dict[str, Any]]:
    head_modules = [model.image_proj, model.text_proj, model.text_gate, model.fusion_in, model.fusion_blocks, model.taxonomy_head]
    head_params = [p for m in head_modules for p in m.parameters() if p.requires_grad]
    text_params = [p for p in model.text_enc.parameters() if p.requires_grad]
    vision_params = [p for p in model.vision.parameters() if p.requires_grad]

    lr_head = float(args.lr_head if args.lr_head > 0 else args.lr)
    lr_text = float(args.lr_text if args.lr_text >= 0 else args.lr)
    lr_vision = float(args.lr_vision if args.lr_vision >= 0 else args.lr)

    param_groups: List[Dict[str, Any]] = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "group_name": "head"})
    if text_params and lr_text > 0:
        param_groups.append({"params": text_params, "lr": lr_text, "group_name": "text"})
    if vision_params and lr_vision > 0:
        param_groups.append({"params": vision_params, "lr": lr_vision, "group_name": "vision"})

    if not param_groups:
        raise RuntimeError("No trainable parameters with positive learning rate.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    group_info = {
        "head": {"lr": lr_head, "param_count": int(sum(p.numel() for p in head_params))},
        "text": {"lr": lr_text, "param_count": int(sum(p.numel() for p in text_params))},
        "vision": {"lr": lr_vision, "param_count": int(sum(p.numel() for p in vision_params))},
    }
    return optimizer, scheduler, group_info


def run_eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, amp: bool) -> EvalOutput:
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


def _save_predictions(path: Path, eval_out: EvalOutput, y_pred: np.ndarray, thresholds: List[Optional[float]]) -> None:
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
                "thresholds": {
                    ERROR_TYPES[j]: (None if thresholds[j] is None else float(thresholds[j]))
                    for j in range(len(ERROR_TYPES))
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", vars(args))

    train_originals_all = load_original_rows(args.train_jsonl)
    val_originals_all = load_original_rows(args.val_jsonl)
    test_originals_all = load_original_rows(args.test_jsonl)

    train_originals, train_filter_stats = _filter_for_taxonomy_training(
        train_originals_all,
        include_clean_negative_ratio=max(0.0, float(args.include_clean_negative_ratio)),
        seed=args.seed,
        split_name="train",
    )
    val_originals, val_filter_stats = _filter_for_taxonomy_training(
        val_originals_all,
        include_clean_negative_ratio=0.0,
        seed=args.seed,
        split_name="val",
    )
    test_originals, test_filter_stats = _filter_for_taxonomy_training(
        test_originals_all,
        include_clean_negative_ratio=0.0,
        seed=args.seed,
        split_name="test",
    )

    support_report = _split_support_report(train_originals, val_originals, test_originals)
    save_json(out_dir / "split_support.json", {
        "filter_stats": {
            "train": train_filter_stats,
            "val": val_filter_stats,
            "test": test_filter_stats,
        },
        **support_report,
    })
    for msg in support_report.get("warnings", []):
        print(json.dumps({"split_warning": msg}, ensure_ascii=False))

    train_rows = expand_rows_by_language(train_originals)
    val_rows = expand_rows_by_language(val_originals)
    test_rows = expand_rows_by_language(test_originals)

    if args.limit > 0:
        train_rows = train_rows[: args.limit]
        val_rows = val_rows[: args.limit]
        test_rows = test_rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    transform = build_image_transform(224)

    train_ds = TriInputTaxonomyHeavyDataset(train_rows, args.image_root, tokenizer, args.max_text_len, transform)
    val_ds = TriInputTaxonomyHeavyDataset(val_rows, args.image_root, tokenizer, args.max_text_len, transform)
    test_ds = TriInputTaxonomyHeavyDataset(test_rows, args.image_root, tokenizer, args.max_text_len, transform)

    if len(train_ds) == 0:
        raise RuntimeError("No valid training rows after taxonomy-only filtering and expansion.")

    pin_mem = torch.cuda.is_available()
    sampler = None
    train_shuffle = True
    if args.use_weighted_sampler:
        weights = build_sample_weights(train_ds.rows)
        if len(weights) > 0:
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            train_shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
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

    model = TriInputTaxonomyHeavyClassifier(
        vision_backbone=args.vision_backbone,
        text_model_name=args.text_model,
        freeze_vision=False,
        freeze_text=False,
        fusion_dim=1024,
        dropout=0.2,
    )
    stage1_load_info = load_stage1_compatible_weights(model, args.stage1_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pos_weight = compute_pos_weight(train_ds.rows).to(device)
    alpha = torch.clamp(pos_weight / torch.clamp(pos_weight.max(), min=1.0), min=0.1, max=1.0)

    if args.use_focal_loss:
        criterion = FocalBCEWithLogitsLoss(alpha=alpha, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    phase_info = _configure_trainable_phase(
        model=model,
        warmup_active=(args.warmup_epochs > 0),
        freeze_vision=args.freeze_vision,
        freeze_text=args.freeze_text,
        unfreeze_text_last_n_layers=args.unfreeze_text_last_n_layers,
    )
    optimizer, scheduler, opt_group_info = _build_optimizer(model, args)

    best_score = -1.0
    best_thresholds: List[Optional[float]] = [0.5] * len(ERROR_TYPES)
    history: List[Dict[str, Any]] = []
    no_improve_epochs = 0

    if not args.eval_only:
        prev_warmup_state: Optional[bool] = None
        for epoch in range(1, args.epochs + 1):
            warmup_active = bool(epoch <= max(args.warmup_epochs, 0))
            if prev_warmup_state is None or warmup_active != prev_warmup_state:
                phase_info = _configure_trainable_phase(
                    model=model,
                    warmup_active=warmup_active,
                    freeze_vision=args.freeze_vision,
                    freeze_text=args.freeze_text,
                    unfreeze_text_last_n_layers=args.unfreeze_text_last_n_layers,
                )
                optimizer, scheduler, opt_group_info = _build_optimizer(model, args)
                prev_warmup_state = warmup_active
                print(json.dumps({"phase_change": phase_info, "optimizer_groups": opt_group_info}, ensure_ascii=False))

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
            if args.tune_thresholds:
                val_thresholds = tune_thresholds(val_eval.y_true, val_eval.y_prob)
            else:
                val_thresholds = [0.5 for _ in ERROR_TYPES]
            val_pred = binarize_with_thresholds(val_eval.y_prob, val_thresholds)
            val_metrics = compute_multilabel_metrics(val_eval.y_true, val_eval.y_prob, val_pred)
            val_diag = prediction_diagnostics(val_eval.y_prob, val_pred)

            primary_metric = float(val_metrics.get(args.early_stop_metric, 0.0))
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_eval.loss,
                "val_metrics": val_metrics,
                "val_pred_diagnostics": val_diag,
                "primary_metric": primary_metric,
                "optimizer_groups": opt_group_info,
            }
            history.append(epoch_record)
            print(json.dumps(epoch_record, ensure_ascii=False))

            if args.save_every_epoch:
                torch.save(model.state_dict(), out_dir / f"checkpoint_epoch_{epoch}.pt")

            if primary_metric > best_score:
                best_score = primary_metric
                best_thresholds = val_thresholds
                no_improve_epochs = 0
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                save_json(out_dir / "best_thresholds.json", _serialize_thresholds(best_thresholds))
                _save_predictions(out_dir / "predictions_val.jsonl", val_eval, val_pred, best_thresholds)
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= max(args.patience, 1):
                print(json.dumps({"early_stopping": True, "epoch": epoch, "patience": args.patience}, ensure_ascii=False))
                break

        torch.save(model.state_dict(), out_dir / "final_model.pt")
        save_json(out_dir / "train_history.json", {"history": history, "best_primary_metric": best_score})

        best_model_path = out_dir / "best_model.pt"
        if best_model_path.exists():
            best_ckpt_report = load_checkpoint_partial(model, str(best_model_path))
            print(json.dumps({"best_checkpoint_reload": best_ckpt_report}, ensure_ascii=False))
    else:
        eval_ckpt = Path(args.eval_checkpoint) if args.eval_checkpoint else (out_dir / "best_model.pt")
        if not eval_ckpt.exists():
            raise FileNotFoundError(f"--eval_only requested but checkpoint not found: {eval_ckpt}")
        eval_ckpt_report = load_checkpoint_partial(model, str(eval_ckpt))
        print(json.dumps({"eval_checkpoint_load": eval_ckpt_report}, ensure_ascii=False))
        thresholds_path = Path(args.thresholds_json) if args.thresholds_json else (out_dir / "best_thresholds.json")
        best_thresholds = _load_thresholds(thresholds_path)
        if not thresholds_path.exists() and args.tune_thresholds:
            val_eval_tmp = run_eval(model, val_loader, criterion, device, args.amp)
            best_thresholds = tune_thresholds(val_eval_tmp.y_true, val_eval_tmp.y_prob)

    val_eval = run_eval(model, val_loader, criterion, device, args.amp)
    val_pred = binarize_with_thresholds(val_eval.y_prob, best_thresholds)
    val_metrics = compute_multilabel_metrics(val_eval.y_true, val_eval.y_prob, val_pred)
    val_pred_diag = prediction_diagnostics(val_eval.y_prob, val_pred)
    val_metrics_by_lang = metrics_by_language(val_eval.y_true, val_eval.y_prob, val_pred, val_eval.langs)
    val_class_metrics = compute_per_class_metrics(val_eval.y_true, val_eval.y_prob, val_pred)
    val_lang_debug = language_debug_summary(val_eval.y_true, val_eval.langs)
    _save_predictions(out_dir / "predictions_val.jsonl", val_eval, val_pred, best_thresholds)

    test_eval = run_eval(model, test_loader, criterion, device, args.amp)
    test_pred = binarize_with_thresholds(test_eval.y_prob, best_thresholds)
    test_metrics = compute_multilabel_metrics(test_eval.y_true, test_eval.y_prob, test_pred)
    test_pred_diag = prediction_diagnostics(test_eval.y_prob, test_pred)
    test_metrics_by_lang = metrics_by_language(test_eval.y_true, test_eval.y_prob, test_pred, test_eval.langs)
    test_class_metrics = compute_per_class_metrics(test_eval.y_true, test_eval.y_prob, test_pred)
    test_lang_debug = language_debug_summary(test_eval.y_true, test_eval.langs)
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
        "split_support": support_report,
        "train_setup": {
            "phase_info_initial": phase_info,
            "optimizer_groups_initial": opt_group_info,
            "use_weighted_sampler": bool(args.use_weighted_sampler),
            "use_focal_loss": bool(args.use_focal_loss),
            "taxonomy_only": True,
        },
        "best_thresholds": _serialize_thresholds(best_thresholds),
        "val": {
            "loss": val_eval.loss,
            "overall": val_metrics,
            "prediction_diagnostics": val_pred_diag,
            "by_lang": val_metrics_by_lang,
            "by_lang_debug": val_lang_debug,
            "per_class": val_class_metrics,
        },
        "test": {
            "loss": test_eval.loss,
            "overall": test_metrics,
            "prediction_diagnostics": test_pred_diag,
            "by_lang": test_metrics_by_lang,
            "by_lang_debug": test_lang_debug,
            "per_class": test_class_metrics,
        },
    }
    save_json(out_dir / "metrics.json", metrics_payload)
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
