"""Evaluate recovered evidence-grounded checkpoints on the benchmark or full split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.evidence_grounded_taxonomy import (
    DEFAULT_V1_MODEL_OUT,
    DEFAULT_V1_REPORT_DIR,
    DEFAULT_V2_MODEL_OUT,
    DEFAULT_V2_REPORT_DIR,
    DEFAULT_V3_MODEL_OUT,
    DEFAULT_V3_REPORT_DIR,
    DatasetPaths,
    FocalBCEWithLogitsLoss,
    LOCAL_IMPORT_V1_MODEL,
    LOCAL_IMPORT_V1_THRESHOLDS,
    LOCAL_IMPORT_V2_MODEL,
    LOCAL_IMPORT_V2_THRESHOLDS,
    LOCAL_IMPORT_V3_MODEL,
    LOCAL_IMPORT_V3_THRESHOLDS,
    ModelConfig,
    V3FinetuneConfig,
    build_dataloaders,
    build_model,
    build_prediction_rows,
    build_vector_prediction_rows,
    compute_multilabel_metrics,
    compute_per_class_metrics,
    compute_pos_weight,
    evaluate_taxonomy_only,
    evaluate_with_evidence,
    load_row_bundle,
    load_thresholds,
    resolve_best_existing_path,
    resolve_device,
    save_json,
    save_jsonl,
)


def apply_config_file(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key, value in payload.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recovered evidence-grounded checkpoints on the benchmark or full held-out split.")
    parser.add_argument("--config", default=None, help="Optional JSON config file with CLI field overrides.")
    parser.add_argument("--version", choices=["v1", "v2", "v3"], default="v3")
    parser.add_argument("--splits-dir", default="data/final/splits")
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--val-jsonl", default=None)
    parser.add_argument("--test-jsonl", default=None)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-mode", choices=["benchmark", "full", "both"], default="both")
    parser.add_argument("--train-neg-ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-originals", type=int, default=120)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vision-model-id", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--text-model-id", default="xlm-roberta-base")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-text-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--unfreeze-text-layers", type=int, default=2)
    parser.add_argument("--unfreeze-vision-layers", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--v3-alpha-plain", type=float, default=V3FinetuneConfig.alpha_plain)
    parser.add_argument("--v3-alpha-weighted", type=float, default=V3FinetuneConfig.alpha_weighted)
    parser.add_argument("--v1-focal-gamma", type=float, default=1.5)
    parser.add_argument("--v1-aux-op-loss-weight", type=float, default=0.20)
    return apply_config_file(parser.parse_args())


def build_dataset_paths(args: argparse.Namespace) -> DatasetPaths:
    splits_dir = Path(args.splits_dir)
    return DatasetPaths(
        image_root=Path(args.image_root),
        train_jsonl=Path(args.train_jsonl) if args.train_jsonl else splits_dir / "train_originals.jsonl",
        val_jsonl=Path(args.val_jsonl) if args.val_jsonl else splits_dir / "val_originals.jsonl",
        test_jsonl=Path(args.test_jsonl) if args.test_jsonl else splits_dir / "test_originals.jsonl",
    )


def default_paths(version: str) -> tuple[Path, Path]:
    if version == "v1":
        return (
            resolve_best_existing_path(DEFAULT_V1_MODEL_OUT, [LOCAL_IMPORT_V1_MODEL]),
            resolve_best_existing_path(DEFAULT_V1_REPORT_DIR / "best_thresholds.json", [LOCAL_IMPORT_V1_THRESHOLDS]),
        )
    if version == "v2":
        return (
            resolve_best_existing_path(DEFAULT_V2_MODEL_OUT, [LOCAL_IMPORT_V2_MODEL]),
            resolve_best_existing_path(DEFAULT_V2_REPORT_DIR / "best_thresholds.json", [LOCAL_IMPORT_V2_THRESHOLDS]),
        )
    return (
        resolve_best_existing_path(DEFAULT_V3_MODEL_OUT, [LOCAL_IMPORT_V3_MODEL]),
        resolve_best_existing_path(DEFAULT_V3_REPORT_DIR / "best_thresholds.json", [LOCAL_IMPORT_V3_THRESHOLDS]),
    )


def maybe_write_outputs(args: argparse.Namespace, payload: dict, per_class=None, predictions=None, evidence=None) -> None:
    if not args.output_dir:
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "metrics.json", payload)
    if per_class is not None:
        per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    if predictions is not None:
        save_jsonl(predictions, output_dir / "predictions.jsonl")
    if evidence is not None:
        evidence.to_csv(output_dir / "evidence.csv", index=False)


def main() -> None:
    args = parse_args()
    checkpoint, thresholds_path = default_paths(args.version)
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    if args.thresholds:
        thresholds_path = Path(args.thresholds)
    thresholds = load_thresholds(thresholds_path)

    device = resolve_device()
    dataset_paths = build_dataset_paths(args)
    row_bundle = load_row_bundle(dataset_paths, args.train_neg_ratio, args.seed, args.smoke_test, args.limit_originals)
    loaders = build_dataloaders(
        row_bundle,
        dataset_paths,
        ModelConfig(
            vision_model_id=args.vision_model_id,
            text_model_id=args.text_model_id,
            img_size=args.img_size,
            max_text_len=args.max_text_len,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            unfreeze_text_layers=args.unfreeze_text_layers,
            unfreeze_vision_layers=args.unfreeze_vision_layers,
            local_files_only=args.local_files_only,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    model = build_model(
        ModelConfig(
            vision_model_id=args.vision_model_id,
            text_model_id=args.text_model_id,
            img_size=args.img_size,
            max_text_len=args.max_text_len,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            unfreeze_text_layers=args.unfreeze_text_layers,
            unfreeze_vision_layers=args.unfreeze_vision_layers,
            local_files_only=args.local_files_only,
        ),
        device,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    payload: dict = {
        "version": args.version,
        "checkpoint": str(checkpoint),
        "thresholds_path": str(thresholds_path),
    }

    if args.version == "v1":
        pos_weight = compute_pos_weight(row_bundle.train_rows, device)
        criterion_tax = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=args.v1_focal_gamma)
        criterion_op = nn.CrossEntropyLoss()

        if args.eval_mode in {"benchmark", "both"}:
            _, y_true_b, y_prob_b, _, langs_b, ids_b, evidence_b = evaluate_with_evidence(
                model,
                loaders.test_loader_benchmark,
                criterion_tax,
                criterion_op,
                device,
                args.v1_aux_op_loss_weight,
                use_amp=not args.no_amp,
            )
            y_pred_b = (y_prob_b >= np.array(thresholds).reshape(1, -1)).astype(int)
            metrics_b = compute_multilabel_metrics(y_true_b, y_prob_b, y_pred_b)
            payload["benchmark_metrics"] = metrics_b
            maybe_write_outputs(
                args,
                payload,
                per_class=compute_per_class_metrics(y_true_b, y_prob_b, y_pred_b),
                predictions=build_prediction_rows(ids_b, langs_b, y_true_b, y_pred_b, y_prob_b),
                evidence=evidence_b,
            )

        if args.eval_mode in {"full", "both"}:
            _, y_true_f, y_prob_f, _, _, _, _ = evaluate_with_evidence(
                model,
                loaders.test_loader_full,
                criterion_tax,
                criterion_op,
                device,
                args.v1_aux_op_loss_weight,
                use_amp=not args.no_amp,
            )
            y_pred_f = (y_prob_f >= np.array(thresholds).reshape(1, -1)).astype(int)
            payload["full_metrics"] = compute_multilabel_metrics(y_true_f, y_prob_f, y_pred_f)
    elif args.version == "v2":
        train_targets = np.concatenate([batch["target"].cpu().numpy() for batch in loaders.train_loader], axis=0)
        positives = train_targets.sum(axis=0)
        negatives = train_targets.shape[0] - positives
        pos_weight = np.clip((negatives / np.clip(positives, 1, None)).astype(np.float32), 1.0, 8.0)
        criterion_tax = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))

        if args.eval_mode in {"benchmark", "both"}:
            _, y_true_b, y_prob_b = evaluate_taxonomy_only(model, loaders.test_loader_benchmark, criterion_tax, device, use_amp=not args.no_amp)
            y_pred_b = (y_prob_b >= np.array(thresholds).reshape(1, -1)).astype(int)
            payload["benchmark_metrics"] = compute_multilabel_metrics(y_true_b, y_prob_b, y_pred_b)
            maybe_write_outputs(
                args,
                payload,
                per_class=compute_per_class_metrics(y_true_b, y_prob_b, y_pred_b),
                predictions=build_vector_prediction_rows(y_true_b, y_pred_b, y_prob_b),
            )

        if args.eval_mode in {"full", "both"}:
            _, y_true_f, y_prob_f = evaluate_taxonomy_only(model, loaders.test_loader_full, criterion_tax, device, use_amp=not args.no_amp)
            y_pred_f = (y_prob_f >= np.array(thresholds).reshape(1, -1)).astype(int)
            payload["full_metrics"] = compute_multilabel_metrics(y_true_f, y_prob_f, y_pred_f)
    else:
        train_targets = np.concatenate([batch["target"].cpu().numpy() for batch in loaders.train_loader], axis=0)
        positives = train_targets.sum(axis=0)
        negatives = train_targets.shape[0] - positives
        pos_weight = np.sqrt(negatives / np.clip(positives, 1, None)).astype(np.float32)
        pos_weight = np.clip(pos_weight, 1.0, 4.0)
        criterion_plain = nn.BCEWithLogitsLoss()
        criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))

        if args.eval_mode in {"benchmark", "both"}:
            _, y_true_b, y_prob_b = evaluate_taxonomy_only(
                model,
                loaders.test_loader_benchmark,
                criterion_plain,
                device,
                use_amp=not args.no_amp,
                alpha_plain=args.v3_alpha_plain,
                alpha_weighted=args.v3_alpha_weighted,
                criterion_tax_weighted=criterion_weighted,
            )
            y_pred_b = (y_prob_b >= np.array(thresholds).reshape(1, -1)).astype(int)
            payload["benchmark_metrics"] = compute_multilabel_metrics(y_true_b, y_prob_b, y_pred_b)
            maybe_write_outputs(
                args,
                payload,
                per_class=compute_per_class_metrics(y_true_b, y_prob_b, y_pred_b),
                predictions=build_vector_prediction_rows(y_true_b, y_pred_b, y_prob_b),
            )

        if args.eval_mode in {"full", "both"}:
            _, y_true_f, y_prob_f = evaluate_taxonomy_only(
                model,
                loaders.test_loader_full,
                criterion_plain,
                device,
                use_amp=not args.no_amp,
                alpha_plain=args.v3_alpha_plain,
                alpha_weighted=args.v3_alpha_weighted,
                criterion_tax_weighted=criterion_weighted,
            )
            y_pred_f = (y_prob_f >= np.array(thresholds).reshape(1, -1)).astype(int)
            payload["full_metrics"] = compute_multilabel_metrics(y_true_f, y_prob_f, y_pred_f)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
