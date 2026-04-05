"""Fine-tune the recovered evidence-grounded v2 model from the v1 lineage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.evidence_grounded_taxonomy import (
    DEFAULT_V1_MODEL_OUT,
    DEFAULT_V2_MODEL_OUT,
    DEFAULT_V2_REPORT_DIR,
    DatasetPaths,
    LOCAL_IMPORT_V1_MODEL,
    ModelConfig,
    V2FinetuneConfig,
    finetune_v2,
    resolve_best_existing_path,
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
    parser = argparse.ArgumentParser(description="Fine-tune the recovered evidence-grounded taxonomy evaluator to v2.")
    parser.add_argument("--config", default=None, help="Optional JSON config file with CLI field overrides.")
    parser.add_argument("--splits-dir", default="data/final/splits")
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--val-jsonl", default=None)
    parser.add_argument("--test-jsonl", default=None)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--report-dir", default=str(DEFAULT_V2_REPORT_DIR))
    parser.add_argument("--model-in", default=None)
    parser.add_argument("--model-out", default=str(DEFAULT_V2_MODEL_OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-originals", type=int, default=120)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--train-neg-ratio", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
    return apply_config_file(parser.parse_args())


def build_dataset_paths(args: argparse.Namespace) -> DatasetPaths:
    splits_dir = Path(args.splits_dir)
    return DatasetPaths(
        image_root=Path(args.image_root),
        train_jsonl=Path(args.train_jsonl) if args.train_jsonl else splits_dir / "train_originals.jsonl",
        val_jsonl=Path(args.val_jsonl) if args.val_jsonl else splits_dir / "val_originals.jsonl",
        test_jsonl=Path(args.test_jsonl) if args.test_jsonl else splits_dir / "test_originals.jsonl",
    )


def main() -> None:
    args = parse_args()
    model_in = args.model_in or str(resolve_best_existing_path(DEFAULT_V1_MODEL_OUT, [LOCAL_IMPORT_V1_MODEL]))
    result = finetune_v2(
        dataset_paths=build_dataset_paths(args),
        report_dir=args.report_dir,
        model_in=model_in,
        model_out=args.model_out,
        model_config=ModelConfig(
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
        finetune_config=V2FinetuneConfig(
            seed=args.seed,
            train_neg_ratio=args.train_neg_ratio,
            smoke_test=args.smoke_test,
            smoke_limit_originals=args.limit_originals,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            use_amp=not args.no_amp,
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
