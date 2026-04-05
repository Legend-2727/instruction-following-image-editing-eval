"""Run live EN/HI/BN evidence-grounded taxonomy inference for one or more rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.evidence_grounded_taxonomy import (
    DEFAULT_V3_MODEL_OUT,
    DEFAULT_V3_REPORT_DIR,
    LANGS,
    LOCAL_IMPORT_V3_MODEL,
    LOCAL_IMPORT_V3_THRESHOLDS,
    ModelConfig,
    build_inference_rows_from_originals,
    infer_prompt_operation,
    load_jsonl,
    normalize_relpath,
    resolve_best_existing_path,
    run_inference,
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
    parser = argparse.ArgumentParser(description="Run live evidence-grounded inference for one or more EN/HI/BN prompt rows.")
    parser.add_argument("--config", default=None, help="Optional JSON config file with CLI field overrides.")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--examples-jsonl", default=None, help="JSONL of original rows with instruction_en/hi/bn fields.")
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--target-path", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--id", default="demo_sample")
    parser.add_argument("--lang", default="en", choices=list(LANGS))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
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
    parser.add_argument("--output", default=None, help="Optional JSON or JSONL output path.")
    return apply_config_file(parser.parse_args())


def build_rows(args: argparse.Namespace):
    if args.examples_jsonl:
        original_rows = load_jsonl(args.examples_jsonl)[: args.limit]
        return build_inference_rows_from_originals(original_rows, args.lang)
    if args.source_path and args.target_path and args.instruction:
        op_type = infer_prompt_operation(args.instruction)
        return [
            {
                "id": args.id,
                "lang": args.lang,
                "instruction": args.instruction,
                "source_path": normalize_relpath(args.source_path),
                "target_path": normalize_relpath(args.target_path),
                "taxonomy_labels": [],
                "target_vec": [0.0] * 11,
                "has_error": 0,
                "op_type": op_type,
                "op_type_id": {
                    "remove": 0,
                    "add": 1,
                    "replace": 2,
                    "recolor": 3,
                    "style": 4,
                    "move": 5,
                    "background": 6,
                    "other": 7,
                }[op_type],
            }
        ]
    raise SystemExit("Provide either --examples-jsonl or all of --source-path, --target-path, and --instruction.")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or str(resolve_best_existing_path(DEFAULT_V3_MODEL_OUT, [LOCAL_IMPORT_V3_MODEL]))
    thresholds = args.thresholds or str(resolve_best_existing_path(DEFAULT_V3_REPORT_DIR / "best_thresholds.json", [LOCAL_IMPORT_V3_THRESHOLDS]))
    rows = build_rows(args)
    records = run_inference(
        rows=rows,
        image_root=args.image_root,
        checkpoint_path=checkpoint,
        thresholds_path=thresholds,
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )
    payload = records[0] if len(records) == 1 else records
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + ("\n" if not serialized.endswith("\n") else ""), encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
