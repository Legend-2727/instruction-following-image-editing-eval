"""Create small artifact-backed or live demo outputs for the released v3 path."""

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
    load_jsonl,
    load_thresholds,
    replay_benchmark_predictions,
    resolve_best_existing_path,
    run_inference,
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
    parser = argparse.ArgumentParser(description="Create small demo inference outputs for the recovered evidence-grounded v3 path.")
    parser.add_argument("--config", default=None, help="Optional JSON config file with CLI field overrides.")
    parser.add_argument("--mode", choices=["auto", "live", "artifact"], default="auto")
    parser.add_argument("--image-root", default=None, help="Required for live inference.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--demo-jsonl", default="data/final/splits/demo_infer_originals.jsonl")
    parser.add_argument("--artifact-metadata-jsonl", default="data/final/splits/test_originals.jsonl")
    parser.add_argument("--artifact-predictions-jsonl", default="evidence_grounded_taxonomy_eval_v3/predictions_test_benchmark.jsonl")
    parser.add_argument("--lang", default="en", choices=list(LANGS))
    parser.add_argument("--limit", type=int, default=2)
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
    parser.add_argument("--output-dir", default="reports/demo_inference")
    return apply_config_file(parser.parse_args())


def choose_mode(args: argparse.Namespace) -> str:
    if args.mode != "auto":
        return args.mode
    if not args.image_root:
        return "artifact"
    demo_rows = load_jsonl(args.demo_jsonl)
    if not demo_rows:
        return "artifact"
    first = demo_rows[0]
    image_path = Path(args.image_root) / Path(first["source_path"])
    return "live" if image_path.exists() else "artifact"


def write_outputs(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(records, output_dir / "demo_predictions.jsonl")
    for record in records:
        out_path = output_dir / f"{record['id']}_{record['lang']}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    mode = choose_mode(args)
    thresholds_path = args.thresholds or str(resolve_best_existing_path(DEFAULT_V3_REPORT_DIR / "best_thresholds.json", [LOCAL_IMPORT_V3_THRESHOLDS]))
    output_dir = Path(args.output_dir)

    if mode == "live":
        checkpoint = args.checkpoint or str(resolve_best_existing_path(DEFAULT_V3_MODEL_OUT, [LOCAL_IMPORT_V3_MODEL]))
        rows = build_inference_rows_from_originals(load_jsonl(args.demo_jsonl)[: args.limit], args.lang)
        records = run_inference(
            rows=rows,
            image_root=args.image_root,
            checkpoint_path=checkpoint,
            thresholds_path=thresholds_path,
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
    else:
        threshold_values = load_thresholds(thresholds_path)
        records = replay_benchmark_predictions(
            original_rows=load_jsonl(args.artifact_metadata_jsonl),
            prediction_rows=load_jsonl(args.artifact_predictions_jsonl),
            thresholds=threshold_values,
        )[: args.limit]
        for record in records:
            record["demo_mode"] = "artifact_replay"

    write_outputs(records, output_dir)
    print(json.dumps({"mode": mode, "output_dir": str(output_dir), "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
