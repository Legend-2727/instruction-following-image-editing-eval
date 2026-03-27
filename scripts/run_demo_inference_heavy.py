#!/usr/bin/env python3
"""Run heavy Stage-2 inference on demo_infer originals and export report files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils.schema import ERROR_TYPES
from scripts.utils.stage2_taxonomy_heavy import (
    LANGS,
    TriInputTaxonomyHeavyClassifier,
    build_image_transform,
    load_checkpoint_partial,
    select_prompt_with_fallback,
    token_count_from_attention_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heavy demo inference over demo_infer originals split.")
    parser.add_argument("--demo_jsonl", type=str, default="data/final/splits/demo_infer_originals.jsonl")
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vision_backbone", type=str, default="vit_b16", choices=["vit_b16", "resnet50"])
    parser.add_argument("--text_model", type=str, default="xlm-roberta-base")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "hi", "bn"])
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--thresholds_json", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="runs/stage2_demo_inference_heavy")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_compare_langs", action="store_true")
    return parser.parse_args()


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_thresholds(path: str) -> List[float]:
    if not path:
        return [0.5] * len(ERROR_TYPES)
    p = Path(path)
    if not p.exists():
        return [0.5] * len(ERROR_TYPES)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [float(payload.get(lbl, 0.5)) for lbl in ERROR_TYPES]
    return [0.5] * len(ERROR_TYPES)


def main() -> None:
    args = parse_args()

    demo_rows = _load_rows(Path(args.demo_jsonl))
    if not demo_rows:
        raise RuntimeError("Demo split is empty or missing.")

    model = TriInputTaxonomyHeavyClassifier(
        vision_backbone=args.vision_backbone,
        text_model_name=args.text_model,
    )
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    ckpt_report = load_checkpoint_partial(model, str(ckpt))
    if args.debug:
        print(json.dumps({"checkpoint_load": ckpt_report}, ensure_ascii=False))

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    transform = build_image_transform(224)
    thresholds = _load_thresholds(args.thresholds_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_jsonl = out_dir / "demo_predictions.jsonl"
    report_csv = out_dir / "demo_report.csv"

    csv_rows: List[Dict[str, Any]] = []
    debug_printed = 0
    compare_printed = 0

    with pred_jsonl.open("w", encoding="utf-8") as f:
        for row in demo_rows:
            sid = str(row.get("id") or "")
            selected = select_prompt_with_fallback(row, args.lang)
            prompt = str(selected["prompt_text"])

            source_path = Path(args.image_root) / str(row.get("source_path") or "")
            target_path = Path(args.image_root) / str(row.get("target_path") or "")
            if not source_path.exists() or not target_path.exists() or not prompt:
                continue

            src = transform(Image.open(source_path).convert("RGB")).unsqueeze(0).to(device)
            tgt = transform(Image.open(target_path).convert("RGB")).unsqueeze(0).to(device)
            toks = tokenizer(
                prompt,
                max_length=args.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = toks["input_ids"].to(device)
            mask = toks["attention_mask"].to(device)

            if args.debug and debug_printed < 3:
                debug_printed += 1
                debug_payload = {
                    "id": sid,
                    "selected_lang": selected["prompt_language_actual"],
                    "requested_lang": selected["prompt_language_requested"],
                    "prompt_preview": prompt[:120],
                    "token_count": token_count_from_attention_mask(mask),
                }
                print(json.dumps({"debug_sample": debug_payload}, ensure_ascii=False))

            if args.debug_compare_langs and compare_printed < 3:
                compare_printed += 1
                toks_by_lang: Dict[str, List[int]] = {}
                meta: Dict[str, Any] = {}
                for lang in LANGS:
                    cand = select_prompt_with_fallback(row, lang)
                    cand_prompt = str(cand["prompt_text"])
                    if not cand_prompt:
                        continue
                    cand_toks = tokenizer(
                        cand_prompt,
                        max_length=args.max_text_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    toks_by_lang[lang] = cand_toks["input_ids"].squeeze(0).tolist()
                    meta[lang] = {
                        "actual_lang": cand["prompt_language_actual"],
                        "fallback_used": bool(cand["fallback_used"]),
                        "token_count": token_count_from_attention_mask(cand_toks["attention_mask"]),
                        "prompt_preview": cand_prompt[:120],
                    }
                keys = list(toks_by_lang.keys())
                pairwise: Dict[str, bool] = {}
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        a, b = keys[i], keys[j]
                        pairwise[f"{a}_vs_{b}_input_ids_differ"] = toks_by_lang[a] != toks_by_lang[b]
                print(json.dumps({"debug_compare_langs": {"id": sid, "meta": meta, "pairwise": pairwise}}, ensure_ascii=False))

            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                    logits = model(src, tgt, input_ids, mask)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().tolist()

            pred_labels = [ERROR_TYPES[i] for i, p in enumerate(probs) if p >= thresholds[i]]
            gold_labels = [str(x) for x in row.get("taxonomy_labels", [])]

            rec = {
                "id": sid,
                "source_path": str(row.get("source_path") or ""),
                "target_path": str(row.get("target_path") or ""),
                "lang": selected["prompt_language_actual"],
                "prompt_language_used": selected["prompt_language_actual"],
                "prompt_language_requested": selected["prompt_language_requested"],
                "prompt_language_actual": selected["prompt_language_actual"],
                "fallback_used": bool(selected["fallback_used"]),
                "prompt": prompt,
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
                "clean_or_no_error": len(pred_labels) == 0,
                "max_pred_confidence": float(max(probs) if probs else 0.0),
                "per_label_probabilities": {ERROR_TYPES[i]: float(probs[i]) for i in range(len(ERROR_TYPES))},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            csv_rows.append(
                {
                    "id": sid,
                    "source_path": rec["source_path"],
                    "target_path": rec["target_path"],
                    "lang": rec["lang"],
                    "prompt_language_used": rec["prompt_language_used"],
                    "prompt_language_requested": rec["prompt_language_requested"],
                    "prompt_language_actual": rec["prompt_language_actual"],
                    "fallback_used": rec["fallback_used"],
                    "gold_labels": "|".join(gold_labels),
                    "pred_labels": "|".join(pred_labels),
                    "max_pred_confidence": rec["max_pred_confidence"],
                }
            )

    with report_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "source_path",
            "target_path",
            "lang",
            "prompt_language_used",
            "prompt_language_requested",
            "prompt_language_actual",
            "fallback_used",
            "gold_labels",
            "pred_labels",
            "max_pred_confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "demo_jsonl": args.demo_jsonl,
        "checkpoint": args.checkpoint,
        "rows_scored": len(csv_rows),
        "predictions_jsonl": str(pred_jsonl),
        "report_csv": str(report_csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
