#!/usr/bin/env python3
"""Run two-stage inference: lightweight gate first, taxonomy-only heavy second.

Stage 1 (gate): clean/no-error vs error
Stage 2 (taxonomy): only runs when gate predicts error
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_stage1_multilingual_binary import TriInputBinaryClassifier
from scripts.utils.schema import ERROR_TYPES
from scripts.utils.stage2_taxonomy_heavy import (
    TriInputTaxonomyHeavyClassifier,
    build_image_transform,
    load_checkpoint_partial,
    save_json,
    select_prompt_with_fallback,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage gate+taxonomy inference pipeline.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--out_summary_json", type=str, default="")

    parser.add_argument("--lang", type=str, default="en", choices=["en", "hi", "bn"])

    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage1_text_model", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--stage1_max_text_len", type=int, default=128)
    parser.add_argument("--stage1_clean_threshold", type=float, default=0.5)

    parser.add_argument("--taxonomy_checkpoint", type=str, required=True)
    parser.add_argument("--taxonomy_thresholds_json", type=str, required=True)
    parser.add_argument("--taxonomy_vision_backbone", type=str, default="vit_b16", choices=["vit_b16", "resnet50"])
    parser.add_argument("--taxonomy_text_model", type=str, default="xlm-roberta-base")
    parser.add_argument("--taxonomy_max_text_len", type=int, default=128)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--debug", action="store_true")
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


def _load_taxonomy_thresholds(path: Path) -> List[float]:
    if not path.exists():
        return [0.5] * len(ERROR_TYPES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        out: List[float] = []
        for lbl in ERROR_TYPES:
            v = payload.get(lbl, 0.5)
            out.append(0.5 if v is None else float(v))
        return out
    return [0.5] * len(ERROR_TYPES)


def _load_stage1_state(model: torch.nn.Module, ckpt_path: Path) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state_dict = state["state_dict"]
    elif isinstance(state, dict):
        state_dict = state
    else:
        raise RuntimeError("Unsupported stage-1 checkpoint format")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "loaded": True,
        "path": str(ckpt_path),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "loaded_key_count": len(state_dict),
    }


def main() -> None:
    args = parse_args()

    rows = _load_rows(Path(args.input_jsonl))
    if not rows:
        raise RuntimeError("Input JSONL is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_image_transform(224)

    # Stage-1 gate model
    stage1_tokenizer = AutoTokenizer.from_pretrained(args.stage1_text_model)
    stage1_model = TriInputBinaryClassifier(
        text_model_name=args.stage1_text_model,
        freeze_vision=True,
        freeze_text=False,
    )
    stage1_load = _load_stage1_state(stage1_model, Path(args.stage1_checkpoint))
    stage1_model.to(device)
    stage1_model.eval()

    # Stage-2 taxonomy model
    tax_tokenizer = AutoTokenizer.from_pretrained(args.taxonomy_text_model)
    tax_model = TriInputTaxonomyHeavyClassifier(
        vision_backbone=args.taxonomy_vision_backbone,
        text_model_name=args.taxonomy_text_model,
    )
    taxonomy_load = load_checkpoint_partial(tax_model, args.taxonomy_checkpoint)
    tax_model.to(device)
    tax_model.eval()

    taxonomy_thresholds = _load_taxonomy_thresholds(Path(args.taxonomy_thresholds_json))

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    gate_clean = 0
    gate_error = 0
    fallback_count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            total += 1
            sid = str(row.get("id") or "")

            selected = select_prompt_with_fallback(row, args.lang)
            prompt = str(selected["prompt_text"])
            if not prompt:
                continue
            fallback_count += int(bool(selected["fallback_used"]))

            source_path = Path(args.image_root) / str(row.get("source_path") or "")
            target_path = Path(args.image_root) / str(row.get("target_path") or "")
            if not source_path.exists() or not target_path.exists():
                continue

            src = transform(Image.open(source_path).convert("RGB")).unsqueeze(0).to(device)
            tgt = transform(Image.open(target_path).convert("RGB")).unsqueeze(0).to(device)

            # Stage-1 gate forward
            gate_toks = stage1_tokenizer(
                prompt,
                max_length=args.stage1_max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            gate_input_ids = gate_toks["input_ids"].to(device)
            gate_mask = gate_toks["attention_mask"].to(device)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                    gate_logit = stage1_model(src, tgt, gate_input_ids, gate_mask)
                    gate_prob_clean = float(torch.sigmoid(gate_logit).squeeze().item())

            gate_predict_clean = bool(gate_prob_clean >= float(args.stage1_clean_threshold))

            pred_labels: List[str] = []
            tax_probs: Optional[Dict[str, float]] = None
            if gate_predict_clean:
                gate_clean += 1
            else:
                gate_error += 1
                tax_toks = tax_tokenizer(
                    prompt,
                    max_length=args.taxonomy_max_text_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                tax_input_ids = tax_toks["input_ids"].to(device)
                tax_mask = tax_toks["attention_mask"].to(device)

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                        tax_logits = tax_model(src, tgt, tax_input_ids, tax_mask)
                        probs = torch.sigmoid(tax_logits).squeeze(0).cpu().numpy().tolist()

                pred_labels = [ERROR_TYPES[i] for i, p in enumerate(probs) if p >= taxonomy_thresholds[i]]
                tax_probs = {ERROR_TYPES[i]: float(probs[i]) for i in range(len(ERROR_TYPES))}

            rec = {
                "id": sid,
                "source_path": str(row.get("source_path") or ""),
                "target_path": str(row.get("target_path") or ""),
                "lang": selected["prompt_language_actual"],
                "prompt_language_requested": selected["prompt_language_requested"],
                "prompt_language_actual": selected["prompt_language_actual"],
                "fallback_used": bool(selected["fallback_used"]),
                "prompt": prompt,
                "stage1_gate_prob_clean": gate_prob_clean,
                "stage1_gate_threshold": float(args.stage1_clean_threshold),
                "stage1_gate_predict_clean": gate_predict_clean,
                "predicted_taxonomy_labels": pred_labels,
                "per_label_probabilities": tax_probs,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": args.input_jsonl,
        "out_jsonl": str(out_path),
        "total_rows": total,
        "gate_predicted_clean": gate_clean,
        "gate_predicted_error": gate_error,
        "fallback_used_count": fallback_count,
        "stage1_load": stage1_load,
        "taxonomy_load": taxonomy_load,
    }

    summary_path = Path(args.out_summary_json) if args.out_summary_json else (out_path.with_suffix(".summary.json"))
    save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
