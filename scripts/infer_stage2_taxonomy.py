#!/usr/bin/env python3
"""Single-sample inference for Stage-2 multilingual taxonomy model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils.schema import ERROR_TYPES
from scripts.utils.stage2_taxonomy import (
    LANGS,
    TriInputTaxonomyClassifier,
    build_image_transform,
    select_prompt_with_fallback,
    token_count_from_attention_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer taxonomy labels for one source/edited/prompt triple.")
    parser.add_argument("--source_image", type=str, required=True)
    parser.add_argument("--edited_image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "hi", "bn"])
    parser.add_argument("--prompt_en", type=str, default="")
    parser.add_argument("--prompt_hi", type=str, default="")
    parser.add_argument("--prompt_bn", type=str, default="")
    parser.add_argument("--debug", action="store_true", help="Print selected prompt/debug tokenization info.")
    parser.add_argument("--debug_compare_langs", action="store_true", help="Compare tokenizer input IDs for en/hi/bn prompts.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text_model", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument(
        "--thresholds_json",
        type=str,
        default="",
        help="Optional JSON file with per-class thresholds (e.g. best_thresholds.json).",
    )
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


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

    source_path = Path(args.source_image)
    edited_path = Path(args.edited_image)
    if not source_path.exists() or not edited_path.exists():
        raise FileNotFoundError("source_image or edited_image does not exist")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    model = TriInputTaxonomyClassifier(text_model_name=args.text_model)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    thresholds = _load_thresholds(args.thresholds_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = build_image_transform()
    src_img = transform(Image.open(source_path).convert("RGB")).unsqueeze(0).to(device)
    tgt_img = transform(Image.open(edited_path).convert("RGB")).unsqueeze(0).to(device)

    row_prompt_pack: Dict[str, Any] = {
        "instruction_en": args.prompt_en,
        "instruction_hi": args.prompt_hi,
        "instruction_bn": args.prompt_bn,
    }
    if not any(str(row_prompt_pack[k]).strip() for k in row_prompt_pack):
        if not str(args.prompt).strip():
            raise ValueError("Provide --prompt or at least one of --prompt_en/--prompt_hi/--prompt_bn.")
        row_prompt_pack[f"instruction_{args.lang}"] = args.prompt

    selected = select_prompt_with_fallback(row_prompt_pack, args.lang)
    prompt_text = str(selected["prompt_text"])
    if not prompt_text:
        raise ValueError("No non-empty prompt found for requested language or fallback languages.")

    toks = tokenizer(
        prompt_text,
        max_length=args.max_text_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].to(device)
    mask = toks["attention_mask"].to(device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
            logits = model(src_img, tgt_img, input_ids, mask)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().tolist()

    pred_labels = [ERROR_TYPES[i] for i, p in enumerate(probs) if p >= thresholds[i]]
    is_clean = len(pred_labels) == 0

    if args.debug:
        debug_payload = {
            "selected_lang": selected["prompt_language_actual"],
            "requested_lang": selected["prompt_language_requested"],
            "fallback_used": bool(selected["fallback_used"]),
            "prompt_preview": prompt_text[:120],
            "token_count": token_count_from_attention_mask(mask),
        }
        print(json.dumps({"debug": debug_payload}, ensure_ascii=False))

    if args.debug_compare_langs:
        toks_by_lang: Dict[str, List[int]] = {}
        compare_info: Dict[str, Any] = {}
        for lang in LANGS:
            cand = select_prompt_with_fallback(row_prompt_pack, lang)
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
            compare_info[lang] = {
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
        print(json.dumps({"debug_compare_langs": {"meta": compare_info, "pairwise": pairwise}}, ensure_ascii=False))

    output: Dict[str, Any] = {
        "lang": selected["prompt_language_actual"],
        "prompt_language_used": selected["prompt_language_actual"],
        "prompt_language_requested": selected["prompt_language_requested"],
        "prompt_language_actual": selected["prompt_language_actual"],
        "fallback_used": bool(selected["fallback_used"]),
        "prompt": prompt_text,
        "source_image": str(source_path),
        "edited_image": str(edited_path),
        "predicted_taxonomy_labels": pred_labels,
        "per_label_probabilities": {ERROR_TYPES[i]: float(probs[i]) for i in range(len(ERROR_TYPES))},
        "per_label_thresholds": {ERROR_TYPES[i]: float(thresholds[i]) for i in range(len(ERROR_TYPES))},
        "clean_or_no_error": bool(is_clean),
        "clean_label": "clean/no-error" if is_clean else "has-error",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
