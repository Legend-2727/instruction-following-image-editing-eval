#!/usr/bin/env python3
"""Evaluate lightweight baseline vs two-stage pipeline on the same expanded test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_stage1_multilingual_binary import TriInputBinaryClassifier
from scripts.utils.schema import ERROR_TYPES
from scripts.utils.stage2_taxonomy import TriInputTaxonomyClassifier, build_image_transform as build_light_transform
from scripts.utils.stage2_taxonomy_heavy import (
    TriInputTaxonomyHeavyClassifier,
    binarize_with_thresholds,
    compute_multilabel_metrics,
    compute_per_class_metrics,
    expand_rows_by_language,
    language_debug_summary,
    load_checkpoint_partial,
    load_original_rows,
    metrics_by_language,
    prediction_diagnostics,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full two-stage system vs lightweight baseline.")
    parser.add_argument("--test_jsonl", type=str, default="data/final/splits/test_originals.jsonl")
    parser.add_argument("--image_root", type=str, required=True)

    parser.add_argument("--stage1_checkpoint", type=str, default="runs/stage1_binary_multilingual/best_model.pt")
    parser.add_argument("--stage1_text_model", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--stage1_clean_threshold", type=float, default=0.5)

    parser.add_argument("--lightweight_taxonomy_checkpoint", type=str, default="")
    parser.add_argument("--lightweight_taxonomy_thresholds_json", type=str, default="")
    parser.add_argument("--lightweight_text_model", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--lightweight_max_text_len", type=int, default=128)

    parser.add_argument("--taxonomy_only_heavy_checkpoint", type=str, default="checkpoints/stage2_taxonomy_only_heavy/best_model.pt")
    parser.add_argument("--taxonomy_only_heavy_thresholds_json", type=str, default="checkpoints/stage2_taxonomy_only_heavy/best_thresholds.json")
    parser.add_argument("--taxonomy_vision_backbone", type=str, default="vit_b16", choices=["vit_b16", "resnet50"])
    parser.add_argument("--taxonomy_text_model", type=str, default="xlm-roberta-base")
    parser.add_argument("--taxonomy_max_text_len", type=int, default=128)

    parser.add_argument("--lang", type=str, default="all", choices=["en", "hi", "bn", "all"])
    parser.add_argument("--out_dir", type=str, default="runs/final_comparison")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def _resolve_lightweight_paths(args: argparse.Namespace) -> Tuple[Path, Path, List[str]]:
    notes: List[str] = []
    ckpt = Path(args.lightweight_taxonomy_checkpoint) if args.lightweight_taxonomy_checkpoint else None
    thr = Path(args.lightweight_taxonomy_thresholds_json) if args.lightweight_taxonomy_thresholds_json else None

    ckpt_candidates = [
        Path("checkpoints/stage2_taxonomy/best_model.pt"),
        Path("runs/stage2_taxonomy/best_model.pt"),
        Path("runs/baseline/best_model.pt"),
    ]
    thr_candidates = [
        Path("checkpoints/stage2_taxonomy/best_thresholds.json"),
        Path("runs/stage2_taxonomy/best_thresholds.json"),
    ]

    if ckpt is None:
        existing = [p for p in ckpt_candidates if p.exists()]
        if not existing:
            raise FileNotFoundError("Could not auto-detect lightweight taxonomy checkpoint; pass --lightweight_taxonomy_checkpoint.")
        ckpt = existing[0]
        if len(existing) > 1:
            notes.append(f"Multiple lightweight checkpoint candidates found, using first: {ckpt}")
        else:
            notes.append(f"Auto-detected lightweight checkpoint: {ckpt}")

    if thr is None:
        existing_thr = [p for p in thr_candidates if p.exists()]
        if not existing_thr:
            raise FileNotFoundError("Could not auto-detect lightweight thresholds JSON; pass --lightweight_taxonomy_thresholds_json.")
        thr = existing_thr[0]
        if len(existing_thr) > 1:
            notes.append(f"Multiple lightweight threshold candidates found, using first: {thr}")
        else:
            notes.append(f"Auto-detected lightweight thresholds: {thr}")

    if not ckpt.exists():
        raise FileNotFoundError(f"Lightweight taxonomy checkpoint not found: {ckpt}")
    if not thr.exists():
        raise FileNotFoundError(f"Lightweight taxonomy thresholds not found: {thr}")

    return ckpt, thr, notes


def _load_thresholds(path: Path) -> List[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        out: List[float] = []
        for lbl in ERROR_TYPES:
            v = payload.get(lbl, 0.5)
            out.append(0.5 if v is None else float(v))
        return out
    return [0.5] * len(ERROR_TYPES)


def _labels_to_multihot(labels: Sequence[str]) -> List[int]:
    s = set(str(x) for x in labels)
    return [1 if lbl in s else 0 for lbl in ERROR_TYPES]


def _load_stage1_state(model: torch.nn.Module, ckpt_path: Path) -> Dict[str, Any]:
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


def _evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    langs: List[str],
) -> Dict[str, Any]:
    return {
        "overall": compute_multilabel_metrics(y_true, y_prob, y_pred),
        "by_lang": metrics_by_language(y_true, y_prob, y_pred, langs),
        "by_lang_debug": language_debug_summary(y_true, langs),
        "per_class": compute_per_class_metrics(y_true, y_prob, y_pred),
        "prediction_diagnostics": prediction_diagnostics(y_prob, y_pred),
    }


def _winner(a: float, b: float, a_name: str = "lightweight", b_name: str = "two_stage") -> str:
    if a > b:
        return a_name
    if b > a:
        return b_name
    return "tie"


def _build_summary_md(
    comparison: Dict[str, Any],
    output_paths: Dict[str, str],
    routing: Dict[str, Any],
) -> str:
    lw = comparison["lightweight"]["overall"]
    ts = comparison["two_stage"]["overall"]

    lines: List[str] = []
    lines.append("# Final Comparison: Lightweight vs Two-Stage")
    lines.append("")
    lines.append("## Overall Winners")
    lines.append(f"- micro_f1: **{_winner(lw['micro_f1'], ts['micro_f1'])}**")
    lines.append(f"- macro_f1_supported: **{_winner(lw['macro_f1_supported'], ts['macro_f1_supported'])}**")
    lines.append(f"- mAP_macro_supported: **{_winner(lw['mAP_macro_supported'], ts['mAP_macro_supported'])}**")
    lines.append("")

    lines.append("## Per-Language Comparison")
    for lang in ["en", "hi", "bn"]:
        lw_l = comparison["lightweight"]["by_lang"].get(lang, {})
        ts_l = comparison["two_stage"]["by_lang"].get(lang, {})
        lines.append(
            f"- {lang}: micro_f1 winner={_winner(float(lw_l.get('micro_f1', 0.0)), float(ts_l.get('micro_f1', 0.0)))}; "
            f"macro_f1_supported winner={_winner(float(lw_l.get('macro_f1_supported', 0.0)), float(ts_l.get('macro_f1_supported', 0.0)))}; "
            f"mAP_macro_supported winner={_winner(float(lw_l.get('mAP_macro_supported', 0.0)), float(ts_l.get('mAP_macro_supported', 0.0)))}"
        )
    lines.append("")

    lines.append("## Per-Class Winners (F1)")
    for lbl in ERROR_TYPES:
        lw_f1 = float(comparison["lightweight"]["per_class"].get(lbl, {}).get("f1", 0.0))
        ts_f1 = float(comparison["two_stage"]["per_class"].get(lbl, {}).get("f1", 0.0))
        lines.append(f"- {lbl}: {_winner(lw_f1, ts_f1)}")
    lines.append("")

    lines.append("## Two-Stage Routing Stats")
    lines.append(f"- total_rows: {routing['total_rows']}")
    lines.append(f"- gate_predicted_clean: {routing['gate_predicted_clean']}")
    lines.append(f"- gate_predicted_error_routed_to_heavy: {routing['gate_predicted_error']}")
    lines.append("")

    lines.append("## Output Files")
    lines.append(f"- metrics: {output_paths['metrics_json']}")
    lines.append(f"- lightweight predictions: {output_paths['lightweight_jsonl']}")
    lines.append(f"- two-stage predictions: {output_paths['two_stage_jsonl']}")
    lines.append(f"- summary: {output_paths['summary_md']}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lightweight_ckpt, lightweight_thr_path, resolve_notes = _resolve_lightweight_paths(args)
    lightweight_thresholds = _load_thresholds(lightweight_thr_path)
    heavy_thresholds = _load_thresholds(Path(args.taxonomy_only_heavy_thresholds_json))

    test_originals = load_original_rows(args.test_jsonl)
    expanded_rows = expand_rows_by_language(test_originals)
    if args.lang != "all":
        expanded_rows = [r for r in expanded_rows if str(r.get("lang")) == args.lang]

    if not expanded_rows:
        raise RuntimeError("No expanded test rows found for selected --lang.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    # Models/tokenizers
    light_tokenizer = AutoTokenizer.from_pretrained(args.lightweight_text_model)
    stage1_tokenizer = AutoTokenizer.from_pretrained(args.stage1_text_model)
    heavy_tokenizer = AutoTokenizer.from_pretrained(args.taxonomy_text_model)

    light_model = TriInputTaxonomyClassifier(text_model_name=args.lightweight_text_model)
    light_load = load_checkpoint_partial(light_model, str(lightweight_ckpt))
    light_model.to(device)
    light_model.eval()

    stage1_model = TriInputBinaryClassifier(
        text_model_name=args.stage1_text_model,
        freeze_vision=True,
        freeze_text=False,
    )
    stage1_load = _load_stage1_state(stage1_model, Path(args.stage1_checkpoint))
    stage1_model.to(device)
    stage1_model.eval()

    heavy_model = TriInputTaxonomyHeavyClassifier(
        vision_backbone=args.taxonomy_vision_backbone,
        text_model_name=args.taxonomy_text_model,
    )
    heavy_load = load_checkpoint_partial(heavy_model, str(Path(args.taxonomy_only_heavy_checkpoint)))
    heavy_model.to(device)
    heavy_model.eval()

    transform = build_light_transform()

    y_true: List[List[int]] = []
    langs: List[str] = []

    lw_prob_rows: List[List[float]] = []
    lw_pred_rows: List[List[int]] = []

    ts_prob_rows: List[List[float]] = []
    ts_pred_rows: List[List[int]] = []

    lightweight_pred_path = out_dir / "lightweight_predictions_test.jsonl"
    two_stage_pred_path = out_dir / "two_stage_predictions_test.jsonl"

    gate_clean = 0
    gate_error = 0

    with lightweight_pred_path.open("w", encoding="utf-8") as f_lw, two_stage_pred_path.open("w", encoding="utf-8") as f_ts:
        for row in expanded_rows:
            sid = str(row.get("id") or "")
            lang = str(row.get("lang") or "")
            instr = str(row.get("instruction") or "")
            src_path = Path(args.image_root) / str(row.get("source_path") or "")
            tgt_path = Path(args.image_root) / str(row.get("target_path") or "")

            if not src_path.exists() or not tgt_path.exists() or not instr:
                continue

            gold_labels = [str(x) for x in row.get("taxonomy_labels", [])]
            gold_vec = _labels_to_multihot(gold_labels)

            src = transform(Image.open(src_path).convert("RGB")).unsqueeze(0).to(device)
            tgt = transform(Image.open(tgt_path).convert("RGB")).unsqueeze(0).to(device)

            # Lightweight baseline
            l_toks = light_tokenizer(
                instr,
                max_length=args.lightweight_max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            l_input_ids = l_toks["input_ids"].to(device)
            l_mask = l_toks["attention_mask"].to(device)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    l_logits = light_model(src, tgt, l_input_ids, l_mask)
                    l_probs = torch.sigmoid(l_logits).squeeze(0).cpu().numpy().tolist()
            l_pred = [1 if l_probs[i] >= lightweight_thresholds[i] else 0 for i in range(len(ERROR_TYPES))]
            l_pred_labels = [ERROR_TYPES[i] for i, v in enumerate(l_pred) if v == 1]

            # Two-stage
            g_toks = stage1_tokenizer(
                instr,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            g_input_ids = g_toks["input_ids"].to(device)
            g_mask = g_toks["attention_mask"].to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    g_logit = stage1_model(src, tgt, g_input_ids, g_mask)
                    g_prob_clean = float(torch.sigmoid(g_logit).squeeze().item())
            gate_predict_clean = bool(g_prob_clean >= float(args.stage1_clean_threshold))

            if gate_predict_clean:
                gate_clean += 1
                t_probs = [0.0] * len(ERROR_TYPES)
            else:
                gate_error += 1
                t_toks = heavy_tokenizer(
                    instr,
                    max_length=args.taxonomy_max_text_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                t_input_ids = t_toks["input_ids"].to(device)
                t_mask = t_toks["attention_mask"].to(device)
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        t_logits = heavy_model(src, tgt, t_input_ids, t_mask)
                        t_probs = torch.sigmoid(t_logits).squeeze(0).cpu().numpy().tolist()

            t_pred = [1 if t_probs[i] >= heavy_thresholds[i] else 0 for i in range(len(ERROR_TYPES))]
            t_pred_labels = [ERROR_TYPES[i] for i, v in enumerate(t_pred) if v == 1]

            y_true.append(gold_vec)
            langs.append(lang)
            lw_prob_rows.append([float(x) for x in l_probs])
            lw_pred_rows.append(l_pred)
            ts_prob_rows.append([float(x) for x in t_probs])
            ts_pred_rows.append(t_pred)

            rec_lw = {
                "id": sid,
                "lang": lang,
                "source_path": str(row.get("source_path") or ""),
                "target_path": str(row.get("target_path") or ""),
                "instruction": instr,
                "gold_labels": gold_labels,
                "pred_labels": l_pred_labels,
                "per_label_probabilities": {ERROR_TYPES[i]: float(l_probs[i]) for i in range(len(ERROR_TYPES))},
                "thresholds": {ERROR_TYPES[i]: float(lightweight_thresholds[i]) for i in range(len(ERROR_TYPES))},
            }
            f_lw.write(json.dumps(rec_lw, ensure_ascii=False) + "\n")

            rec_ts = {
                "id": sid,
                "lang": lang,
                "source_path": str(row.get("source_path") or ""),
                "target_path": str(row.get("target_path") or ""),
                "instruction": instr,
                "gold_labels": gold_labels,
                "stage1_gate_prob_clean": g_prob_clean,
                "stage1_gate_threshold": float(args.stage1_clean_threshold),
                "stage1_gate_predict_clean": gate_predict_clean,
                "pred_labels": t_pred_labels,
                "per_label_probabilities": {ERROR_TYPES[i]: float(t_probs[i]) for i in range(len(ERROR_TYPES))},
                "thresholds": {ERROR_TYPES[i]: float(heavy_thresholds[i]) for i in range(len(ERROR_TYPES))},
            }
            f_ts.write(json.dumps(rec_ts, ensure_ascii=False) + "\n")

    y_true_arr = np.array(y_true, dtype=np.int32)
    lw_prob_arr = np.array(lw_prob_rows, dtype=np.float32)
    lw_pred_arr = np.array(lw_pred_rows, dtype=np.int32)
    ts_prob_arr = np.array(ts_prob_rows, dtype=np.float32)
    ts_pred_arr = np.array(ts_pred_rows, dtype=np.int32)

    lw_metrics = _evaluate_predictions(y_true_arr, lw_prob_arr, lw_pred_arr, langs)
    ts_metrics = _evaluate_predictions(y_true_arr, ts_prob_arr, ts_pred_arr, langs)

    routing = {
        "total_rows": int(len(langs)),
        "gate_predicted_clean": int(gate_clean),
        "gate_predicted_error": int(gate_error),
    }

    comparison_payload: Dict[str, Any] = {
        "config": {
            "test_jsonl": args.test_jsonl,
            "image_root": args.image_root,
            "lang": args.lang,
            "stage1_clean_threshold": float(args.stage1_clean_threshold),
            "lightweight_thresholds_json": str(lightweight_thr_path),
            "taxonomy_only_heavy_thresholds_json": args.taxonomy_only_heavy_thresholds_json,
        },
        "artifact_resolution_notes": resolve_notes,
        "model_load": {
            "stage1": stage1_load,
            "lightweight_taxonomy": light_load,
            "taxonomy_only_heavy": heavy_load,
            "lightweight_checkpoint_used": str(lightweight_ckpt),
            "lightweight_thresholds_used": str(lightweight_thr_path),
        },
        "routing": routing,
        "lightweight": lw_metrics,
        "two_stage": ts_metrics,
    }

    metrics_path = out_dir / "test_comparison_metrics.json"
    save_json(metrics_path, comparison_payload)

    summary_md = _build_summary_md(
        comparison={"lightweight": lw_metrics, "two_stage": ts_metrics},
        output_paths={
            "metrics_json": str(metrics_path),
            "lightweight_jsonl": str(lightweight_pred_path),
            "two_stage_jsonl": str(two_stage_pred_path),
            "summary_md": str(out_dir / "comparison_summary.md"),
        },
        routing=routing,
    )
    summary_path = out_dir / "comparison_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    print(json.dumps({
        "metrics_json": str(metrics_path),
        "lightweight_predictions": str(lightweight_pred_path),
        "two_stage_predictions": str(two_stage_pred_path),
        "summary_md": str(summary_path),
        "lightweight_checkpoint_used": str(lightweight_ckpt),
        "lightweight_thresholds_used": str(lightweight_thr_path),
        "routing": routing,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
