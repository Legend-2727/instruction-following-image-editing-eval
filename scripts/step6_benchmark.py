#!/usr/bin/env python
"""step6_benchmark.py — Benchmark classifier on all languages & compute metrics.

Evaluates the trained classifier on English + low-resource languages (Nepali,
Bangla, Hindi) to measure cross-lingual performance gap.

Also computes CLIP-I similarity and DINO scores for generated edits.

Usage
-----
    python scripts/step6_benchmark.py --data data/magicbrush --model_dir runs/classifier
    nohup python scripts/step6_benchmark.py --data data/magicbrush --model_dir runs/classifier > benchmark.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_jsonl, save_jsonl, ensure_dirs
from utils.schema import ERROR_TYPES, NUM_ERROR_TYPES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


BENCHMARK_LANGS = ["en", "ne", "bn", "hi"]
LANG_NAMES = {"en": "English", "ne": "Nepali", "bn": "Bangla", "hi": "Hindi"}


def compute_clip_similarity(data_dir: Path, records: list[dict], device: str = "cuda") -> dict:
    """Compute CLIP-I similarity between source and target images."""
    import open_clip

    logger.info("Computing CLIP-I similarity...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()

    similarities = []
    for rec in tqdm(records, desc="CLIP-I"):
        src_path = data_dir / rec["source_path"]
        tgt_path = data_dir / rec["target_path"]

        if not src_path.exists() or not tgt_path.exists():
            continue

        src_img = preprocess(Image.open(src_path).convert("RGB")).unsqueeze(0).to(device)
        tgt_img = preprocess(Image.open(tgt_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            src_feat = model.encode_image(src_img)
            tgt_feat = model.encode_image(tgt_img)
            src_feat = src_feat / src_feat.norm(dim=-1, keepdim=True)
            tgt_feat = tgt_feat / tgt_feat.norm(dim=-1, keepdim=True)
            sim = (src_feat * tgt_feat).sum().item()
            similarities.append(sim)

    del model
    torch.cuda.empty_cache()

    return {
        "clip_i_mean": float(np.mean(similarities)) if similarities else 0.0,
        "clip_i_std": float(np.std(similarities)) if similarities else 0.0,
        "clip_i_median": float(np.median(similarities)) if similarities else 0.0,
        "n_samples": len(similarities),
    }


def compute_dino_similarity(data_dir: Path, records: list[dict], device: str = "cuda") -> dict:
    """Compute DINO structure-preserving similarity."""
    import timm
    from torchvision import transforms

    logger.info("Computing DINO similarity...")
    model = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    similarities = []
    for rec in tqdm(records, desc="DINO"):
        src_path = data_dir / rec["source_path"]
        tgt_path = data_dir / rec["target_path"]

        if not src_path.exists() or not tgt_path.exists():
            continue

        src_img = transform(Image.open(src_path).convert("RGB")).unsqueeze(0).to(device)
        tgt_img = transform(Image.open(tgt_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            src_feat = model.forward_features(src_img)[:, 0, :]  # CLS token
            tgt_feat = model.forward_features(tgt_img)[:, 0, :]
            src_feat = src_feat / src_feat.norm(dim=-1, keepdim=True)
            tgt_feat = tgt_feat / tgt_feat.norm(dim=-1, keepdim=True)
            sim = (src_feat * tgt_feat).sum().item()
            similarities.append(sim)

    del model
    torch.cuda.empty_cache()

    return {
        "dino_mean": float(np.mean(similarities)) if similarities else 0.0,
        "dino_std": float(np.std(similarities)) if similarities else 0.0,
        "dino_median": float(np.median(similarities)) if similarities else 0.0,
        "n_samples": len(similarities),
    }


def run_classifier_benchmark(
    model_dir: Path,
    data_dir: Path,
    test_records: list[dict],
    device: str = "cuda",
    batch_size: int = 8,
) -> dict:
    """Run trained classifier on test set for all languages."""
    from step5_train_classifier import (
        MultilingualEditErrorClassifier,
        EditErrorDataset,
        evaluate,
        compute_class_weights,
    )
    import open_clip
    from transformers import AutoTokenizer

    # Load config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    clip_model_name = config.get("clip_model", "ViT-B-32")
    clip_pretrained = config.get("clip_pretrained", "laion2b_s34b_b79k")
    xlmr_model = config.get("xlmr_model", "xlm-roberta-base")

    # Load CLIP preprocessor and XLM-R tokenizer
    _, clip_preprocess, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
    tokenizer = AutoTokenizer.from_pretrained(xlmr_model)

    # Build model
    model = MultilingualEditErrorClassifier(
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
        xlmr_model_name=xlmr_model,
        num_classes=NUM_ERROR_TYPES,
        hidden_dim=config.get("hidden_dim", 512),
        dropout=config.get("dropout", 0.3),
        freeze_vision=True,
        freeze_text=True,
    ).to(device)

    # Load best weights
    ckpt_path = model_dir / "best_model.pt" if (model_dir / "best_model.pt").exists() else model_dir / "last_model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Class weights
    class_weights = compute_class_weights(test_records)

    results = {}
    for lang in BENCHMARK_LANGS:
        lang_key = f"instruction_{lang}" if lang != "en" else "instruction_en"

        # Check if translations exist
        has_translations = all(lang_key in r for r in test_records)
        if not has_translations and lang != "en":
            logger.warning("No %s translations found — skipping", LANG_NAMES[lang])
            continue

        logger.info("Evaluating on %s (%s)...", LANG_NAMES[lang], lang)

        dataset = EditErrorDataset(
            test_records, data_dir, clip_preprocess,
            tokenizer, lang=lang,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        metrics = evaluate(model, loader, device, class_weights)
        results[lang] = metrics
        logger.info(
            "  %s — macro_f1=%.4f weighted_f1=%.4f exact_match=%.4f",
            LANG_NAMES[lang], metrics["macro_f1"], metrics["weighted_f1"], metrics["exact_match"]
        )

    # Compute cross-lingual gap
    if "en" in results:
        en_f1 = results["en"]["weighted_f1"]
        for lang in ["ne", "bn", "hi"]:
            if lang in results:
                gap = en_f1 - results[lang]["weighted_f1"]
                results[lang]["cross_lingual_gap"] = round(gap, 4)
                logger.info(
                    "  Cross-lingual gap (en → %s): %.4f",
                    LANG_NAMES[lang], gap
                )

    del model
    torch.cuda.empty_cache()
    return results


def generate_report(all_results: dict, out_dir: Path):
    """Generate a comprehensive benchmark report."""
    ensure_dirs(out_dir)

    # Save full results
    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate markdown report
    lines = [
        "# Multilingual Image Editing Error Classifier — Benchmark Report\n",
        f"## Summary\n",
    ]

    # Classifier results table
    if "classifier" in all_results:
        clf_results = all_results["classifier"]
        lines.append("### Cross-Lingual Error Classification\n")
        lines.append("| Language | Macro-F1 | Weighted-F1 | Exact Match | Hamming Loss | XL Gap |")
        lines.append("|----------|----------|-------------|-------------|--------------|--------|")
        for lang in BENCHMARK_LANGS:
            if lang in clf_results:
                m = clf_results[lang]
                gap = m.get("cross_lingual_gap", "—")
                if isinstance(gap, float):
                    gap = f"{gap:.4f}"
                lines.append(
                    f"| {LANG_NAMES[lang]} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | "
                    f"{m['exact_match']:.4f} | {m['hamming_loss']:.4f} | {gap} |"
                )
        lines.append("")

        # Per-class F1 table
        lines.append("### Per-Class F1 Scores\n")
        header = "| Error Type |"
        separator = "|------------|"
        for lang in BENCHMARK_LANGS:
            if lang in clf_results:
                header += f" {LANG_NAMES[lang]} |"
                separator += "---------|"
        lines.append(header)
        lines.append(separator)

        for etype in ERROR_TYPES:
            row = f"| {etype} |"
            for lang in BENCHMARK_LANGS:
                if lang in clf_results:
                    f1 = clf_results[lang]["per_class_f1"].get(etype, 0.0)
                    row += f" {f1:.4f} |"
            lines.append(row)
        lines.append("")

    # Image quality metrics
    if "clip_i" in all_results:
        lines.append("### Image Quality Metrics\n")
        clip = all_results["clip_i"]
        lines.append(f"- **CLIP-I Similarity**: {clip['clip_i_mean']:.4f} ± {clip['clip_i_std']:.4f}")

    if "dino" in all_results:
        dino = all_results["dino"]
        lines.append(f"- **DINO Score**: {dino['dino_mean']:.4f} ± {dino['dino_std']:.4f}")
    lines.append("")

    # Error rate analysis
    if "error_analysis" in all_results:
        ea = all_results["error_analysis"]
        lines.append("### Error Rate Analysis\n")
        lines.append(f"- **Overall error rate**: {ea.get('error_rate', 0):.2%}")
        lines.append(f"- **Samples with errors**: {ea.get('n_with_errors', 0)}/{ea.get('total', 0)}")
        if "top_errors" in ea:
            lines.append("\n**Most common errors:**")
            for err, count in ea["top_errors"]:
                lines.append(f"  - {err}: {count}")
        lines.append("")

    report = "\n".join(lines)
    report_path = out_dir / "BENCHMARK_REPORT.md"
    report_path.write_text(report)
    logger.info("Saved benchmark report → %s", report_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark multilingual classifier")
    parser.add_argument("--data", type=str, default="data/magicbrush")
    parser.add_argument("--model_dir", type=str, default="runs/classifier")
    parser.add_argument("--out", type=str, default="runs/benchmark")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_clip", action="store_true")
    parser.add_argument("--skip_dino", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    ensure_dirs(out_dir)

    # Load test set
    test_path = model_dir / "test_set.jsonl"
    if not test_path.exists():
        logger.error("No test set found at %s. Run step5_train_classifier.py first.", test_path)
        sys.exit(1)

    test_records = load_jsonl(test_path)
    logger.info("Loaded %d test records", len(test_records))

    all_results = {}

    # 1. Classifier benchmark across languages
    logger.info("=" * 60)
    logger.info("CLASSIFIER BENCHMARK")
    logger.info("=" * 60)
    classifier_results = run_classifier_benchmark(
        model_dir=model_dir,
        data_dir=data_dir,
        test_records=test_records,
        device=args.device,
        batch_size=args.batch_size,
    )
    all_results["classifier"] = classifier_results

    # 2. CLIP-I similarity
    if not args.skip_clip:
        logger.info("=" * 60)
        logger.info("CLIP-I SIMILARITY")
        logger.info("=" * 60)
        clip_results = compute_clip_similarity(data_dir, test_records, device=args.device)
        all_results["clip_i"] = clip_results
        logger.info("CLIP-I: %.4f ± %.4f", clip_results["clip_i_mean"], clip_results["clip_i_std"])

    # 3. DINO similarity
    if not args.skip_dino:
        logger.info("=" * 60)
        logger.info("DINO SIMILARITY")
        logger.info("=" * 60)
        dino_results = compute_dino_similarity(data_dir, test_records, device=args.device)
        all_results["dino"] = dino_results
        logger.info("DINO: %.4f ± %.4f", dino_results["dino_mean"], dino_results["dino_std"])

    # 4. Error rate analysis
    error_counts = {}
    n_with_errors = 0
    for rec in test_records:
        vec = rec.get("error_label_vector", [0] * 11)
        if any(v == 1 for v in vec):
            n_with_errors += 1
        for i, v in enumerate(vec):
            if v == 1:
                etype = ERROR_TYPES[i]
                error_counts[etype] = error_counts.get(etype, 0) + 1

    all_results["error_analysis"] = {
        "total": len(test_records),
        "n_with_errors": n_with_errors,
        "error_rate": n_with_errors / max(len(test_records), 1),
        "error_counts": error_counts,
        "top_errors": sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5],
    }

    # Generate report
    report = generate_report(all_results, out_dir)
    print(report)

    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE — outputs in %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
