#!/usr/bin/env python
"""analyze_vlm_results.py — Aggregate and visualize VLM judgments.

Produces:
  - adherence_distribution.png   (pie chart)
  - error_type_frequency.png     (bar chart)
  - heatmap_correlations.png     (prompt features vs failure)
  - confidence_distribution.png  (histogram of VLM confidence)
  - failure_examples.png         (gallery of worst edits)
  - summary.json                 (key statistics)
  - correlations.csv

Usage
-----
    python scripts/analyze_vlm_results.py --data data/eval --out runs/eval_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_jsonl, ensure_dirs
from utils.schema import ADHERENCE_LABELS, ERROR_TYPES, VLMJudgment
from utils.prompt_features import extract_prompt_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data(data_dir: Path):
    """Load metadata + judgments into a merged DataFrame."""
    meta = load_jsonl(data_dir / "metadata.jsonl")
    judgments = load_jsonl(data_dir / "vlm_judgments.jsonl")

    meta_df = pd.DataFrame(meta)
    judg_df = pd.DataFrame(judgments)

    if judg_df.empty:
        logger.error("No VLM judgments found!")
        sys.exit(1)

    df = meta_df.merge(judg_df, on="id", how="inner", suffixes=("_meta", "_vlm"))
    logger.info("Merged %d samples (metadata=%d, judgments=%d)", len(df), len(meta), len(judgments))
    return df


def add_prompt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add prompt-level features as columns."""
    prompt_col = "prompt_meta" if "prompt_meta" in df.columns else "prompt"
    feats = df[prompt_col].apply(extract_prompt_features).apply(pd.Series)
    return pd.concat([df, feats], axis=1)


def plot_adherence_pie(df: pd.DataFrame, out_dir: Path):
    """Pie chart of adherence distribution."""
    adh_col = "adherence_vlm" if "adherence_vlm" in df.columns else "adherence"
    counts = df[adh_col].value_counts()
    colors = {"Success": "#4CAF50", "Partial": "#FFC107", "No": "#F44336"}
    c = [colors.get(l, "#999") for l in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=c, startangle=90, textprops={"fontsize": 13}
    )
    ax.set_title(
        f"InstructPix2Pix Instruction Adherence\n(n={len(df)} samples, judged by Qwen2.5-VL)",
        fontsize=14, pad=20,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "adherence_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved adherence_distribution.png")


def plot_error_frequency(df: pd.DataFrame, out_dir: Path):
    """Bar chart of error type frequencies."""
    adh_col = "adherence_vlm" if "adherence_vlm" in df.columns else "adherence"
    error_col = "error_types_vlm" if "error_types_vlm" in df.columns else "error_types"

    failed = df[df[adh_col].isin(["Partial", "No"])]
    error_counts = {}
    for row_errors in failed[error_col]:
        if isinstance(row_errors, list):
            for e in row_errors:
                error_counts[e] = error_counts.get(e, 0) + 1

    if not error_counts:
        logger.warning("No errors found — skipping error frequency plot")
        return

    errors_df = pd.DataFrame(
        sorted(error_counts.items(), key=lambda x: x[1], reverse=True),
        columns=["Error Type", "Count"],
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=errors_df, x="Count", y="Error Type", ax=ax, palette="Reds_r")
    ax.set_title(
        f"Error Type Distribution (n={len(failed)} failed edits)", fontsize=14
    )
    ax.set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "error_type_frequency.png", dpi=150)
    plt.close(fig)
    logger.info("Saved error_type_frequency.png")


def plot_confidence_dist(df: pd.DataFrame, out_dir: Path):
    """Histogram of VLM confidence scores."""
    conf_col = "confidence" if "confidence" in df.columns else None
    if conf_col is None or df[conf_col].isna().all():
        logger.warning("No confidence data — skipping histogram")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[conf_col].dropna(), bins=20, color="#2196F3", edgecolor="white")
    ax.set_xlabel("VLM Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of VLM Judge Confidence", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved confidence_distribution.png")


def plot_correlations(df: pd.DataFrame, out_dir: Path):
    """Heatmap of prompt features vs failure."""
    adh_col = "adherence_vlm" if "adherence_vlm" in df.columns else "adherence"
    df["failed"] = (df[adh_col] != "Success").astype(int)

    feature_cols = [
        "word_count", "char_count", "has_spatial_words",
        "count_of_colors", "num_changes_proxy", "specificity_proxy",
    ]
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        logger.warning("No prompt features — skipping correlation heatmap")
        return

    corr_cols = available + ["failed"]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, ax=ax)
    ax.set_title("Prompt Features vs Edit Failure Correlation", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_correlations.png", dpi=150)
    plt.close(fig)

    corr.to_csv(out_dir / "correlations.csv")
    logger.info("Saved heatmap_correlations.png + correlations.csv")


def plot_failure_gallery(df: pd.DataFrame, data_dir: Path, out_dir: Path, k: int = 8):
    """Grid of worst failure examples (lowest confidence or No adherence)."""
    adh_col = "adherence_vlm" if "adherence_vlm" in df.columns else "adherence"
    failed = df[df[adh_col] == "No"].copy()
    if failed.empty:
        failed = df[df[adh_col] == "Partial"].copy()
    if failed.empty:
        logger.warning("No failures found — skipping gallery")
        return

    # Sort by confidence ascending (least confident = most problematic)
    if "confidence" in failed.columns:
        failed = failed.sort_values("confidence", ascending=True)

    samples = failed.head(k)
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (_, row) in enumerate(samples.iterrows()):
        orig_path = data_dir / row["orig_path"]
        edit_path = data_dir / row["edited_path"]
        gt_path = data_dir / row.get("gt_path", "")

        # Original
        if orig_path.exists():
            axes[i, 0].imshow(Image.open(orig_path))
        axes[i, 0].set_title("Original", fontsize=10)
        axes[i, 0].axis("off")

        # Model edited
        if edit_path.exists():
            axes[i, 1].imshow(Image.open(edit_path))
        axes[i, 1].set_title("Model Edit (IP2P)", fontsize=10)
        axes[i, 1].axis("off")

        # Ground truth (if available)
        if gt_path and Path(data_dir / gt_path).exists():
            axes[i, 2].imshow(Image.open(data_dir / gt_path))
            axes[i, 2].set_title("Ground Truth", fontsize=10)
        else:
            axes[i, 2].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
            axes[i, 2].set_title("Ground Truth", fontsize=10)
        axes[i, 2].axis("off")

        # Annotate
        prompt_col = "prompt_meta" if "prompt_meta" in row.index else "prompt"
        prompt_text = str(row.get(prompt_col, ""))[:80]
        reasoning = str(row.get("reasoning", ""))[:100]
        error_col = "error_types_vlm" if "error_types_vlm" in row.index else "error_types"
        errors = row.get(error_col, [])
        if isinstance(errors, list):
            errors = ", ".join(errors)

        caption = f"ID: {row['id']} | Prompt: {prompt_text}\nErrors: {errors}\nReason: {reasoning}"
        fig.text(
            0.5, 1.0 - (i * (1.0 / n)) - 0.01,
            caption, ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#FFF3E0", alpha=0.8),
        )

    fig.suptitle(
        f"Worst Failure Examples (n={n})", fontsize=16, y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_dir / "failure_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved failure_examples.png")


def write_summary(df: pd.DataFrame, out_dir: Path):
    """Write summary statistics to JSON."""
    adh_col = "adherence_vlm" if "adherence_vlm" in df.columns else "adherence"
    error_col = "error_types_vlm" if "error_types_vlm" in df.columns else "error_types"

    total = len(df)
    adh_counts = df[adh_col].value_counts().to_dict()
    failure_rate = 1.0 - (adh_counts.get("Success", 0) / total) if total else 0

    # Count errors
    error_counts = {}
    for row_errors in df[error_col]:
        if isinstance(row_errors, list):
            for e in row_errors:
                error_counts[e] = error_counts.get(e, 0) + 1

    avg_confidence = float(df["confidence"].mean()) if "confidence" in df.columns else None

    summary = {
        "total_samples": total,
        "adherence_distribution": adh_counts,
        "failure_rate": round(failure_rate, 4),
        "error_type_counts": error_counts,
        "top_3_errors": sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3],
        "avg_vlm_confidence": round(avg_confidence, 4) if avg_confidence else None,
        "model_editor": df.get("editor_model", pd.Series(["unknown"])).iloc[0]
            if "editor_model" in df.columns else "unknown",
        "model_judge": df.get("model_name", pd.Series(["unknown"])).iloc[0]
            if "model_name" in df.columns else "unknown",
    }

    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved summary.json")

    # Print key findings
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total samples evaluated: {total}")
    print(f"  Failure rate: {failure_rate:.1%}")
    print(f"  Adherence: {adh_counts}")
    if avg_confidence:
        print(f"  Avg VLM confidence: {avg_confidence:.3f}")
    if error_counts:
        print(f"  Top errors: {summary['top_3_errors']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VLM judgments on image edits"
    )
    parser.add_argument("--data", type=str, default="data/eval")
    parser.add_argument("--out", type=str, default="runs/eval_analysis")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    ensure_dirs(out_dir)

    df = load_data(data_dir)
    df = add_prompt_features(df)

    plot_adherence_pie(df, out_dir)
    plot_error_frequency(df, out_dir)
    plot_confidence_dist(df, out_dir)
    plot_correlations(df, out_dir)
    plot_failure_gallery(df, data_dir, out_dir)
    write_summary(df, out_dir)

    logger.info("All analysis outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
