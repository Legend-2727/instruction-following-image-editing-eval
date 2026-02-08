#!/usr/bin/env python
"""analyze_failures.py — Correlate prompt properties with editing failures.

Usage
-----
    python scripts/analyze_failures.py \\
        --data data/sample \\
        --labels data/annotations/labels.jsonl \\
        --out runs/baseline/analysis

Outputs
-------
    runs/baseline/analysis/
        correlations.csv
        heatmap_correlations.png
        error_type_frequency.png
        wordcount_by_adherence.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_metadata, load_labels, ensure_dirs
from utils.schema import (
    ADHERENCE_LABELS,
    ERROR_TYPES,
    LabelRecord,
)
from utils.prompt_features import extract_prompt_features, PROMPT_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prompt–failure correlations.")
    parser.add_argument("--data", type=str, default="data/sample")
    parser.add_argument("--labels", type=str, default="data/annotations/labels.jsonl")
    parser.add_argument("--out", type=str, default="runs/baseline/analysis")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    ensure_dirs(out_dir)

    # ── Load & merge ─────────────────────────────────────────────────────
    meta_raw = load_metadata(data_dir / "metadata.jsonl")
    labels_raw = load_labels(args.labels)

    if not meta_raw:
        logger.error("No metadata found.")
        sys.exit(1)
    if not labels_raw:
        logger.warning("No labels found — cannot analyze. Exiting gracefully.")
        sys.exit(0)

    meta_by_id = {m["id"]: m for m in meta_raw}
    # Keep latest label per id
    label_by_id = {}
    for r in labels_raw:
        label_by_id[r["id"]] = r
    common_ids = sorted(set(meta_by_id) & set(label_by_id))
    if len(common_ids) < 3:
        logger.warning("Only %d labeled samples — analysis may be trivial.", len(common_ids))

    logger.info("Analyzing %d labeled samples.", len(common_ids))

    # ── Build DataFrame ──────────────────────────────────────────────────
    rows = []
    for sid in common_ids:
        m = meta_by_id[sid]
        lbl = label_by_id[sid]
        lr = LabelRecord.from_dict(lbl)
        pf = extract_prompt_features(m["prompt"])

        row = {"id": sid, "prompt": m["prompt"]}
        row.update(pf)
        row["adherence"] = lr.adherence
        row["adherence_fail"] = int(lr.adherence in ("Partial", "No"))  # binary
        evec = lr.error_vector()
        for i, ename in enumerate(ERROR_TYPES):
            row[ename] = evec[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("DataFrame shape: %s", df.shape)

    # ── Correlation matrix ───────────────────────────────────────────────
    feat_cols = PROMPT_FEATURE_NAMES
    target_cols = ["adherence_fail"] + ERROR_TYPES
    corr_cols = feat_cols + target_cols
    corr_df = df[corr_cols].corr()
    # Restrict to features × targets
    corr_sub = corr_df.loc[feat_cols, target_cols]

    csv_path = out_dir / "correlations.csv"
    corr_sub.to_csv(csv_path)
    logger.info("Saved correlations → %s", csv_path)

    # ── Plots ────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 1. Heatmap
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(
            corr_sub,
            annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.5, ax=ax,
        )
        ax.set_title("Prompt Features × Failure / Error-Type Correlations")
        fig.tight_layout()
        fig.savefig(str(out_dir / "heatmap_correlations.png"), dpi=150)
        plt.close(fig)
        logger.info("Saved heatmap → %s", out_dir / "heatmap_correlations.png")

        # 2. Error-type frequency bar chart
        err_freq = df[ERROR_TYPES].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        err_freq.plot.bar(ax=ax, color="steelblue")
        ax.set_ylabel("Count")
        ax.set_title("Error-Type Frequency")
        fig.tight_layout()
        fig.savefig(str(out_dir / "error_type_frequency.png"), dpi=150)
        plt.close(fig)
        logger.info("Saved error freq → %s", out_dir / "error_type_frequency.png")

        # 3. Word-count by adherence (box plot)
        fig, ax = plt.subplots(figsize=(6, 4))
        adherence_order = [a for a in ADHERENCE_LABELS if a in df["adherence"].values]
        if adherence_order:
            df.boxplot(column="word_count", by="adherence", ax=ax, positions=range(len(adherence_order)))
            ax.set_title("Word Count by Adherence Label")
            ax.set_xlabel("Adherence")
            ax.set_ylabel("word_count")
            fig.suptitle("")  # remove auto-title
        fig.tight_layout()
        fig.savefig(str(out_dir / "wordcount_by_adherence.png"), dpi=150)
        plt.close(fig)
        logger.info("Saved box plot → %s", out_dir / "wordcount_by_adherence.png")

        # 4. Adherence distribution pie chart
        fig, ax = plt.subplots(figsize=(5, 5))
        adh_counts = df["adherence"].value_counts()
        ax.pie(adh_counts.values, labels=adh_counts.index, autopct="%1.1f%%",
               colors=["#2ecc71", "#f39c12", "#e74c3c"])
        ax.set_title("Adherence Distribution")
        fig.tight_layout()
        fig.savefig(str(out_dir / "adherence_distribution.png"), dpi=150)
        plt.close(fig)
        logger.info("Saved pie chart → %s", out_dir / "adherence_distribution.png")

    except Exception as exc:
        logger.warning("Plotting failed: %s", exc)

    # ── Summary to console ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Labeled samples : {len(common_ids)}")
    print(f"Adherence dist  :")
    for a in ADHERENCE_LABELS:
        cnt = (df["adherence"] == a).sum()
        print(f"  {a:10s}  {cnt:4d}  ({100*cnt/len(df):.1f}%)")
    print(f"\nTop correlated feature with failure:")
    top_feat = corr_sub["adherence_fail"].abs().idxmax()
    top_val = corr_sub.loc[top_feat, "adherence_fail"]
    print(f"  {top_feat} → r = {top_val:.3f}")
    print("=" * 60)

    logger.info("Done. All analysis outputs in %s", out_dir)


if __name__ == "__main__":
    main()
