#!/usr/bin/env python3
"""Split final taxonomy originals into train/val/test/demo_infer.

Splitting is done strictly at original id level (no language expansion here).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils.io import ensure_dirs, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build original-level splits for taxonomy training.")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="data/final/final_taxonomy_originals.jsonl",
        help="Final originals JSONL path.",
    )
    parser.add_argument("--out_dir", type=str, default="data/final/splits", help="Output splits directory.")
    parser.add_argument("--demo_size", type=int, default=30, help="Demo split size in original samples.")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Validation ratio (of non-demo set).")
    parser.add_argument("--test_ratio", type=float, default=0.10, help="Test ratio (of non-demo set).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _row_strata_key(row: Dict[str, Any], major_labels: Sequence[str]) -> str:
    has_error = int(row.get("has_error", 0))
    edit_type = str(row.get("edit_type") or "unknown")
    labels = set(str(x) for x in row.get("taxonomy_labels", []) if str(x).strip())
    majors_present = [m for m in major_labels if m in labels]
    major_tag = "|".join(majors_present[:2]) if majors_present else "none"
    return f"he={has_error}__et={edit_type}__maj={major_tag}"


def _safe_train_test_split(
    rows: List[Dict[str, Any]],
    test_size: float | int,
    seed: int,
    major_labels: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        return [], []
    if isinstance(test_size, int):
        if test_size <= 0:
            return list(rows), []
        if test_size >= len(rows):
            return [], list(rows)
    else:
        if test_size <= 0:
            return list(rows), []
        if test_size >= 1.0:
            return [], list(rows)

    strata = [_row_strata_key(r, major_labels) for r in rows]
    counts = Counter(strata)

    # train_test_split with stratify requires at least 2 members per class.
    if any(v < 2 for v in counts.values()):
        stratify = None
    else:
        stratify = strata

    train_rows, test_rows = train_test_split(
        rows,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return list(train_rows), list(test_rows)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_summary(name: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    edit_counter: Counter[str] = Counter()
    has_error_counter: Counter[int] = Counter()
    label_counter: Counter[str] = Counter()
    for row in rows:
        edit_counter[str(row.get("edit_type") or "unknown")] += 1
        has_error_counter[int(row.get("has_error", 0))] += 1
        for lbl in row.get("taxonomy_labels", []):
            label_counter[str(lbl)] += 1
    return {
        "split": name,
        "size": len(rows),
        "has_error_counts": {str(k): int(v) for k, v in sorted(has_error_counter.items())},
        "edit_type_counts": dict(sorted(edit_counter.items())),
        "taxonomy_label_counts": dict(sorted(label_counter.items())),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_rows = load_jsonl(args.input_jsonl)
    rows = [r for r in input_rows if str(r.get("id") or "").strip()]

    # Deterministic ordering before split.
    rows.sort(key=lambda x: str(x["id"]))

    # Estimate major labels by frequency from this finalized dataset.
    global_label_counts: Counter[str] = Counter()
    for r in rows:
        for lbl in r.get("taxonomy_labels", []):
            global_label_counts[str(lbl)] += 1
    major_labels = [x[0] for x in global_label_counts.most_common(4)]

    demo_size = max(0, min(args.demo_size, len(rows)))
    rest_rows, demo_rows = _safe_train_test_split(
        rows,
        test_size=demo_size,
        seed=args.seed,
        major_labels=major_labels,
    )

    # Split remaining data into train/val/test with val/test each ratio of non-demo pool.
    total_holdout_ratio = max(0.0, min(0.95, args.val_ratio + args.test_ratio))
    train_rows, holdout_rows = _safe_train_test_split(
        rest_rows,
        test_size=total_holdout_ratio,
        seed=args.seed,
        major_labels=major_labels,
    )

    if holdout_rows and total_holdout_ratio > 0:
        # Make test share in holdout proportional to requested ratio.
        test_share = args.test_ratio / (args.val_ratio + args.test_ratio)
        val_rows, test_rows = _safe_train_test_split(
            holdout_rows,
            test_size=test_share,
            seed=args.seed,
            major_labels=major_labels,
        )
    else:
        val_rows, test_rows = [], []

    # Sort each split by id for reproducibility.
    for split_rows in (train_rows, val_rows, test_rows, demo_rows):
        split_rows.sort(key=lambda x: str(x["id"]))

    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    train_path = out_dir / "train_originals.jsonl"
    val_path = out_dir / "val_originals.jsonl"
    test_path = out_dir / "test_originals.jsonl"
    demo_path = out_dir / "demo_infer_originals.jsonl"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    _write_jsonl(test_path, test_rows)
    _write_jsonl(demo_path, demo_rows)

    ids_train = {r["id"] for r in train_rows}
    ids_val = {r["id"] for r in val_rows}
    ids_test = {r["id"] for r in test_rows}
    ids_demo = {r["id"] for r in demo_rows}

    overlap = {
        "train_val": len(ids_train & ids_val),
        "train_test": len(ids_train & ids_test),
        "train_demo": len(ids_train & ids_demo),
        "val_test": len(ids_val & ids_test),
        "val_demo": len(ids_val & ids_demo),
        "test_demo": len(ids_test & ids_demo),
    }

    summary = {
        "input_jsonl": args.input_jsonl,
        "out_dir": str(out_dir),
        "seed": args.seed,
        "major_labels_for_stratification": major_labels,
        "requested": {
            "demo_size": args.demo_size,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "actual": {
            "train": _split_summary("train", train_rows),
            "val": _split_summary("val", val_rows),
            "test": _split_summary("test", test_rows),
            "demo_infer": _split_summary("demo_infer", demo_rows),
        },
        "id_overlap_counts": overlap,
    }

    summary_path = out_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
