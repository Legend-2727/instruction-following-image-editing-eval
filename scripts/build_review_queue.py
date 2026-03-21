#!/usr/bin/env python3
"""Build a prioritized human-review queue from dual-judge outputs.

This stage bridges:
  run_dual_judge.py  ->  human_review_tool.py  ->  merge_review_log.py

The output queue preserves the raw judge outputs, adds a deterministic
machine-provisional label bundle, optional original labels from a base
classifier dataset, and a paper-auditable review priority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.io import load_jsonl, save_jsonl
from utils.review import (
    choose_provisional_labels,
    compute_review_priority,
    extract_existing_labels,
    load_latest_review_actions,
    sort_review_queue,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prioritized review queue from dual-judge outputs."
    )
    parser.add_argument(
        "--judged_dataset",
        required=True,
        help="JSONL output from run_dual_judge.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL review queue",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Optional JSON summary path",
    )
    parser.add_argument(
        "--base_dataset",
        default="",
        help=(
            "Optional base dataset with original labels, e.g. "
            "artifacts/english_classifier/train.jsonl"
        ),
    )
    parser.add_argument(
        "--review_log",
        default="",
        help="Optional existing review log; reviewed ids are excluded by default",
    )
    parser.add_argument(
        "--include_reviewed",
        action="store_true",
        help="Keep samples even if they already appear in the review log",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on output records after sorting (0 = no cap)",
    )
    parser.add_argument(
        "--low_confidence",
        type=float,
        default=0.7,
        help="High-priority threshold for mean confidence",
    )
    parser.add_argument(
        "--medium_confidence",
        type=float,
        default=0.8,
        help="Medium-priority threshold for mean confidence",
    )
    return parser.parse_args()



def _index_by_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in records:
        sample_id = str(row.get("id", "")).strip()
        if sample_id:
            indexed[sample_id] = row
    return indexed



def _build_queue_record(
    judged_row: Dict[str, Any],
    base_row: Optional[Dict[str, Any]],
    low_confidence: float,
    medium_confidence: float,
) -> Dict[str, Any]:
    record = dict(judged_row)

    existing_labels = extract_existing_labels(base_row or {})
    provisional = choose_provisional_labels(judged_row)
    if provisional is None:
        raise ValueError(f"Row {judged_row.get('id')} is missing judge predictions")

    if existing_labels:
        record["original_adherence_label"] = existing_labels["adherence"]
        record["original_taxonomy_labels"] = existing_labels["taxonomy"]
        record["original_label_source"] = existing_labels["source"]
    else:
        record["original_adherence_label"] = None
        record["original_taxonomy_labels"] = []
        record["original_label_source"] = None

    record["previous_adherence_label"] = provisional["adherence"]
    record["previous_taxonomy_labels"] = provisional["taxonomy"]
    record["previous_label_source"] = provisional["source"]
    record["previous_label_confidence"] = provisional["selected_confidence"]

    record.update(
        compute_review_priority(
            judged_row,
            low_confidence=low_confidence,
            medium_confidence=medium_confidence,
        )
    )
    return record



def compute_summary(
    queue_records: List[Dict[str, Any]],
    total_judged: int,
    already_reviewed_count: int,
) -> Dict[str, Any]:
    priority_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    for row in queue_records:
        priority = str(row.get("review_priority", "unknown"))
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

        source = str(row.get("previous_label_source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1

    return {
        "total_judged_records": total_judged,
        "already_reviewed_count": already_reviewed_count,
        "queued_count": len(queue_records),
        "priority_distribution": priority_counts,
        "previous_label_source_distribution": source_counts,
    }



def main() -> None:
    args = parse_args()

    print(f"[1] Loading judged dataset from {args.judged_dataset}...")
    judged_records = load_jsonl(args.judged_dataset)
    print(f"    Loaded {len(judged_records)} judged records")

    base_by_id: Dict[str, Dict[str, Any]] = {}
    if args.base_dataset:
        print(f"[2] Loading optional base dataset from {args.base_dataset}...")
        base_by_id = _index_by_id(load_jsonl(args.base_dataset))
        print(f"    Indexed {len(base_by_id)} base records")
    else:
        print("[2] No base dataset provided; queue will omit original labels")

    latest_reviews: Dict[str, Dict[str, Any]] = {}
    if args.review_log:
        print(f"[3] Loading existing review log from {args.review_log}...")
        latest_reviews = load_latest_review_actions(args.review_log)
        print(f"    Found {len(latest_reviews)} already-reviewed sample ids")
    else:
        print("[3] No review log provided")

    queue_records: List[Dict[str, Any]] = []
    skipped_reviewed = 0
    for row in judged_records:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            continue
        if (not args.include_reviewed) and sample_id in latest_reviews:
            skipped_reviewed += 1
            continue
        queue_records.append(
            _build_queue_record(
                row,
                base_by_id.get(sample_id),
                low_confidence=args.low_confidence,
                medium_confidence=args.medium_confidence,
            )
        )

    queue_records = sort_review_queue(queue_records)
    if args.limit > 0:
        queue_records = queue_records[: args.limit]

    print(f"[4] Writing review queue to {args.output}...")
    save_jsonl(queue_records, args.output)
    print(f"    Wrote {len(queue_records)} queue records")

    summary = compute_summary(
        queue_records,
        total_judged=len(judged_records),
        already_reviewed_count=skipped_reviewed,
    )
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[5] Summary saved to {args.summary}")

    print("[OK] Review queue build complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
