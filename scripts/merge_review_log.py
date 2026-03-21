#!/usr/bin/env python3
"""Merge an append-only human review log with a dataset or review queue.

This script is intentionally schema-aware and conservative:
- raw judge outputs are preserved untouched
- baseline/original labels are surfaced separately from machine-provisional labels
- final labels are only filled from human review, auto-accept, or pre-existing
  baseline labels depending on the input record type and CLI flags
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.io import load_jsonl, save_jsonl
from utils.review import (
    choose_provisional_labels,
    extract_existing_labels,
    load_latest_review_actions,
)
from utils.schema import ReviewActionRecord, normalize_error_types, validate_adherence_label



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge append-only review log with judged/base dataset."
    )
    parser.add_argument(
        "--base_dataset",
        required=True,
        help="Input JSONL dataset, review queue, or judged dataset",
    )
    parser.add_argument(
        "--review_log",
        required=True,
        help="Append-only human review log JSONL",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Merged output JSONL",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Optional summary JSON path",
    )
    parser.add_argument(
        "--accept_auto_candidates",
        action="store_true",
        help="Promote auto_accept_candidate rows to final labels when no human review exists",
    )
    return parser.parse_args()



def _extract_queue_provisional(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    adherence = row.get("previous_adherence_label")
    if not adherence:
        return None
    try:
        return {
            "adherence": validate_adherence_label(adherence),
            "taxonomy": normalize_error_types(row.get("previous_taxonomy_labels", [])),
            "source": str(row.get("previous_label_source") or "review_queue"),
        }
    except ValueError:
        return None



def _safe_review_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        return ReviewActionRecord.from_dict(raw).to_dict()
    except ValueError as exc:
        sample_id = raw.get("sample_id") or raw.get("id") or "<unknown>"
        print(f"[WARN] Skipping invalid review record for {sample_id}: {exc}")
        return None



def _merge_one(
    row: Dict[str, Any],
    latest_review: Optional[Dict[str, Any]],
    accept_auto_candidates: bool,
) -> Dict[str, Any]:
    merged = dict(row)

    baseline = extract_existing_labels(row)
    provisional = _extract_queue_provisional(row) or choose_provisional_labels(row)
    if provisional is None and baseline is not None:
        provisional = dict(baseline)

    merged["baseline_adherence"] = baseline["adherence"] if baseline else None
    merged["baseline_taxonomy"] = baseline["taxonomy"] if baseline else []
    merged["baseline_label_source"] = baseline["source"] if baseline else None

    merged["provisional_adherence"] = provisional["adherence"] if provisional else None
    merged["provisional_taxonomy"] = provisional["taxonomy"] if provisional else []
    merged["provisional_label_source"] = provisional["source"] if provisional else None

    final_adherence = None
    final_taxonomy: List[str] = []
    final_source = None
    review_status = "pending"
    reviewer_id = None
    reviewed_at = None
    review_action_type = None
    review_notes = ""

    if latest_review is not None:
        final_adherence = latest_review["updated_labels"]["adherence"]
        final_taxonomy = latest_review["updated_labels"]["taxonomy"]
        final_source = "human_review"
        review_action_type = latest_review["action_type"]
        review_status = review_action_type
        reviewer_id = latest_review.get("reviewer_id")
        reviewed_at = latest_review.get("timestamp_utc")
        review_notes = latest_review.get("notes", "")
    elif accept_auto_candidates and bool(row.get("auto_accept_candidate")) and provisional is not None:
        final_adherence = provisional["adherence"]
        final_taxonomy = provisional["taxonomy"]
        final_source = f"auto_accept:{provisional['source']}"
        review_status = "auto_accepted"
        review_action_type = "auto_accepted"
    elif baseline is not None:
        final_adherence = baseline["adherence"]
        final_taxonomy = baseline["taxonomy"]
        final_source = baseline["source"]
        review_status = "unchanged"
        review_action_type = "unchanged"

    merged["final_adherence"] = final_adherence
    merged["final_taxonomy"] = final_taxonomy
    merged["final_label_source"] = final_source
    merged["review_status"] = review_status
    merged["review_action_type"] = review_action_type
    merged["reviewer_id"] = reviewer_id
    merged["reviewed_at"] = reviewed_at
    merged["review_notes"] = review_notes
    return merged



def compute_summary(merged_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    final_source_counts: Dict[str, int] = {}
    for row in merged_records:
        status = str(row.get("review_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

        source = str(row.get("final_label_source", "None"))
        final_source_counts[source] = final_source_counts.get(source, 0) + 1

    return {
        "total_records": len(merged_records),
        "review_status_distribution": status_counts,
        "final_label_source_distribution": final_source_counts,
        "records_with_final_labels": sum(
            1 for row in merged_records if row.get("final_adherence")
        ),
        "records_pending_final_labels": sum(
            1 for row in merged_records if not row.get("final_adherence")
        ),
    }



def main() -> None:
    args = parse_args()

    print(f"[1] Loading base dataset from {args.base_dataset}...")
    base_rows = load_jsonl(args.base_dataset)
    print(f"    Loaded {len(base_rows)} rows")

    print(f"[2] Loading latest review actions from {args.review_log}...")
    raw_latest_reviews = load_latest_review_actions(args.review_log)
    latest_reviews: Dict[str, Dict[str, Any]] = {}
    invalid_review_count = 0
    for sample_id, raw in raw_latest_reviews.items():
        safe = _safe_review_record(raw)
        if safe is None:
            invalid_review_count += 1
            continue
        latest_reviews[sample_id] = safe
    print(f"    Valid latest reviews: {len(latest_reviews)}")
    if invalid_review_count:
        print(f"    Invalid latest reviews skipped: {invalid_review_count}")

    print("[3] Merging records...")
    merged = []
    for row in base_rows:
        sample_id = str(row.get("id") or row.get("sample_id") or "").strip()
        latest = latest_reviews.get(sample_id)
        merged.append(
            _merge_one(
                row,
                latest_review=latest,
                accept_auto_candidates=args.accept_auto_candidates,
            )
        )
    print(f"    Merged {len(merged)} rows")

    print(f"[4] Writing merged dataset to {args.output}...")
    save_jsonl(merged, args.output)

    summary = compute_summary(merged)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[5] Summary saved to {args.summary}")

    print("[OK] Merge complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
