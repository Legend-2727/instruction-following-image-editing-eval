#!/usr/bin/env python3
"""Validate the cleaned xLingual multilingual taxonomy dataset.

Checks performed:
- exact row count
- required fields present
- taxonomy labels are canonical
- all three language prompts are present and non-empty
- POSIX relative paths only
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils.io import load_jsonl, normalize_relpath
from scripts.utils.schema import ERROR_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate xLingual dataset rows.")
    parser.add_argument(
        "--jsonl",
        default="data/final/final_taxonomy_originals.jsonl",
        help="Input dataset JSONL file.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=6000,
        help="Expected number of rows.",
    )
    parser.add_argument(
        "--require-image-files",
        action="store_true",
        help="Also require referenced image files to exist under --image-root.",
    )
    parser.add_argument(
        "--image-root",
        default="",
        help="Local image root used with --require-image-files.",
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional path to write a JSON summary.",
    )
    return parser.parse_args()


def _missing_fields(row: Dict[str, Any], fields: Iterable[str]) -> List[str]:
    missing = []
    for field in fields:
        if field not in row:
            missing.append(field)
            continue
        value = row.get(field)
        if field == "taxonomy_labels":
            if value is None:
                missing.append(field)
        elif not str(value).strip():
            missing.append(field)
    return missing


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.jsonl)

    required_fields = [
        "id",
        "source_path",
        "target_path",
        "instruction_en",
        "instruction_hi",
        "instruction_bn",
        "taxonomy_labels",
    ]

    field_missing_counts = Counter()
    taxonomy_counts = Counter()
    source_type_counts = Counter()
    bad_paths = 0
    missing_images = 0
    invalid_taxonomy = Counter()
    rows_with_labels = 0
    rows_without_labels = 0
    prompt_languages = {"instruction_en": 0, "instruction_hi": 0, "instruction_bn": 0}

    image_root = Path(args.image_root) if args.image_root else None
    for row in rows:
        for field in _missing_fields(row, required_fields):
            field_missing_counts[field] += 1

        if row.get("taxonomy_labels"):
            rows_with_labels += 1
            for label in row.get("taxonomy_labels", []):
                taxonomy_counts[str(label)] += 1
        else:
            rows_without_labels += 1

        for key in prompt_languages:
            if str(row.get(key, "")).strip():
                prompt_languages[key] += 1

        source_type_counts[str(row.get("source_type") or "unknown")] += 1

        source_path = str(row.get("source_path") or "")
        target_path = str(row.get("target_path") or "")
        if source_path != normalize_relpath(source_path) or target_path != normalize_relpath(target_path):
            bad_paths += 1

        if args.require_image_files:
            if image_root is None:
                raise ValueError("--image-root is required with --require-image-files")
            if not (image_root / normalize_relpath(source_path)).exists():
                missing_images += 1
            if not (image_root / normalize_relpath(target_path)).exists():
                missing_images += 1

        for label in row.get("taxonomy_labels", []):
            if label not in ERROR_TYPES:
                invalid_taxonomy[str(label)] += 1

    summary = {
        "jsonl": str(Path(args.jsonl)),
        "expected_count": args.expected_count,
        "actual_count": len(rows),
        "rows_with_labels": rows_with_labels,
        "rows_without_labels": rows_without_labels,
        "prompt_language_counts": prompt_languages,
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "taxonomy_label_counts": dict((label, taxonomy_counts.get(label, 0)) for label in ERROR_TYPES),
        "missing_field_counts": dict(sorted(field_missing_counts.items())),
        "invalid_taxonomy_counts": dict(sorted(invalid_taxonomy.items())),
        "bad_path_rows": bad_paths,
        "missing_image_checks": missing_images,
        "passed": True,
    }

    errors = []
    if len(rows) != args.expected_count:
        errors.append(f"expected {args.expected_count} rows, found {len(rows)}")
    if any(field_missing_counts.get(field, 0) for field in required_fields):
        errors.append("one or more required fields are missing")
    if any(count != len(rows) for count in prompt_languages.values()):
        errors.append("one or more language prompt columns are missing or empty")
    if bad_paths:
        errors.append("one or more rows contain non-POSIX relative paths")
    if invalid_taxonomy:
        errors.append("invalid taxonomy labels found")
    if args.require_image_files and missing_images:
        errors.append("referenced image files are missing")

    if errors:
        summary["passed"] = False

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
