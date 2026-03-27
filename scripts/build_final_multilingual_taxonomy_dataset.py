#!/usr/bin/env python3
"""Build the final original-sample taxonomy dataset for Stage-2 training.

Rules implemented:
- source_type=sft -> taxonomy_labels=[] and taxonomy_is_verified=true
- source_type=preference_rejected + annotation -> taxonomy_labels=annotation_errors
- unannotated rejected rows are excluded by default
- multilingual text fields are normalized to instruction_en/instruction_hi/instruction_bn
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils.io import ensure_dirs, load_metadata, normalize_relpath
from scripts.utils.schema import ERROR_TYPES, normalize_error_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final multilingual taxonomy dataset.")
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/hf_snapshots/xlingual_picobanana_multilingual_6k/metadata.jsonl",
        help="Base multilingual metadata JSONL path.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Verified rejected-annotation file (JSONL or JSON list/object).",
    )
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default="data/final/final_taxonomy_originals.jsonl",
        help="Output final originals JSONL.",
    )
    parser.add_argument(
        "--out_summary",
        type=str,
        default="data/final/final_taxonomy_summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--include_unannotated_rejected",
        action="store_true",
        help="If set, keep unannotated rejected rows as unverified negatives.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit from metadata for smoke checks (0 means all).",
    )
    return parser.parse_args()


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    # JSON list/dict support
    if raw[0] in "[{":
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            if isinstance(parsed, dict):
                # If object has a records-like field, use that; else treat object as one record.
                for key in ("records", "data", "items"):
                    val = parsed.get(key)
                    if isinstance(val, list):
                        return [x for x in val if isinstance(x, dict)]
                return [parsed]
        except json.JSONDecodeError:
            # This commonly happens when the file is JSONL and starts with '{'.
            pass

    # JSONL fallback
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _norm_key(text: str) -> str:
    return "".join(ch for ch in text.lower().strip() if ch.isalnum())


def _build_taxonomy_aliases() -> Dict[str, str]:
    aliases = { _norm_key(lbl): lbl for lbl in ERROR_TYPES }

    # Common schema drift seen in reviewed annotation files.
    aliases.update(
        {
            _norm_key("Under Editing"): "Under-editing",
            _norm_key("Over Editing"): "Over-editing",
            _norm_key("StyleMismatch"): "Style Mismatch",
            _norm_key("Artifact/Quality Issue"): "Artifact / Quality Issue",
            _norm_key("Artifact Quality Issue"): "Artifact / Quality Issue",
            _norm_key("FailedRemoval"): "Failed Removal",
            _norm_key("WrongObject"): "Wrong Object",
            _norm_key("MissingObject"): "Missing Object",
            _norm_key("ExtraObject"): "Extra Object",
            _norm_key("WrongAttribute"): "Wrong Attribute",
            _norm_key("SpatialError"): "Spatial Error",
            _norm_key("AmbiguousPrompt"): "Ambiguous Prompt",
            _norm_key("UnderEditing"): "Under-editing",
            _norm_key("OverEditing"): "Over-editing",
        }
    )
    return aliases


def _normalize_annotation_errors(raw_errors: Iterable[Any], aliases: Dict[str, str]) -> List[str]:
    mapped: List[str] = []
    unknown: List[str] = []

    for item in raw_errors or []:
        txt = str(item).strip()
        if not txt:
            continue
        key = _norm_key(txt)
        canonical = aliases.get(key)
        if canonical is None:
            unknown.append(txt)
            continue
        mapped.append(canonical)

    if unknown:
        raise ValueError(
            "Unknown taxonomy labels in annotations: "
            + ", ".join(sorted(set(unknown)))
            + ". Please map them before training."
        )

    return normalize_error_types(mapped)


def _pick_instruction(base_row: Dict[str, Any], ann_row: Dict[str, Any], lang: str) -> str:
    if lang == "en":
        return str(ann_row.get("instruction_en") or base_row.get("instruction_en") or "").strip()
    if lang == "hi":
        return str(
            ann_row.get("translation_en_to_hi")
            or ann_row.get("instruction_hi")
            or base_row.get("instruction_hi")
            or ""
        ).strip()
    if lang == "bn":
        return str(
            ann_row.get("translation_en_to_bn")
            or ann_row.get("instruction_bn")
            or base_row.get("instruction_bn")
            or ""
        ).strip()
    return ""


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    metadata_path = Path(args.metadata)
    annotations_path = Path(args.annotations)
    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)

    metadata = load_metadata(metadata_path)
    if args.limit > 0:
        metadata = metadata[: args.limit]

    annotation_rows = _load_json_or_jsonl(annotations_path)
    ann_by_id: Dict[str, Dict[str, Any]] = {}
    for row in annotation_rows:
        sample_id = str(row.get("id") or "").strip()
        if sample_id:
            ann_by_id[sample_id] = row

    aliases = _build_taxonomy_aliases()

    final_rows: List[Dict[str, Any]] = []
    excluded_unannotated_rejected = 0
    total_sft = 0
    total_annotated_rejected = 0

    for row in metadata:
        sample_id = str(row.get("id") or "").strip()
        if not sample_id:
            continue

        source_type = str(row.get("source_type") or "").strip()
        ann = ann_by_id.get(sample_id, {})

        if source_type == "sft":
            taxonomy_labels: List[str] = []
            taxonomy_is_verified = True
            total_sft += 1
        elif source_type == "preference_rejected":
            if ann:
                taxonomy_labels = _normalize_annotation_errors(ann.get("annotation_errors", []), aliases)
                taxonomy_is_verified = True
                total_annotated_rejected += 1
            elif args.include_unannotated_rejected:
                taxonomy_labels = []
                taxonomy_is_verified = False
            else:
                excluded_unannotated_rejected += 1
                continue
        else:
            # Keep training set strict and reproducible.
            continue

        out_row: Dict[str, Any] = {
            "id": sample_id,
            "source_type": source_type,
            "edit_type": str(row.get("edit_type") or "").strip(),
            "source_path": normalize_relpath(str(row.get("source_path") or "")),
            "target_path": normalize_relpath(str(row.get("target_path") or "")),
            "instruction_en": _pick_instruction(row, ann, "en"),
            "instruction_hi": _pick_instruction(row, ann, "hi"),
            "instruction_bn": _pick_instruction(row, ann, "bn"),
            "taxonomy_labels": taxonomy_labels,
            "taxonomy_is_verified": bool(taxonomy_is_verified),
            "annotation_model": ann.get("annotation_model") if ann else None,
            "has_error": int(len(taxonomy_labels) > 0),
            # Kept for audit/debug only; training scripts must ignore them.
            "annotation_visual_analysis": ann.get("annotation_visual_analysis") if ann else None,
            "annotation_note": ann.get("annotation_note") if ann else None,
        }
        final_rows.append(out_row)

    # Deterministic ordering.
    final_rows.sort(key=lambda x: x["id"])

    _write_jsonl(out_jsonl, final_rows)

    class_counts: Counter[str] = Counter()
    edit_type_counts: Counter[str] = Counter()
    for row in final_rows:
        edit_type_counts[str(row.get("edit_type") or "unknown")] += 1
        for lbl in row.get("taxonomy_labels", []):
            class_counts[str(lbl)] += 1

    total_nonempty = sum(1 for r in final_rows if r.get("taxonomy_labels"))
    total_zero = len(final_rows) - total_nonempty

    summary = {
        "metadata_path": str(metadata_path),
        "annotations_path": str(annotations_path),
        "output_jsonl": str(out_jsonl),
        "total_originals": len(final_rows),
        "total_sft": total_sft,
        "total_annotated_rejected": total_annotated_rejected,
        "excluded_unannotated_rejected": excluded_unannotated_rejected,
        "total_all_zero_taxonomy_rows": total_zero,
        "total_non_empty_taxonomy_rows": total_nonempty,
        "per_class_taxonomy_counts": {k: class_counts.get(k, 0) for k in ERROR_TYPES},
        "per_edit_type_counts": dict(sorted(edit_type_counts.items())),
    }

    ensure_dirs(out_summary.parent)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
