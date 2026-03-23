"""review.py — Shared helpers for review queue building and review-log merging."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .io import load_jsonl
from .schema import normalize_error_types, validate_adherence_label


HIGH_PRIORITY = "high"
MEDIUM_PRIORITY = "medium"
LOW_PRIORITY = "low"


def parse_timestamp_utc(raw_value: Optional[str]) -> datetime:
    """Parse common ISO-8601 timestamp variants into a timezone-aware datetime."""
    raw = (raw_value or "").strip()
    if not raw:
        return datetime.min.replace(tzinfo=timezone.utc)

    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)



def choose_provisional_labels(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Derive a deterministic machine-provisional label bundle from dual-judge output.

    Policy:
    - if overall_agreement is true, use judge consensus
    - if one judge parse-failed, use the other judge if available
    - otherwise pick the higher-confidence judge; ties go to judge A
    - return None if both judges parse-failed
    """
    judge_a_adh = row.get("judge_a_adherence")
    judge_b_adh = row.get("judge_b_adherence")
    judge_a_tax = row.get("judge_a_taxonomy", [])
    judge_b_tax = row.get("judge_b_taxonomy", [])
    judge_a_parse_failed = bool(row.get("judge_a_parse_failed", False))
    judge_b_parse_failed = bool(row.get("judge_b_parse_failed", False))

    # If both parse failed, no provisional labels possible
    if judge_a_parse_failed and judge_b_parse_failed:
        return None
    
    # If one parse failed, use the other
    if judge_a_parse_failed:
        return {
            "adherence": validate_adherence_label(judge_b_adh),
            "taxonomy": normalize_error_types(judge_b_tax),
            "source": "judge_b_only_a_failed",
            "selected_judge": "judge_b",
            "selected_confidence": float(row.get("judge_b_confidence", 0.0) or 0.0),
        }
    
    if judge_b_parse_failed:
        return {
            "adherence": validate_adherence_label(judge_a_adh),
            "taxonomy": normalize_error_types(judge_a_tax),
            "source": "judge_a_only_b_failed",
            "selected_judge": "judge_a",
            "selected_confidence": float(row.get("judge_a_confidence", 0.0) or 0.0),
        }

    # Both parsed successfully
    if not judge_a_adh and not judge_b_adh:
        return None

    judge_a_conf = float(row.get("judge_a_confidence", 0.0) or 0.0)
    judge_b_conf = float(row.get("judge_b_confidence", 0.0) or 0.0)

    if bool(row.get("overall_agreement")):
        adherence = validate_adherence_label(judge_a_adh)
        taxonomy = normalize_error_types(judge_a_tax)
        return {
            "adherence": adherence,
            "taxonomy": taxonomy,
            "source": "judge_consensus",
            "selected_judge": "consensus",
            "selected_confidence": float(row.get("mean_confidence", judge_a_conf)),
        }

    use_judge_b = judge_b_conf > judge_a_conf
    selected_judge = "judge_b" if use_judge_b else "judge_a"
    adherence = judge_b_adh if use_judge_b else judge_a_adh
    taxonomy = judge_b_tax if use_judge_b else judge_a_tax
    confidence = judge_b_conf if use_judge_b else judge_a_conf

    return {
        "adherence": validate_adherence_label(adherence),
        "taxonomy": normalize_error_types(taxonomy),
        "source": selected_judge,
        "selected_judge": selected_judge,
        "selected_confidence": confidence,
    }



def extract_existing_labels(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract baseline human/original labels if present on a record."""
    
    # Handle the raw dataset's source_type annotation directly:
    source_type = row.get("source_type")
    if source_type in ("sft", "preference_rejected", "baseline_good"):
        if source_type in ("sft", "baseline_good"):
            adherence = "Success"
        else:
            adherence = "Partial/No" # Cannot differentiate Partial vs No from preference_rejected alone
        return {
            "adherence": adherence,
            "taxonomy": [],
            "source": f"raw_dataset ({source_type})",
        }

    candidates: List[Tuple[str, str, str]] = [
        ("adherence_label", "taxonomy_labels", "existing_dataset_label"),
        ("final_adherence", "final_taxonomy", "existing_final_label"),
        ("original_adherence_label", "original_taxonomy_labels", "queue_original_label"),
        ("adherence", "error_types", "legacy_label_record"),
    ]

    for adh_key, tax_key, source in candidates:
        adherence = row.get(adh_key)
        taxonomy = row.get(tax_key, [])
        if not adherence:
            continue
        try:
            return {
                "adherence": validate_adherence_label(adherence),
                "taxonomy": normalize_error_types(taxonomy),
                "source": source,
            }
        except ValueError:
            continue
    return None



def compute_review_priority(
    row: Dict[str, Any],
    low_confidence: float = 0.7,
    medium_confidence: float = 0.8,
) -> Dict[str, Any]:
    """Assign review priority using the queue policy.
    
    Priorities (in order):
    1. Parse failures (either or both judges failed to parse)
    2. Judge disagreement or very low confidence (<0.7)
    3. Adherence disagreement or moderate confidence (<0.8)
    4. Spot check (high agreement + high confidence)
    """
    reasons: List[str] = []
    mean_conf = float(row.get("mean_confidence", 0.0) or 0.0)
    adherence_agreement = bool(row.get("adherence_agreement"))
    overall_agreement = bool(row.get("overall_agreement"))
    
    # HIGHEST PRIORITY: Parse failures
    judge_a_parse_failed = bool(row.get("judge_a_parse_failed", False))
    judge_b_parse_failed = bool(row.get("judge_b_parse_failed", False))
    
    if judge_a_parse_failed or judge_b_parse_failed:
        reasons.append("parse_failure")
        if judge_a_parse_failed:
            reasons.append("judge_a_parse_failed")
        if judge_b_parse_failed:
            reasons.append("judge_b_parse_failed")
        return {
            "review_priority": HIGH_PRIORITY,
            "review_priority_rank": 0,
            "review_reasons": reasons,
        }
    
    # HIGH PRIORITY: Judge disagreement or very low confidence
    if not overall_agreement or mean_conf < low_confidence:
        reasons.append("judge_disagreement")
        return {
            "review_priority": HIGH_PRIORITY,
            "review_priority_rank": 0,
            "review_reasons": reasons,
        }
    
    # MEDIUM PRIORITY: Adherence disagreement or moderate confidence
    if (not adherence_agreement) or (mean_conf < medium_confidence):
        reasons = []
        if not adherence_agreement:
            reasons.append("adherence_disagreement")
        if mean_conf < medium_confidence:
            reasons.append("moderate_confidence")
        return {
            "review_priority": MEDIUM_PRIORITY,
            "review_priority_rank": 1,
            "review_reasons": reasons,
        }
    
    # LOW PRIORITY: High agreement + high confidence (spot check)
    return {
        "review_priority": LOW_PRIORITY,
        "review_priority_rank": 2,
        "review_reasons": ["spot_check"],
    }



def sort_review_queue(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort queue items deterministically by priority, confidence, and id."""
    return sorted(
        records,
        key=lambda r: (
            int(r.get("review_priority_rank", 99)),
            float(r.get("mean_confidence", 1.0) or 1.0),
            str(r.get("id", "")),
        ),
    )



def load_latest_review_actions(review_log_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load an append-only review log and keep only the latest action per sample."""
    path = Path(review_log_path)
    if not path.exists():
        return {}

    latest: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
        if not sample_id:
            continue
        current = latest.get(sample_id)
        if current is None:
            latest[sample_id] = row
            continue
        if parse_timestamp_utc(row.get("timestamp_utc") or row.get("timestamp")) >= parse_timestamp_utc(
            current.get("timestamp_utc") or current.get("timestamp")
        ):
            latest[sample_id] = row
    return latest
