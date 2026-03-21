"""schema.py — Canonical labels, review schemas, and record helpers.

This module is the single source of truth for label vocabularies and
serialization formats used across dataset builders, judges, review UIs,
training code, and analysis scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

# ── Adherence ────────────────────────────────────────────────────────────────
ADHERENCE_LABELS: List[str] = ["Success", "Partial", "No"]
ADHERENCE_TO_IDX: Dict[str, int] = {l: i for i, l in enumerate(ADHERENCE_LABELS)}
IDX_TO_ADHERENCE: Dict[int, str] = {i: l for i, l in enumerate(ADHERENCE_LABELS)}

# ── Error taxonomy (fixed ordering, used for multi-label vectors) ────────────
ERROR_TYPES: List[str] = [
    "Wrong Object",
    "Missing Object",
    "Extra Object",
    "Wrong Attribute",
    "Spatial Error",
    "Style Mismatch",
    "Over-editing",
    "Under-editing",
    "Artifact / Quality Issue",
    "Ambiguous Prompt",
    "Failed Removal",
]
ERROR_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(ERROR_TYPES)}
IDX_TO_ERROR: Dict[int, str] = {i: e for i, e in enumerate(ERROR_TYPES)}
NUM_ERROR_TYPES: int = len(ERROR_TYPES)

REVIEW_ACTION_TYPES: List[str] = ["approved", "corrected", "disputed"]
REVIEW_ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(REVIEW_ACTION_TYPES)}


def now_utc_iso() -> str:
    """Return a stable UTC timestamp with trailing ``Z``."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ── Validation helpers ───────────────────────────────────────────────────────
def validate_adherence_label(label: str) -> str:
    """Validate and return a canonical adherence label."""
    if label not in ADHERENCE_TO_IDX:
        raise ValueError(
            f"Unknown adherence label: {label!r}. Valid labels: {ADHERENCE_LABELS}"
        )
    return label



def normalize_error_types(error_types: Optional[Sequence[str]]) -> List[str]:
    """Validate, deduplicate, and order taxonomy labels canonically."""
    if not error_types:
        return []

    ordered: List[str] = []
    seen = set()
    for raw in error_types:
        if raw is None:
            continue
        label = str(raw).strip()
        if not label:
            continue
        if label not in ERROR_TO_IDX:
            raise ValueError(
                f"Unknown error label: {label!r}. Valid labels: {ERROR_TYPES}"
            )
        if label not in seen:
            ordered.append(label)
            seen.add(label)

    ordered.sort(key=lambda x: ERROR_TO_IDX[x])
    return ordered



def _validate_label_bundle(bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if bundle is None:
        raise ValueError("Label bundle may not be None")
    return {
        "adherence": validate_adherence_label(str(bundle.get("adherence", ""))),
        "taxonomy": normalize_error_types(bundle.get("taxonomy", [])),
    }


# ── Data records ─────────────────────────────────────────────────────────────
@dataclass
class MetadataRecord:
    """One row in ``metadata.jsonl``."""

    id: str
    prompt: str
    orig_path: str
    edited_path: str
    split: str = "train"
    lang: str = "en"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MetadataRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LabelRecord:
    """One row in ``labels.jsonl``."""

    id: str
    adherence: str
    error_types: List[str] = field(default_factory=list)
    annotator_name: str = "annotator1"
    timestamp: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = now_utc_iso()
        self.adherence = validate_adherence_label(self.adherence)
        self.error_types = normalize_error_types(self.error_types)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LabelRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def adherence_idx(self) -> int:
        return ADHERENCE_TO_IDX[self.adherence]

    def error_vector(self) -> List[int]:
        """Return a binary vector of length ``NUM_ERROR_TYPES``."""
        vec = [0] * NUM_ERROR_TYPES
        for e in self.error_types:
            vec[ERROR_TO_IDX[e]] = 1
        return vec


# ── VLM Judgment (Checkpoint 2) ──────────────────────────────────────────────
@dataclass
class VLMJudgment:
    """One row in ``vlm_judgments.jsonl`` — produced by the VLM-as-judge."""

    id: str
    adherence: str
    error_types: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    raw_response: str = ""
    model_name: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = now_utc_iso()
        self.adherence = validate_adherence_label(self.adherence)
        self.error_types = normalize_error_types(self.error_types)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VLMJudgment":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def adherence_idx(self) -> int:
        return ADHERENCE_TO_IDX.get(self.adherence, -1)

    def error_vector(self) -> List[int]:
        vec = [0] * NUM_ERROR_TYPES
        for e in self.error_types:
            vec[ERROR_TO_IDX[e]] = 1
        return vec


@dataclass
class HumanReview:
    """Legacy human-review row used by the original Streamlit VLM spot-check UI.

    Kept for backward compatibility with older artifacts. New review tooling
    should prefer :class:`ReviewActionRecord`.
    """

    id: str
    vlm_adherence_correct: bool
    vlm_errors_correct: bool
    human_adherence: str = ""
    human_error_types: List[str] = field(default_factory=list)
    notes: str = ""
    reviewer: str = "reviewer1"
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = now_utc_iso()
        if self.human_adherence:
            self.human_adherence = validate_adherence_label(self.human_adherence)
        self.human_error_types = normalize_error_types(self.human_error_types)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HumanReview":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Review queue / merge records (Phase 2+) ─────────────────────────────────
@dataclass
class ReviewActionRecord:
    """Append-only human review action for the dual-judge merge pipeline."""

    sample_id: str
    previous_labels: Dict[str, Any]
    updated_labels: Dict[str, Any]
    reviewer_id: str
    timestamp_utc: str = ""
    action_type: str = "approved"
    notes: str = ""
    source: str = "streamlit-review-queue-v1"

    def __post_init__(self) -> None:
        if not self.timestamp_utc:
            self.timestamp_utc = now_utc_iso()
        self.previous_labels = _validate_label_bundle(self.previous_labels)
        self.updated_labels = _validate_label_bundle(self.updated_labels)
        if self.action_type not in REVIEW_ACTION_TO_IDX:
            raise ValueError(
                f"Unknown review action: {self.action_type!r}. "
                f"Valid actions: {REVIEW_ACTION_TYPES}"
            )
        if not str(self.sample_id).strip():
            raise ValueError("sample_id may not be empty")
        if not str(self.reviewer_id).strip():
            raise ValueError("reviewer_id may not be empty")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewActionRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
