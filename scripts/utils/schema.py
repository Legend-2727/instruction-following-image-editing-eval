"""schema.py — Canonical labels, error taxonomy, and data-record helpers.

This module is the single source of truth for label vocabularies and
serialization formats used across the dataset sampler, labeling tool,
baseline trainer, and analysis script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict

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
    "Failed Removal",           # optional, always last
]
ERROR_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(ERROR_TYPES)}
IDX_TO_ERROR: Dict[int, str] = {i: e for i, e in enumerate(ERROR_TYPES)}
NUM_ERROR_TYPES: int = len(ERROR_TYPES)


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
    adherence: str                          # one of ADHERENCE_LABELS
    error_types: List[str] = field(default_factory=list)
    annotator_name: str = "annotator1"
    timestamp: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LabelRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    # ── helpers ──────────────────────────────────────────────────────────
    def adherence_idx(self) -> int:
        return ADHERENCE_TO_IDX[self.adherence]

    def error_vector(self) -> List[int]:
        """Return a binary vector of length NUM_ERROR_TYPES."""
        vec = [0] * NUM_ERROR_TYPES
        for e in self.error_types:
            if e in ERROR_TO_IDX:
                vec[ERROR_TO_IDX[e]] = 1
        return vec
