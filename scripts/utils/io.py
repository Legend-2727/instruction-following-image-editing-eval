"""io.py — Shared I/O helpers for JSONL, images, and directories."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Union

from PIL import Image


# ── Directory helpers ────────────────────────────────────────────────────────
def ensure_dirs(*paths: Union[str, Path]) -> None:
    """Create directories (including parents) if they don't exist."""
    for p in paths:
        os.makedirs(str(p), exist_ok=True)


# ── JSONL helpers ────────────────────────────────────────────────────────────
def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Returns [] if file missing."""
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Overwrite a JSONL file with *records*."""
    path = Path(path)
    ensure_dirs(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(record: Dict[str, Any], path: Union[str, Path]) -> None:
    """Append a single record to a JSONL file (append-only)."""
    path = Path(path)
    ensure_dirs(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Metadata / label convenience wrappers ────────────────────────────────────
def load_metadata(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load metadata.jsonl."""
    return load_jsonl(path)


def save_metadata(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Save metadata.jsonl."""
    save_jsonl(records, path)


def load_labels(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load labels.jsonl."""
    return load_jsonl(path)


def append_label(record: Dict[str, Any], path: Union[str, Path]) -> None:
    """Append one label record to labels.jsonl."""
    append_jsonl(record, path)


# ── Image helpers ────────────────────────────────────────────────────────────
def resize_image(img: Image.Image, max_side: int = 512) -> Image.Image:
    """Resize an image so the longest side is at most *max_side*, preserving
    the aspect ratio. Returns a new RGB image."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)
