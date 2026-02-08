"""label_tool.py — Streamlit app for human annotation of image edits.

Launch
------
    streamlit run apps/label_tool.py

Features
--------
- Side-by-side display of original and edited images with the edit prompt.
- Adherence radio selector: Success / Partial / No.
- Multi-select error-type checkboxes (shown only for Partial / No).
- Append-only JSONL output to ``data/annotations/labels.jsonl``.
- Progress counter, Back / Skip / Save & Next navigation.
- Persisted annotator name across session.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Resolve project root so we can import utils ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.io import load_jsonl, append_jsonl, ensure_dirs
from utils.schema import ADHERENCE_LABELS, ERROR_TYPES, LabelRecord

# ── Paths ────────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sample"
LABELS_PATH = PROJECT_ROOT / "data" / "annotations" / "labels.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# Session-state helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _init_state() -> None:
    """Initialise session state on first run."""
    if "inited" in st.session_state:
        return

    # Load metadata
    data_dir_str = st.query_params.get("data", str(DEFAULT_DATA_DIR))
    data_dir = Path(data_dir_str)
    meta_path = data_dir / "metadata.jsonl"
    meta = load_jsonl(meta_path)
    if not meta:
        st.error(f"No metadata found at `{meta_path}`. Run `make_sample_dataset.py` first.")
        st.stop()

    st.session_state.data_dir = data_dir
    st.session_state.meta = meta
    st.session_state.current_idx = 0

    # Load existing labels (keyed by id for quick lookup)
    raw_labels = load_jsonl(LABELS_PATH)
    st.session_state.labels = {r["id"]: r for r in raw_labels}

    st.session_state.annotator = "annotator1"
    st.session_state.inited = True


def _current_record() -> dict:
    return st.session_state.meta[st.session_state.current_idx]


def _total() -> int:
    return len(st.session_state.meta)


def _labeled_count() -> int:
    return len(st.session_state.labels)


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.set_page_config(page_title="Image Edit Labeler", layout="wide")
    _init_state()

    st.title("🏷️ Image Edit — Labeling Tool")

    # ── Sidebar: annotator + progress ────────────────────────────────────
    with st.sidebar:
        st.session_state.annotator = st.text_input(
            "Annotator name", value=st.session_state.annotator
        )
        st.divider()
        labeled = _labeled_count()
        total = _total()
        st.metric("Progress", f"{labeled} / {total}")
        st.progress(labeled / total if total else 0)

        st.divider()
        jump = st.number_input(
            "Jump to sample #", min_value=1, max_value=total,
            value=st.session_state.current_idx + 1, step=1,
        )
        if st.button("Go"):
            st.session_state.current_idx = int(jump) - 1
            st.rerun()

    # ── Main area ────────────────────────────────────────────────────────
    idx = st.session_state.current_idx
    rec = _current_record()
    data_dir = st.session_state.data_dir

    st.markdown(f"**Sample {idx + 1} / {_total()}** — `{rec['id']}`")

    # Images side by side
    col1, col2 = st.columns(2)
    orig_path = data_dir / rec["orig_path"]
    edit_path = data_dir / rec["edited_path"]

    with col1:
        st.subheader("Original")
        if orig_path.exists():
            st.image(str(orig_path), use_container_width=True)
        else:
            st.warning(f"Image not found: {orig_path}")

    with col2:
        st.subheader("Edited")
        if edit_path.exists():
            st.image(str(edit_path), use_container_width=True)
        else:
            st.warning(f"Image not found: {edit_path}")

    st.markdown(f"**Prompt:** {rec['prompt']}")
    st.markdown(f"**Language:** `{rec.get('lang', 'en')}`")

    st.divider()

    # ── Existing label (if any) ──────────────────────────────────────────
    existing = st.session_state.labels.get(rec["id"])
    default_adh_idx = 0
    default_errs: list = []
    if existing:
        st.info("This sample has been labeled before. You can re-label to overwrite.")
        if existing.get("adherence") in ADHERENCE_LABELS:
            default_adh_idx = ADHERENCE_LABELS.index(existing["adherence"])
        default_errs = existing.get("error_types", [])

    # ── Adherence ────────────────────────────────────────────────────────
    adherence = st.radio(
        "Adherence",
        ADHERENCE_LABELS,
        index=default_adh_idx,
        horizontal=True,
    )

    # ── Error types (only for Partial / No) ──────────────────────────────
    error_types: list[str] = []
    if adherence in ("Partial", "No"):
        error_types = st.multiselect(
            "Error types (select all that apply)",
            ERROR_TYPES,
            default=[e for e in default_errs if e in ERROR_TYPES],
        )

    # ── Optional notes ───────────────────────────────────────────────────
    notes = st.text_input("Notes (optional)", value="")

    st.divider()

    # ── Navigation buttons ───────────────────────────────────────────────
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)

    with bcol1:
        if st.button("⬅️ Back", disabled=(idx == 0), use_container_width=True):
            st.session_state.current_idx = max(0, idx - 1)
            st.rerun()

    with bcol2:
        if st.button("⏭️ Skip", disabled=(idx >= _total() - 1), use_container_width=True):
            st.session_state.current_idx = min(_total() - 1, idx + 1)
            st.rerun()

    with bcol3:
        save_disabled = idx >= _total()
        if st.button("💾 Save & Next", type="primary", disabled=save_disabled, use_container_width=True):
            label_rec = LabelRecord(
                id=rec["id"],
                adherence=adherence,
                error_types=error_types,
                annotator_name=st.session_state.annotator,
                notes=notes,
            )
            # Append to file
            ensure_dirs(LABELS_PATH.parent)
            append_jsonl(label_rec.to_dict(), LABELS_PATH)
            # Update in-memory map
            st.session_state.labels[rec["id"]] = label_rec.to_dict()
            # Advance
            st.session_state.current_idx = min(_total() - 1, idx + 1)
            st.rerun()

    with bcol4:
        if st.button("🔄 Undo last", use_container_width=True):
            # Remove last entry from labels file (rewrite without last line)
            if LABELS_PATH.exists():
                lines = LABELS_PATH.read_text(encoding="utf-8").strip().split("\n")
                if lines:
                    removed = json.loads(lines[-1])
                    lines = lines[:-1]
                    LABELS_PATH.write_text(
                        "\n".join(lines) + ("\n" if lines else ""),
                        encoding="utf-8",
                    )
                    # Update in-memory
                    if removed.get("id") in st.session_state.labels:
                        del st.session_state.labels[removed["id"]]
                    st.success(f"Removed last label (id={removed.get('id')})")
                    st.rerun()

    # ── Keyboard shortcut hint ───────────────────────────────────────────
    st.caption("Tip: Use Tab + Enter to navigate buttons quickly.")


if __name__ == "__main__":
    main()
