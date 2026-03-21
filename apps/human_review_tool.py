"""human_review_tool.py — Streamlit UI for adjudicating dual-judge outputs.

Run from the repo root, for example:

    streamlit run apps/human_review_tool.py -- \
      --queue artifacts/reviews/pilot_100_review_queue.jsonl \
      --review_log artifacts/reviews/human_reviews.jsonl \
      --data_dir /path/to/hf_snapshot_root

The app reads a prioritized review queue, displays source/edited images, both
judge outputs, and appends one schema-compliant review action per submission.
It never mutates the base judged dataset.
"""

from __future__ import annotations

# Streamlit should be imported before other third-party imports.
import streamlit as st

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.io import append_jsonl, ensure_dirs, load_jsonl, resolve_data_path
from utils.review import load_latest_review_actions, sort_review_queue
from utils.schema import ADHERENCE_LABELS, ERROR_TYPES, ReviewActionRecord

DEFAULT_QUEUE_PATH = PROJECT_ROOT / "artifacts" / "reviews" / "review_queue.jsonl"
DEFAULT_REVIEW_LOG = PROJECT_ROOT / "artifacts" / "reviews" / "human_reviews.jsonl"
DEFAULT_DATA_DIR = PROJECT_ROOT
DEFAULT_REVIEWER = "reviewer1"



def parse_args() -> Dict[str, Any]:
    """Parse lightweight Streamlit extra args from ``sys.argv``."""
    args = {
        "queue": DEFAULT_QUEUE_PATH,
        "review_log": DEFAULT_REVIEW_LOG,
        "data_dir": DEFAULT_DATA_DIR,
        "reviewer_id": DEFAULT_REVIEWER,
    }
    raw = sys.argv[1:]
    for idx, token in enumerate(raw):
        if token == "--queue" and idx + 1 < len(raw):
            args["queue"] = Path(raw[idx + 1])
        elif token == "--review_log" and idx + 1 < len(raw):
            args["review_log"] = Path(raw[idx + 1])
        elif token == "--data_dir" and idx + 1 < len(raw):
            args["data_dir"] = Path(raw[idx + 1])
        elif token == "--reviewer_id" and idx + 1 < len(raw):
            args["reviewer_id"] = str(raw[idx + 1])
    return args



def _load_queue(queue_path: Path) -> List[Dict[str, Any]]:
    queue = load_jsonl(queue_path)
    return sort_review_queue(queue)



def _format_label_bundle(adherence: Any, taxonomy: Any) -> str:
    adh = adherence if adherence else "—"
    if isinstance(taxonomy, list):
        tax = ", ".join(taxonomy) if taxonomy else "None"
    else:
        tax = str(taxonomy or "None")
    return f"**Adherence:** {adh}  \\n**Taxonomy:** {tax}"



def _show_image_panel(title: str, path: Path) -> None:
    st.markdown(f"**{title}**")
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Image not found: {path}")



def _latest_action_counts(latest_reviews: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    counts = {"approved": 0, "corrected": 0, "disputed": 0}
    for row in latest_reviews.values():
        action = str(row.get("action_type", "")).strip()
        if action in counts:
            counts[action] += 1
    return counts



def main() -> None:
    cfg = parse_args()

    st.set_page_config(
        page_title="Human Review — Dual Judge Queue",
        page_icon="🔍",
        layout="wide",
    )
    st.title("🔍 Human Review — Dual Judge Adjudication")
    st.caption(
        "Append-only review logging for dual-judge disagreement resolution. "
        "The queue is read-only; only the review log is written."
    )

    queue_path: Path = Path(cfg["queue"])
    review_log_path: Path = Path(cfg["review_log"])
    data_dir: Path = Path(cfg["data_dir"])

    queue_records = _load_queue(queue_path)
    latest_reviews = load_latest_review_actions(review_log_path)
    reviewed_ids = set(latest_reviews.keys())
    action_counts = _latest_action_counts(latest_reviews)

    if not queue_records:
        st.error(
            f"No review queue found at `{queue_path}`. Build one first with `scripts/build_review_queue.py`."
        )
        return

    if "reviewer_id" not in st.session_state:
        st.session_state.reviewer_id = str(cfg["reviewer_id"])
    if "hide_reviewed" not in st.session_state:
        st.session_state.hide_reviewed = True
    if "review_idx" not in st.session_state:
        st.session_state.review_idx = 0

    with st.sidebar:
        st.header("Queue")
        st.session_state.reviewer_id = st.text_input(
            "Reviewer ID",
            value=st.session_state.reviewer_id,
        )
        st.session_state.hide_reviewed = st.checkbox(
            "Hide already reviewed samples",
            value=st.session_state.hide_reviewed,
        )
        st.caption(f"Queue: `{queue_path}`")
        st.caption(f"Review log: `{review_log_path}`")
        st.caption(f"Data dir: `{data_dir}`")
        st.divider()

        total = len(queue_records)
        reviewed = len(reviewed_ids & {str(r.get('id', '')) for r in queue_records})
        remaining = total - reviewed
        st.metric("Reviewed", f"{reviewed}/{total}")
        st.metric("Remaining", remaining)
        st.progress((reviewed / total) if total else 0.0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Approved", action_counts["approved"])
        c2.metric("Corrected", action_counts["corrected"])
        c3.metric("Disputed", action_counts["disputed"])

    visible_queue = (
        [r for r in queue_records if str(r.get("id", "")) not in reviewed_ids]
        if st.session_state.hide_reviewed
        else queue_records
    )

    if not visible_queue:
        st.success("✅ No pending samples remain in the current view.")
        return

    if st.session_state.review_idx >= len(visible_queue):
        st.session_state.review_idx = 0

    idx = st.session_state.review_idx
    sample = visible_queue[idx]
    sample_id = str(sample.get("id", ""))
    latest_review = latest_reviews.get(sample_id)

    source_image = resolve_data_path(data_dir, sample.get("source_image", ""))
    edited_image = resolve_data_path(data_dir, sample.get("edited_image", ""))

    st.markdown(f"### Sample {idx + 1} / {len(visible_queue)} — `{sample_id}`")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Priority", str(sample.get("review_priority", "—")).title())
    meta_cols[1].metric("Mean confidence", f"{float(sample.get('mean_confidence', 0.0) or 0.0):.2f}")
    meta_cols[2].metric("Agreement", "Yes" if bool(sample.get("overall_agreement")) else "No")
    meta_cols[3].metric("Mode", str(sample.get("judge_mode", "unknown")))

    reasons = sample.get("review_reasons", []) or []
    if reasons:
        st.caption("Review reasons: " + ", ".join(reasons))

    instruction = sample.get("instruction_en") or sample.get("instruction") or ""
    st.info(f"**Instruction:** {instruction}")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        _show_image_panel("Source image", source_image)
    with img_col2:
        _show_image_panel("Edited image", edited_image)

    st.markdown("---")
    st.markdown("### Labels and judge outputs")
    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown("**Original / existing labels**")
        st.markdown(
            _format_label_bundle(
                sample.get("original_adherence_label"),
                sample.get("original_taxonomy_labels", []),
            )
        )
        if sample.get("original_label_source"):
            st.caption(f"Source: {sample.get('original_label_source')}")
        st.markdown("**Current provisional labels**")
        st.markdown(
            _format_label_bundle(
                sample.get("previous_adherence_label"),
                sample.get("previous_taxonomy_labels", []),
            )
        )
        st.caption(
            f"Source: {sample.get('previous_label_source', 'unknown')} | "
            f"Confidence: {float(sample.get('previous_label_confidence', 0.0) or 0.0):.2f}"
        )

    with info_col2:
        st.markdown("**Judge A**")
        st.markdown(
            _format_label_bundle(
                sample.get("judge_a_adherence"),
                sample.get("judge_a_taxonomy", []),
            )
        )
        st.caption(f"Confidence: {float(sample.get('judge_a_confidence', 0.0) or 0.0):.2f}")
        raw_a = str(sample.get("judge_a_raw", "")).strip()
        if raw_a:
            st.text_area("Judge A notes", value=raw_a, height=120, disabled=True)

    with info_col3:
        st.markdown("**Judge B**")
        st.markdown(
            _format_label_bundle(
                sample.get("judge_b_adherence"),
                sample.get("judge_b_taxonomy", []),
            )
        )
        st.caption(f"Confidence: {float(sample.get('judge_b_confidence', 0.0) or 0.0):.2f}")
        raw_b = str(sample.get("judge_b_raw", "")).strip()
        if raw_b:
            st.text_area("Judge B notes", value=raw_b, height=120, disabled=True)

    if latest_review:
        st.markdown("---")
        st.warning(
            "This sample already has a latest review action in the log. "
            "Submitting again will append a new row; the latest timestamp wins during merge."
        )
        st.json(latest_review)

    st.markdown("---")
    st.markdown("### Submit human decision")

    default_adh = sample.get("previous_adherence_label")
    if default_adh not in ADHERENCE_LABELS:
        default_adh = ADHERENCE_LABELS[0]
    default_adh_idx = ADHERENCE_LABELS.index(default_adh)

    default_taxonomy = [
        t for t in sample.get("previous_taxonomy_labels", []) if t in ERROR_TYPES
    ]

    with st.form(key=f"review_form_{sample_id}"):
        final_adherence = st.radio(
            "Final adherence",
            ADHERENCE_LABELS,
            index=default_adh_idx,
            horizontal=True,
        )
        final_taxonomy = st.multiselect(
            "Final taxonomy (select all that apply)",
            ERROR_TYPES,
            default=default_taxonomy,
        )
        mark_disputed = st.checkbox(
            "Mark as disputed / needs second reviewer",
            value=False,
        )
        notes = st.text_area("Notes (optional)", value="", height=120)
        submitted = st.form_submit_button(
            "Save review",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        previous_labels = {
            "adherence": sample.get("previous_adherence_label"),
            "taxonomy": sample.get("previous_taxonomy_labels", []),
        }
        updated_labels = {
            "adherence": final_adherence,
            "taxonomy": final_taxonomy,
        }

        if mark_disputed:
            action_type = "disputed"
        elif previous_labels == updated_labels:
            action_type = "approved"
        else:
            action_type = "corrected"

        record = ReviewActionRecord(
            sample_id=sample_id,
            previous_labels=previous_labels,
            updated_labels=updated_labels,
            reviewer_id=str(st.session_state.reviewer_id).strip() or DEFAULT_REVIEWER,
            action_type=action_type,
            notes=notes,
            source="streamlit-review-queue-v1",
        )
        ensure_dirs(review_log_path.parent)
        append_jsonl(record.to_dict(), review_log_path)
        st.success(f"Saved {action_type} review for `{sample_id}`")
        st.session_state.review_idx = min(idx + 1, max(0, len(visible_queue) - 1))
        st.rerun()

    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("⬅️ Previous", disabled=(idx == 0), use_container_width=True):
            st.session_state.review_idx = max(0, idx - 1)
            st.rerun()
    with nav_col2:
        if st.button("⏭️ Skip", use_container_width=True):
            st.session_state.review_idx = min(idx + 1, len(visible_queue) - 1)
            st.rerun()
    with nav_col3:
        jump = st.number_input(
            "Jump to #",
            min_value=1,
            max_value=len(visible_queue),
            value=idx + 1,
            step=1,
            label_visibility="collapsed",
        )
        if st.button("Go", use_container_width=True):
            st.session_state.review_idx = int(jump) - 1
            st.rerun()


if __name__ == "__main__":
    main()
