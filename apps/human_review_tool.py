"""human_review_tool.py — Streamlit UI for spot-checking VLM judgments.

Run:
    streamlit run apps/human_review_tool.py -- --data data/eval

The reviewer sees:
  1. Original image, Model-edited image, Ground Truth (side by side)
  2. The edit instruction
  3. The VLM's judgment (adherence + errors + reasoning)
  4. Buttons: "VLM Correct" / "VLM Wrong" + option to provide corrected labels

Results saved to ``data/eval/human_reviews.jsonl``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

# Streamlit must be imported first
import streamlit as st

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils.io import load_jsonl, append_jsonl, ensure_dirs
from utils.schema import ADHERENCE_LABELS, ERROR_TYPES, VLMJudgment, HumanReview

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/eval")
REVIEW_PATH = DATA_DIR / "human_reviews.jsonl"


def parse_args():
    """Parse --data from streamlit's special arg handling."""
    args = sys.argv[1:]
    data_dir = DATA_DIR
    for i, a in enumerate(args):
        if a == "--data" and i + 1 < len(args):
            data_dir = Path(args[i + 1])
    return data_dir


def load_review_data(data_dir: Path):
    """Load metadata + VLM judgments, return merged list."""
    meta = {m["id"]: m for m in load_jsonl(data_dir / "metadata.jsonl")}
    judgments = load_jsonl(data_dir / "vlm_judgments.jsonl")

    merged = []
    for j in judgments:
        uid = j["id"]
        if uid in meta:
            merged.append({**meta[uid], **j})
    return merged


def get_reviewed_ids(review_path: Path) -> set:
    """Load already reviewed sample IDs."""
    reviews = load_jsonl(review_path)
    return {r["id"] for r in reviews}


# ── Streamlit App ────────────────────────────────────────────────────────────
def main():
    data_dir = parse_args()

    st.set_page_config(
        page_title="Human Review: VLM Judgments",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Human Review: VLM-as-Judge Verification")
    st.markdown(
        "Verify whether the VLM judge correctly evaluated each image edit. "
        "This helps measure **VLM-human agreement** and calibrate the automated pipeline."
    )

    # Load data
    all_samples = load_review_data(data_dir)
    if not all_samples:
        st.error(f"No data found in `{data_dir}`. Run `generate_edits.py` and `vlm_judge.py` first.")
        return

    reviewed_ids = get_reviewed_ids(data_dir / "human_reviews.jsonl")
    pending = [s for s in all_samples if s["id"] not in reviewed_ids]

    # Sidebar stats
    with st.sidebar:
        st.header("Progress")
        total = len(all_samples)
        done = len(reviewed_ids)
        st.metric("Reviewed", f"{done}/{total}")
        st.progress(done / total if total else 0)

        if done > 0:
            reviews = load_jsonl(data_dir / "human_reviews.jsonl")
            correct_adh = sum(1 for r in reviews if r.get("vlm_adherence_correct", False))
            correct_err = sum(1 for r in reviews if r.get("vlm_errors_correct", False))
            st.metric("VLM Adherence Accuracy", f"{correct_adh}/{done} ({100*correct_adh/done:.0f}%)")
            st.metric("VLM Error-Type Accuracy", f"{correct_err}/{done} ({100*correct_err/done:.0f}%)")

        st.divider()
        st.caption(f"Data dir: `{data_dir}`")

    if not pending:
        st.success(f"✅ All {total} samples reviewed! VLM-human agreement stats are in the sidebar.")
        return

    # Session state for navigation
    if "review_idx" not in st.session_state:
        st.session_state.review_idx = 0

    idx = st.session_state.review_idx
    if idx >= len(pending):
        st.session_state.review_idx = 0
        idx = 0

    sample = pending[idx]
    uid = sample["id"]
    prompt = sample.get("prompt_meta", sample.get("prompt", ""))

    st.markdown(f"### Sample {idx + 1} / {len(pending)} remaining — `{uid}`")

    # ── Instruction ──────────────────────────────────────────────────────
    st.info(f"**Edit instruction:** {prompt}")

    # ── Images side by side ──────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    orig_path = data_dir / sample.get("orig_path", "")
    edit_path = data_dir / sample.get("edited_path", "")
    gt_path   = data_dir / sample.get("gt_path", "")

    with col1:
        st.markdown("**Original**")
        if orig_path.exists():
            st.image(str(orig_path), use_container_width=True)
        else:
            st.warning("Image not found")

    with col2:
        st.markdown("**Model Edit (InstructPix2Pix)**")
        if edit_path.exists():
            st.image(str(edit_path), use_container_width=True)
        else:
            st.warning("Image not found")

    with col3:
        st.markdown("**Ground Truth (MagicBrush)**")
        if gt_path and Path(gt_path).exists():
            st.image(str(gt_path), use_container_width=True)
        else:
            st.caption("Not available")

    # ── VLM Judgment Display ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 VLM Judgment")

    vlm_adherence = sample.get("adherence_vlm", sample.get("adherence", "N/A"))
    vlm_errors = sample.get("error_types_vlm", sample.get("error_types", []))
    vlm_reasoning = sample.get("reasoning", "")
    vlm_confidence = sample.get("confidence", 0.0)

    if isinstance(vlm_errors, list):
        vlm_errors_str = ", ".join(vlm_errors) if vlm_errors else "None"
    else:
        vlm_errors_str = str(vlm_errors)

    adh_color = {"Success": "🟢", "Partial": "🟡", "No": "🔴"}.get(vlm_adherence, "⚪")

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.markdown(f"**Adherence:** {adh_color} {vlm_adherence}")
        st.markdown(f"**Confidence:** {vlm_confidence:.2f}")
    with vcol2:
        st.markdown(f"**Error types:** {vlm_errors_str}")
        st.markdown(f"**Reasoning:** {vlm_reasoning}")

    # ── Human Review Form ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ✅ Your Review")

    with st.form(key=f"review_{uid}"):
        r1, r2 = st.columns(2)

        with r1:
            adh_correct = st.radio(
                "Is the VLM's **adherence** rating correct?",
                ["Yes", "No"],
                index=0,
                horizontal=True,
            )

        with r2:
            err_correct = st.radio(
                "Are the VLM's **error types** correct?",
                ["Yes", "No"],
                index=0,
                horizontal=True,
            )

        # If VLM is wrong, let human provide corrections
        st.markdown("**If VLM is wrong, provide your corrections below:**")
        c1, c2 = st.columns(2)
        with c1:
            human_adh = st.selectbox(
                "Correct adherence",
                ["(keep VLM)"] + ADHERENCE_LABELS,
                index=0,
            )
        with c2:
            human_errors = st.multiselect(
                "Correct error types",
                ERROR_TYPES,
                default=[],
            )

        notes = st.text_input("Notes (optional)", "")

        submitted = st.form_submit_button("Submit Review", type="primary", use_container_width=True)

    if submitted:
        review = HumanReview(
            id=uid,
            vlm_adherence_correct=(adh_correct == "Yes"),
            vlm_errors_correct=(err_correct == "Yes"),
            human_adherence=human_adh if human_adh != "(keep VLM)" else vlm_adherence,
            human_error_types=human_errors if human_errors else (vlm_errors if isinstance(vlm_errors, list) else []),
            notes=notes,
            reviewer="reviewer1",
        )

        ensure_dirs((data_dir / "human_reviews.jsonl").parent)
        append_jsonl(review.to_dict(), data_dir / "human_reviews.jsonl")

        st.success(f"✅ Saved review for `{uid}`")
        st.session_state.review_idx = idx + 1
        st.rerun()

    # ── Navigation ───────────────────────────────────────────────────────
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("⬅️ Previous", disabled=(idx == 0)):
            st.session_state.review_idx = max(0, idx - 1)
            st.rerun()
    with nav2:
        if st.button("⏭️ Skip"):
            st.session_state.review_idx = idx + 1
            st.rerun()
    with nav3:
        st.caption(f"Remaining: {len(pending) - idx}")


if __name__ == "__main__":
    main()
