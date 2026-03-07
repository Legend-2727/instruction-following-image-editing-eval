#!/usr/bin/env python
"""step4_human_review.py — Gradio web app for human verification of VLM annotations.

Loads source/target images + VLM annotations. Human reviewer can:
1. See the images side-by-side with the instruction
2. See the VLM's judgment (adherence, errors, reasoning)
3. Confirm or correct the annotation
4. Save verified annotations

Usage
-----
    python scripts/step4_human_review.py --data data/magicbrush --port 7860
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_jsonl, append_jsonl, ensure_dirs
from utils.schema import ERROR_TYPES, ADHERENCE_LABELS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_app(data_dir: Path):
    """Build and return the Gradio app."""
    import gradio as gr
    from PIL import Image

    annotations_path = data_dir / "vlm_annotations.jsonl"
    reviews_path = data_dir / "human_reviews.jsonl"
    final_path = data_dir / "annotations_final.jsonl"
    ensure_dirs(data_dir)

    # Load data
    annotations = load_jsonl(annotations_path)
    if not annotations:
        raise FileNotFoundError(f"No VLM annotations at {annotations_path}. Run step3 first.")

    # Load metadata for image paths
    metadata = {m["id"]: m for m in load_jsonl(data_dir / "metadata.jsonl")}

    # Load already reviewed
    reviewed = load_jsonl(reviews_path)
    reviewed_ids = {r["id"] for r in reviewed}

    # State
    state = {
        "current_idx": 0,
        "annotations": annotations,
        "reviewed_ids": reviewed_ids,
        "filter_mode": "all",  # all / pending / low_confidence
    }

    def get_filtered_list():
        if state["filter_mode"] == "pending":
            return [a for a in state["annotations"] if a["id"] not in state["reviewed_ids"]]
        elif state["filter_mode"] == "low_confidence":
            return [a for a in state["annotations"]
                    if a.get("confidence", 1.0) < 0.7 and a["id"] not in state["reviewed_ids"]]
        return state["annotations"]

    def load_sample(idx):
        filtered = get_filtered_list()
        if not filtered:
            return None, None, "No samples available", "", [], "", 0.0, f"0/0"

        idx = max(0, min(idx, len(filtered) - 1))
        state["current_idx"] = idx
        ann = filtered[idx]
        uid = ann["id"]

        meta = metadata.get(uid, {})
        src_path = data_dir / meta.get("source_path", "")
        tgt_path = data_dir / meta.get("target_path", "")

        src_img = Image.open(src_path) if src_path.exists() else None
        tgt_img = Image.open(tgt_path) if tgt_path.exists() else None

        instruction = ann.get("instruction_en", "")
        adherence = ann.get("adherence", "No")
        errors = ann.get("error_types", [])
        reasoning = ann.get("reasoning", "")
        confidence = ann.get("confidence", 0.0)

        status = "✅ Reviewed" if uid in state["reviewed_ids"] else "⏳ Pending"
        progress = f"{idx + 1}/{len(filtered)} ({status})"

        return src_img, tgt_img, instruction, adherence, errors, reasoning, confidence, progress

    def save_review(adherence, error_types, notes, idx):
        filtered = get_filtered_list()
        if not filtered:
            return "No samples to review"

        ann = filtered[idx]
        uid = ann["id"]

        review = {
            "id": uid,
            "human_adherence": adherence,
            "human_error_types": error_types if error_types else [],
            "human_error_label_vector": [
                1 if et in (error_types or []) else 0 for et in ERROR_TYPES
            ],
            "vlm_adherence": ann.get("adherence", ""),
            "vlm_adherence_correct": adherence == ann.get("adherence", ""),
            "notes": notes,
            "reviewer": "human_reviewer",
            "timestamp": datetime.utcnow().isoformat(),
        }

        append_jsonl(review, reviews_path)
        state["reviewed_ids"].add(uid)

        # Also append to final annotations
        final = {**ann, **review}
        final["is_verified"] = True
        append_jsonl(final, final_path)

        return f"✅ Saved review for {uid}"

    def go_next(idx):
        return load_sample(idx + 1)

    def go_prev(idx):
        return load_sample(max(0, idx - 1))

    def change_filter(mode):
        state["filter_mode"] = mode
        state["current_idx"] = 0
        return load_sample(0)

    # ── Build UI ──────────────────────────────────────────────────────────
    with gr.Blocks(title="Human Review: VLM Annotations", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔍 Human Review: VLM-as-Judge Verification")
        gr.Markdown("Verify or correct VLM annotations for image editing quality assessment.")

        with gr.Row():
            filter_radio = gr.Radio(
                choices=["all", "pending", "low_confidence"],
                value="all",
                label="Filter",
                interactive=True,
            )
            progress_text = gr.Textbox(label="Progress", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                source_img = gr.Image(label="Source Image", type="pil")
            with gr.Column(scale=1):
                target_img = gr.Image(label="Edited Image (Target)", type="pil")

        with gr.Row():
            instruction_box = gr.Textbox(label="Instruction (English)", lines=2, interactive=False)

        gr.Markdown("### VLM Judgment (auto-generated)")
        with gr.Row():
            vlm_reasoning = gr.Textbox(label="VLM Reasoning", lines=3, interactive=False)
            vlm_confidence = gr.Number(label="VLM Confidence", interactive=False)

        gr.Markdown("### Human Review")
        with gr.Row():
            adherence_input = gr.Radio(
                choices=ADHERENCE_LABELS,
                label="Adherence",
                value="No",
                interactive=True,
            )

        error_checkboxes = gr.CheckboxGroup(
            choices=ERROR_TYPES,
            label="Error Types (select all that apply)",
            interactive=True,
        )

        notes_input = gr.Textbox(label="Notes (optional)", lines=2, interactive=True)

        with gr.Row():
            prev_btn = gr.Button("← Previous", variant="secondary")
            save_btn = gr.Button("💾 Save & Next", variant="primary")
            next_btn = gr.Button("Next →", variant="secondary")

        save_status = gr.Textbox(label="Status", interactive=False)

        # Hidden state
        idx_state = gr.State(0)

        # Outputs
        outputs = [source_img, target_img, instruction_box,
                   adherence_input, error_checkboxes, vlm_reasoning,
                   vlm_confidence, progress_text]

        # Events
        def on_save(adherence, errors, notes, idx):
            status = save_review(adherence, errors, notes, idx)
            new_idx = idx + 1
            out = load_sample(new_idx)
            return [status, new_idx] + list(out)

        save_btn.click(
            on_save,
            inputs=[adherence_input, error_checkboxes, notes_input, idx_state],
            outputs=[save_status, idx_state] + outputs,
        )

        def on_next(idx):
            new_idx = idx + 1
            return [new_idx] + list(load_sample(new_idx))

        next_btn.click(on_next, inputs=[idx_state], outputs=[idx_state] + outputs)

        def on_prev(idx):
            new_idx = max(0, idx - 1)
            return [new_idx] + list(load_sample(new_idx))

        prev_btn.click(on_prev, inputs=[idx_state], outputs=[idx_state] + outputs)

        def on_filter(mode):
            state["filter_mode"] = mode
            return [0] + list(load_sample(0))

        filter_radio.change(on_filter, inputs=[filter_radio], outputs=[idx_state] + outputs)

        # Load first sample on start
        app.load(lambda: list(load_sample(0)), outputs=outputs)

    return app


def main():
    parser = argparse.ArgumentParser(description="Human review web app for VLM annotations")
    parser.add_argument("--data", type=str, default="data/magicbrush")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args()

    data_dir = Path(args.data)
    app = build_app(data_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
