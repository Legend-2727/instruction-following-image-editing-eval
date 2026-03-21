#!/usr/bin/env python
"""generate_edits.py — Generate edited images using InstructPix2Pix.

Streams samples from MagicBrush, applies InstructPix2Pix edits, and saves
the results alongside originals for VLM evaluation.

Usage
-----
    # Quick test (10 samples)
    python scripts/generate_edits.py --n 10 --out data/eval

    # Full run (500 samples)
    python scripts/generate_edits.py --n 500 --out data/eval

    # Resume interrupted run
    python scripts/generate_edits.py --n 500 --out data/eval --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import ensure_dirs, save_jsonl, load_jsonl, resize_image
from utils.schema import MetadataRecord

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def stream_and_edit(
    n: int,
    out_dir: Path,
    max_side: int,
    seed: int,
    resume: bool,
    device: str,
    guidance_scale: float,
    image_guidance_scale: float,
    num_inference_steps: int,
) -> None:
    """Stream MagicBrush, generate IP2P edits, save triplets."""
    from datasets import load_dataset
    from utils.editor_model import load_editor

    orig_dir = out_dir / "images" / "orig"
    edit_dir = out_dir / "images" / "model_edited"
    gt_dir   = out_dir / "images" / "ground_truth"
    ensure_dirs(orig_dir, edit_dir, gt_dir)

    meta_path = out_dir / "metadata.jsonl"

    # Resume support
    done_ids = set()
    existing_records = []
    if resume and meta_path.exists():
        existing_records = load_jsonl(meta_path)
        done_ids = {r["id"] for r in existing_records}
        logger.info("Resuming — %d samples already done", len(done_ids))

    # Load editor
    editor = load_editor(
        device=device,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    # Stream dataset
    logger.info("Streaming MagicBrush (train split) — targeting %d samples …", n)
    ds = load_dataset("osunlp/MagicBrush", split="train", streaming=True)

    import random as _rng
    _rng.seed(seed)
    skip_n = _rng.randint(0, 200)
    ds = ds.skip(skip_n)

    records = list(existing_records)
    new_count = 0
    total_needed = n - len(done_ids)

    if total_needed <= 0:
        logger.info("All %d samples already generated. Nothing to do.", n)
        editor.unload()
        return

    pbar = tqdm(total=total_needed, desc="Generating edits")

    for sample in ds:
        if new_count >= total_needed:
            break

        img_id = str(sample["img_id"])
        turn = int(sample["turn_index"])
        uid = f"{img_id}_{turn}"

        if uid in done_ids:
            continue

        prompt = sample["instruction"]
        orig_img: Image.Image = sample["source_img"]
        gt_img: Image.Image = sample["target_img"]

        # Resize original
        orig_resized = resize_image(orig_img, max_side)

        # Generate edit with InstructPix2Pix
        t0 = time.time()
        try:
            model_edited = editor.edit(orig_resized, prompt)
            model_edited = resize_image(model_edited, max_side)
        except Exception as e:
            logger.warning("Edit failed for %s: %s — skipping", uid, e)
            continue
        elapsed = time.time() - t0

        # Save images
        orig_path = orig_dir / f"{uid}.png"
        edit_path = edit_dir / f"{uid}.png"
        gt_path   = gt_dir / f"{uid}.png"

        orig_resized.save(str(orig_path))
        model_edited.save(str(edit_path))
        resize_image(gt_img, max_side).save(str(gt_path))

        records.append(
            MetadataRecord(
                id=uid,
                prompt=prompt,
                orig_path=f"images/orig/{uid}.png",
                edited_path=f"images/model_edited/{uid}.png",
                split="train",
                lang="en",
            ).to_dict()
        )
        # Also store the ground-truth path in the metadata
        records[-1]["gt_path"] = f"images/ground_truth/{uid}.png"
        records[-1]["editor_model"] = editor.model_id
        records[-1]["edit_time_sec"] = round(elapsed, 2)

        new_count += 1
        pbar.update(1)
        pbar.set_postfix({"last": f"{elapsed:.1f}s", "id": uid})

    pbar.close()
    save_jsonl(records, meta_path)
    logger.info(
        "Done! %d total samples (%d new) → %s",
        len(records), new_count, meta_path,
    )

    # Free GPU
    editor.unload()


def main():
    parser = argparse.ArgumentParser(
        description="Generate InstructPix2Pix edits from MagicBrush"
    )
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--out", type=str, default="data/eval", help="Output dir")
    parser.add_argument("--max_side", type=int, default=512, help="Max image dim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Continue from last run")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    args = parser.parse_args()

    stream_and_edit(
        n=args.n,
        out_dir=Path(args.out),
        max_side=args.max_side,
        seed=args.seed,
        resume=args.resume,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )


if __name__ == "__main__":
    main()
