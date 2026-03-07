#!/usr/bin/env python
"""step1_load_dataset.py — Stream MagicBrush and save a local dataset.

Downloads N samples from MagicBrush (osunlp/MagicBrush) via HuggingFace
streaming, saves source images, target (ground truth) images, and metadata.
Uses streaming with robust per-sample retry logic to avoid rate-limit issues.

Usage
-----
    python scripts/step1_load_dataset.py --n 5000 --out data/magicbrush --seed 42
    python scripts/step1_load_dataset.py --n 100 --out data/magicbrush --seed 42  # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import ensure_dirs, save_jsonl, load_jsonl, resize_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Increase timeouts for large image datasets
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")


def stream_with_retry(max_retries=10):
    """Create a streaming dataset iterator with retry on network errors."""
    from datasets import load_dataset

    for attempt in range(max_retries):
        try:
            logger.info("Opening MagicBrush stream (attempt %d/%d)...", attempt + 1, max_retries)
            ds = load_dataset("osunlp/MagicBrush", split="train", streaming=True)
            return ds
        except Exception as e:
            wait = min(2 ** attempt * 3, 120)
            logger.warning("Stream open failed: %s. Retry in %ds...", str(e)[:200], wait)
            time.sleep(wait)
    raise RuntimeError("Failed to open MagicBrush stream after %d attempts" % max_retries)


def iterate_with_retry(ds_iter, max_retries=5):
    """Yields samples from a streaming dataset with retry on mid-stream failures."""
    retry_count = 0
    while True:
        try:
            sample = next(ds_iter)
            retry_count = 0  # reset on success
            yield sample
        except StopIteration:
            return
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                logger.error("Too many consecutive failures (%d). Stopping.", retry_count)
                raise
            wait = min(2 ** retry_count * 2, 60)
            logger.warning("Stream error (retry %d/%d): %s. Waiting %ds...",
                           retry_count, max_retries, str(e)[:150], wait)
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Download MagicBrush samples")
    parser.add_argument("--n", type=int, default=5000, help="Number of samples")
    parser.add_argument("--out", type=str, default="data/magicbrush", help="Output dir")
    parser.add_argument("--max_side", type=int, default=512, help="Max image dimension")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from existing metadata")
    args = parser.parse_args()

    out_dir = Path(args.out)
    src_dir = out_dir / "images" / "source"
    tgt_dir = out_dir / "images" / "target"
    ensure_dirs(src_dir, tgt_dir)

    meta_path = out_dir / "metadata.jsonl"

    # Resume support
    done_ids = set()
    existing_records = []
    if args.resume and meta_path.exists():
        existing_records = load_jsonl(meta_path)
        done_ids = {r["id"] for r in existing_records}
        logger.info("Resuming — %d samples already downloaded", len(done_ids))

    total_needed = args.n - len(done_ids)
    if total_needed <= 0:
        logger.info("Already have %d samples. Nothing to do.", len(done_ids))
        return

    # Use streaming to avoid downloading full ~50GB dataset
    ds = stream_with_retry()
    ds_iter = iter(ds)

    records = list(existing_records)
    new_count = 0
    skip_count = 0

    pbar = tqdm(total=total_needed, desc="Downloading MagicBrush")

    for sample in iterate_with_retry(ds_iter):
        if new_count >= total_needed:
            break

        img_id = str(sample["img_id"])
        turn = int(sample["turn_index"])
        uid = f"{img_id}_t{turn}"

        if uid in done_ids:
            skip_count += 1
            continue

        try:
            instruction = sample["instruction"]
            source_img: Image.Image = sample["source_img"]
            target_img: Image.Image = sample["target_img"]

            # Resize and save
            src_path = src_dir / f"{uid}.png"
            tgt_path = tgt_dir / f"{uid}.png"

            resize_image(source_img, args.max_side).save(str(src_path))
            resize_image(target_img, args.max_side).save(str(tgt_path))

            record = {
                "id": uid,
                "img_id": img_id,
                "turn_index": turn,
                "instruction_en": instruction,
                "source_path": f"images/source/{uid}.png",
                "target_path": f"images/target/{uid}.png",
            }
            records.append(record)
            done_ids.add(uid)
            new_count += 1
            pbar.update(1)

            # Periodic save every 50 samples
            if new_count % 50 == 0:
                save_jsonl(records, meta_path)
                logger.info("Checkpoint: %d/%d saved (skipped %d existing)",
                            new_count, total_needed, skip_count)

        except Exception as e:
            logger.warning("Error processing sample %s: %s — skipping", uid, str(e)[:100])
            continue

    pbar.close()
    save_jsonl(records, meta_path)
    logger.info("Done! Saved %d total samples (%d new, %d skipped) → %s",
                len(records), new_count, skip_count, meta_path)


if __name__ == "__main__":
    main()
