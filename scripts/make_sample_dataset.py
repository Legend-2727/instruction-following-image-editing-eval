#!/usr/bin/env python
"""make_sample_dataset.py — Stream MagicBrush and build a tiny local dataset.

Usage
-----
    # Stream from HuggingFace (default)
    python scripts/make_sample_dataset.py --n 200 --out data/sample --max_side 512 --seed 42

    # Local-folder fallback (no internet)
    python scripts/make_sample_dataset.py --local_orig path/to/orig --local_edited path/to/edited \\
        --out data/sample --max_side 512
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

# Allow running as `python scripts/make_sample_dataset.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import ensure_dirs, resize_image, save_jsonl
from utils.schema import MetadataRecord

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming mode (HuggingFace datasets)
# ═══════════════════════════════════════════════════════════════════════════════
def stream_magicbrush(
    n: int,
    out_dir: Path,
    max_side: int,
    seed: int,
) -> None:
    """Download *n* samples from ``osunlp/MagicBrush`` via streaming."""
    from datasets import load_dataset

    logger.info("Streaming MagicBrush (train split) — taking %d samples …", n)
    ds = load_dataset("osunlp/MagicBrush", split="train", streaming=True)
    # NOTE: .shuffle() with streaming forces downloading multiple parquet shards
    # (each 100+ MB with embedded images), which is extremely slow.
    # For speed, we skip shuffle and just take the first N items.
    # To get diversity across the dataset, we skip a random offset instead.
    import random as _rng
    _rng.seed(seed)
    skip_n = _rng.randint(0, 200)  # skip a small random prefix for variety
    ds = ds.skip(skip_n)

    orig_dir = out_dir / "images" / "orig"
    edit_dir = out_dir / "images" / "edited"
    ensure_dirs(orig_dir, edit_dir)

    records = []
    for i, sample in enumerate(tqdm(ds, total=n, desc="Downloading")):
        if i >= n:
            break

        img_id = str(sample["img_id"])
        turn = int(sample["turn_index"])
        uid = f"{img_id}_{turn}"
        prompt = sample["instruction"]

        # source_img / target_img are PIL Images (decoded by datasets)
        orig_img: Image.Image = sample["source_img"]
        edit_img: Image.Image = sample["target_img"]

        # Resize and save
        orig_path = orig_dir / f"{uid}.png"
        edit_path = edit_dir / f"{uid}.png"

        resize_image(orig_img, max_side).save(str(orig_path))
        resize_image(edit_img, max_side).save(str(edit_path))

        records.append(
            MetadataRecord(
                id=uid,
                prompt=prompt,
                orig_path=str(orig_path.relative_to(out_dir)),
                edited_path=str(edit_path.relative_to(out_dir)),
                split="train",
                lang="en",
            ).to_dict()
        )

    meta_path = out_dir / "metadata.jsonl"
    save_jsonl(records, meta_path)
    logger.info("Saved %d records → %s", len(records), meta_path)

    # Write stub translations.csv
    _write_translations_stub(out_dir, records)


# ═══════════════════════════════════════════════════════════════════════════════
# Local-folder fallback
# ═══════════════════════════════════════════════════════════════════════════════
def local_folder_mode(
    orig_dir: Path,
    edited_dir: Path,
    out_dir: Path,
    max_side: int,
) -> None:
    """Build a dataset from user-supplied local image folders.

    Files are matched by name (stem).  E.g. ``orig/001.png`` pairs with
    ``edited/001.png``.
    """
    logger.info("Local-folder mode: orig=%s, edited=%s", orig_dir, edited_dir)

    orig_files = {p.stem: p for p in sorted(orig_dir.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")}
    edit_files = {p.stem: p for p in sorted(edited_dir.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")}
    common = sorted(set(orig_files) & set(edit_files))
    if not common:
        logger.error("No matching file pairs found. Check folder contents.")
        sys.exit(1)

    dst_orig = out_dir / "images" / "orig"
    dst_edit = out_dir / "images" / "edited"
    ensure_dirs(dst_orig, dst_edit)

    records = []
    for stem in tqdm(common, desc="Processing local images"):
        orig_img = Image.open(orig_files[stem])
        edit_img = Image.open(edit_files[stem])

        orig_path = dst_orig / f"{stem}.png"
        edit_path = dst_edit / f"{stem}.png"

        resize_image(orig_img, max_side).save(str(orig_path))
        resize_image(edit_img, max_side).save(str(edit_path))

        records.append(
            MetadataRecord(
                id=stem,
                prompt="(local — no prompt)",
                orig_path=str(orig_path.relative_to(out_dir)),
                edited_path=str(edit_path.relative_to(out_dir)),
                split="local",
                lang="en",
            ).to_dict()
        )

    meta_path = out_dir / "metadata.jsonl"
    save_jsonl(records, meta_path)
    logger.info("Saved %d records → %s", len(records), meta_path)
    _write_translations_stub(out_dir, records)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _write_translations_stub(out_dir: Path, records: list) -> None:
    """Create a small ``translations.csv`` with English + Bengali stubs."""
    csv_path = out_dir / "translations.csv"
    if csv_path.exists():
        logger.info("translations.csv already exists — skipping stub creation.")
        return
    lines = ["id,lang,prompt"]
    # Take first 3 records as examples
    for rec in records[:3]:
        sid = rec["id"]
        prompt = rec["prompt"].replace('"', '""')
        lines.append(f'{sid},en,"{prompt}"')
        lines.append(f'{sid},bn,"(বাংলা অনুবাদ এখানে লিখুন)"')  # placeholder Bengali
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote stub translations.csv with %d example rows.", len(lines) - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a small sample dataset for Checkpoint 1.")
    p.add_argument("--n", type=int, default=200, help="Number of samples to stream (default: 200).")
    p.add_argument("--out", type=str, default="data/sample", help="Output directory.")
    p.add_argument("--max_side", type=int, default=512, help="Max side length for resized images.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    p.add_argument("--local_orig", type=str, default=None, help="Path to local original-image dir (fallback mode).")
    p.add_argument("--local_edited", type=str, default=None, help="Path to local edited-image dir (fallback mode).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    ensure_dirs(out_dir)

    if args.local_orig and args.local_edited:
        local_folder_mode(
            Path(args.local_orig),
            Path(args.local_edited),
            out_dir,
            args.max_side,
        )
    else:
        try:
            stream_magicbrush(args.n, out_dir, args.max_side, args.seed)
        except Exception as exc:
            logger.error("Streaming failed: %s", exc)
            logger.info(
                "Re-run with --local_orig <dir> --local_edited <dir> "
                "to use local-folder fallback mode."
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
