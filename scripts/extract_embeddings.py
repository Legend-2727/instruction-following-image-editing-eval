#!/usr/bin/env python
"""extract_embeddings.py — Pre-compute CLIP embeddings for all samples.

Usage
-----
    python scripts/extract_embeddings.py --data data/sample --out data/sample/embeddings.npz

Produces ``embeddings.npz`` with arrays:
    ids       (N,)    — sample IDs (object array of strings)
    emb_orig  (N, D)  — L2-normalised image embeddings for originals
    emb_edit  (N, D)  — L2-normalised image embeddings for edited images
    emb_text  (N, D)  — L2-normalised CLIP text embeddings for prompts
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_metadata
from utils.clip_encoder import CLIPEmbedder

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings for sample dataset.")
    parser.add_argument("--data", type=str, default="data/sample", help="Sample data directory (contains metadata.jsonl).")
    parser.add_argument("--out", type=str, default=None, help="Output .npz path (default: <data>/embeddings.npz).")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="CLIP model name.")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="Pretrained weights tag.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_path = Path(args.out) if args.out else data_dir / "embeddings.npz"

    # Load metadata
    meta = load_metadata(data_dir / "metadata.jsonl")
    if not meta:
        logger.error("No metadata found at %s/metadata.jsonl", data_dir)
        sys.exit(1)
    logger.info("Loaded %d metadata records.", len(meta))

    ids = [m["id"] for m in meta]
    orig_paths = [str(data_dir / m["orig_path"]) for m in meta]
    edit_paths = [str(data_dir / m["edited_path"]) for m in meta]
    prompts = [m["prompt"] for m in meta]

    # Build embeddings
    embedder = CLIPEmbedder(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )

    logger.info("Encoding original images …")
    emb_orig = embedder.encode_images(orig_paths, batch_size=args.batch_size, desc="orig images")

    logger.info("Encoding edited images …")
    emb_edit = embedder.encode_images(edit_paths, batch_size=args.batch_size, desc="edit images")

    logger.info("Encoding prompts …")
    emb_text = embedder.encode_texts(prompts, batch_size=args.batch_size, desc="prompts")

    # Save
    np.savez(
        str(out_path),
        ids=np.array(ids, dtype=object),
        emb_orig=emb_orig,
        emb_edit=emb_edit,
        emb_text=emb_text,
    )
    logger.info("Saved embeddings (%d samples, dim=%d) → %s", len(ids), emb_orig.shape[1], out_path)


if __name__ == "__main__":
    main()
