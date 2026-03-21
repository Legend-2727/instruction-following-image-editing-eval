#!/usr/bin/env python
"""vlm_judge.py — Use Qwen2.5-VL as an automated judge for image edits.

Loads the VLM, iterates over generated edits, and produces structured
judgments saved to ``vlm_judgments.jsonl``.

Usage
-----
    # Quick test (10 samples)
    python scripts/vlm_judge.py --data data/eval

    # With specific VLM model
    python scripts/vlm_judge.py --data data/eval --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ

    # Resume interrupted run
    python scripts/vlm_judge.py --data data/eval --resume
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

from utils.io import load_metadata, load_jsonl, append_jsonl, ensure_dirs, resolve_data_path
from utils.schema import VLMJudgment

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_judgments(
    data_dir: Path,
    out_path: Path,
    model_id: str,
    resume: bool,
    device: str,
) -> None:
    """Load VLM, judge all edits, save to JSONL."""
    from utils.vlm_evaluator import QwenVLMJudge

    meta_path = data_dir / "metadata.jsonl"
    metadata = load_metadata(meta_path)
    if not metadata:
        logger.error("No metadata found at %s", meta_path)
        sys.exit(1)

    logger.info("Found %d samples to judge in %s", len(metadata), meta_path)

    # Resume support
    done_ids = set()
    if resume and out_path.exists():
        existing = load_jsonl(out_path)
        done_ids = {r["id"] for r in existing}
        logger.info("Resuming — %d judgments already done", len(done_ids))

    todo = [m for m in metadata if m["id"] not in done_ids]
    if not todo:
        logger.info("All samples already judged. Nothing to do.")
        return

    logger.info("%d samples remaining to judge", len(todo))

    # Load VLM
    judge = QwenVLMJudge(model_id=model_id, device=device)

    ensure_dirs(out_path.parent)

    for meta in tqdm(todo, desc="VLM judging"):
        uid = meta["id"]
        prompt = meta["prompt"]
        orig_path = resolve_data_path(data_dir, meta["orig_path"])
        edit_path = resolve_data_path(data_dir, meta["edited_path"])

        if not orig_path.exists() or not edit_path.exists():
            logger.warning("Missing images for %s — skipping", uid)
            continue

        orig_img = Image.open(orig_path)
        edit_img = Image.open(edit_path)

        t0 = time.time()
        try:
            result = judge.evaluate(orig_img, edit_img, prompt)
        except Exception as e:
            logger.warning("VLM failed for %s: %s — skipping", uid, e)
            continue
        elapsed = time.time() - t0

        judgment = VLMJudgment(
            id=uid,
            adherence=result["adherence"],
            error_types=result.get("error_types", []),
            reasoning=result.get("reasoning", ""),
            confidence=result.get("confidence", 0.0),
            raw_response=result.get("raw_response", ""),
            model_name=result.get("model_name", model_id),
        )

        append_jsonl(judgment.to_dict(), out_path)
        logger.debug(
            "%s → %s (%.1fs) errors=%s",
            uid, judgment.adherence, elapsed, judgment.error_types,
        )

    judge.unload()
    logger.info("All judgments saved → %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="VLM-as-judge: evaluate image edits with Qwen2.5-VL"
    )
    parser.add_argument("--data", type=str, default="data/eval", help="Data dir")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSONL path (default: <data>/vlm_judgments.jsonl)"
    )
    parser.add_argument(
        "--model", type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Qwen VLM model ID",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_path = Path(args.out) if args.out else data_dir / "vlm_judgments.jsonl"

    run_judgments(
        data_dir=data_dir,
        out_path=out_path,
        model_id=args.model,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
