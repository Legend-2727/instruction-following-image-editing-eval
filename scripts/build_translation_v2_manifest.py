#!/usr/bin/env python
"""build_translation_v2_manifest.py — create one translation job per id/lang.

Examples
--------
    python scripts/build_translation_v2_manifest.py \
        --out artifacts/translation_v2/translation_jobs.jsonl

    python scripts/build_translation_v2_manifest.py \
        --local_dir /path/to/xlingual_dataset_snapshot \
        --out artifacts/translation_v2/translation_jobs_local.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

from utils.hf_xlingual_loader import XlingualPicoBanana

TARGET_LANGS = ["bn", "hi", "ne"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build translation_v2 manifest from HF dataset")
    parser.add_argument("--repo_id", type=str, default="Legend2727/xLingual-picobanana-12k")
    parser.add_argument("--local_dir", type=str, default=None)
    parser.add_argument("--out", type=str, default="artifacts/translation_v2/translation_jobs.jsonl")
    parser.add_argument("--langs", nargs="+", default=TARGET_LANGS)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = XlingualPicoBanana(repo_id=args.repo_id, local_dir=args.local_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds.rows[: args.limit if args.limit is not None else None]:
            for lang in args.langs:
                item = {
                    "id": row["id"],
                    "lang": lang,
                    "edit_type": row.get("edit_type", ""),
                    "source_type": row.get("source_type", ""),
                    "instruction_en": row["instruction_en"],
                    "original_translation": row.get(f"instruction_{lang}", ""),
                    "translated_text_v2": "",
                    "backtranslate_to_en": "",
                    "translation_model": "",
                    "translation_source_text": row["instruction_en"],
                    "qa_method": "",
                    "qa_score": None,
                    "qa_flag": "",
                    "qa_notes": "",
                    "translation_status": "pending",
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[lang] += 1
                n += 1

    print(f"saved: {out_path}")
    print(f"jobs: {n}")
    print(f"per_lang: {dict(counts)}")


if __name__ == "__main__":
    main()
