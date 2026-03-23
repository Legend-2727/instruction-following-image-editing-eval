#!/usr/bin/env python
"""
Prepare a subset of preference_rejected samples for pure human taxonomy labeling.
Leaves the remainder (and all sft samples) available for stage-1 or stage-2 training.

Example Usage:
python scripts/prepare_human_taxonomy_pool.py \
    --metadata data/hf_snapshots/xlingual_picobanana_multilingual_6k/metadata.jsonl \
    --out_dir data/splits \
    --pool_size 1200 \
    --seed 42
"""

import argparse
import json
import logging
from pathlib import Path
from collections import Counter
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def get_stratified_sample(items, target_size, stratify_key, seed):
    # Group items by the stratify key
    grouped = {}
    for item in items:
        k = item.get(stratify_key, "unknown")
        if k not in grouped:
            grouped[k] = []
        grouped[k].append(item)
    
    # Shuffle each group
    rng = random.Random(seed)
    for k in grouped:
        rng.shuffle(grouped[k])
        
    sampled = []
    total_items = len(items)
    
    # Calculate target counts per group
    targets = {}
    for k, group_items in grouped.items():
        proportion = len(group_items) / total_items
        targets[k] = int(target_size * proportion)
        
    # Adjust to meet exact target_size if there's rounding error
    current_total = sum(targets.values())
    keys = list(targets.keys())
    while current_total < target_size and keys:
        targets[rng.choice(keys)] += 1
        current_total += 1
        
    # Draw samples
    for k in targets:
        k_target = targets[k]
        sampled.extend(grouped[k][:k_target])
        # If we asked for more than available, just take all available
        
    # If we still have slightly fewer due to some groups being smaller than target, fill remainder from remaining items
    remaining = []
    for k in grouped:
        remaining.extend(grouped[k][targets.get(k, 0):])
    
    if len(sampled) < target_size:
        rng.shuffle(remaining)
        sampled.extend(remaining[:target_size - len(sampled)])
        
    return sampled

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pool_size", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metadata, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Filter rejected samples
    rejected = [row for row in data if row.get("source_type") == "preference_rejected"]
    others = [row for row in data if row.get("source_type") != "preference_rejected"]

    logger.info(f"Total samples: {len(data)}")
    logger.info(f"Rejected (pool candidates): {len(rejected)}")
    logger.info(f"SFT/Other: {len(others)}")

    actual_target = min(args.pool_size, len(rejected))
    reserved = get_stratified_sample(rejected, actual_target, "edit_type", args.seed)
    
    reserved_ids = {r["id"] for r in reserved}
    
    # Calculate remainder
    remaining_rejected = [r for r in rejected if r["id"] not in reserved_ids]
    
    stage1_train_data = others + remaining_rejected
    
    # Summary info
    reserved_edit_types = Counter(r.get("edit_type", "unknown") for r in reserved)
    
    summary = {
        "total_samples": len(data),
        "total_rejected": len(rejected),
        "reserved_count": len(reserved),
        "remaining_count": len(stage1_train_data),
        "reserved_by_edit_type": dict(reserved_edit_types)
    }

    # Save
    with open(out_dir / "human_taxonomy_pool.jsonl", "w", encoding="utf-8") as f:
        for r in reserved:
            f.write(json.dumps(r) + "\n")
            
    with open(out_dir / "human_taxonomy_pool_ids.json", "w", encoding="utf-8") as f:
        json.dump(list(reserved_ids), f, indent=2)
        
    with open(out_dir / "stage1_train_excluded_ids.json", "w", encoding="utf-8") as f:
        json.dump(list(reserved_ids), f, indent=2)
        
    with open(out_dir / "stage1_train_metadata.jsonl", "w", encoding="utf-8") as f:
        for r in stage1_train_data:
            f.write(json.dumps(r) + "\n")
            
    with open(out_dir / "pool_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    logger.info("Saved pool files. Summary:")
    logger.info(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
