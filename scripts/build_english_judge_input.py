#!/usr/bin/env python3
"""
Build English-only judge input dataset for dual-judge LLM labeling.

This script extracts English samples from metadata for labeling by VLM judges.
No labels required at this stage (Phase 2, before labeling).

Supports both:
- Local metadata JSONL files
- Direct HuggingFace dataset loading via hf_xlingual_loader

Output format per row:
{
  "id": "sample_id",
  "source_image": "images/orig/sample_id.png",
  "edited_image": "images/edited/sample_id.png",
  "instruction_en": "edit instruction text"
}
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
import sys

# Add scripts dir to path for utils imports
sys.path.insert(0, str(Path(__file__).parent))


def normalize_path(raw_path: str) -> str:
    """Convert Windows or mixed separators to forward slashes."""
    return raw_path.replace("\\", "/")


def load_metadata(metadata_path: str) -> Dict[str, Dict]:
    """Load metadata.jsonl; return dict indexed by id."""
    records = {}
    with open(metadata_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records[row["id"]] = row
    return records


def load_from_hf_dataset(
    repo_id: str = "Legend2727/xLingual-picobanana-12k",
    local_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Load metadata from HuggingFace dataset.
    Returns dict indexed by id in the expected metadata format.
    """
    try:
        from utils.hf_xlingual_loader import XlingualPicoBanana
    except ImportError:
        print("[ERROR] Failed to import hf_xlingual_loader. Check utils/ directory.")
        sys.exit(1)
    
    print(f"[*] Loading HF dataset: {repo_id}")
    if local_dir:
        print(f"[*] Using local directory: {local_dir}")
    
    dataset = XlingualPicoBanana(repo_id=repo_id, local_dir=local_dir)
    
    # Convert HF format to expected format
    records = {}
    for row in dataset:
        # Map HF fields to expected format
        sample_id = row.get("id")
        if not sample_id:
            continue
        
        # Normalize paths to forward slashes
        source_path = normalize_path(row.get("source_path", ""))
        target_path = normalize_path(row.get("target_path", ""))
        instruction_en = row.get("instruction_en", "")
        
        records[sample_id] = {
            "id": sample_id,
            "orig_path": source_path,
            "edited_path": target_path,
            "prompt": instruction_en,
            "lang": "en",
        }
    
    return records


def build_judge_input_records(
    metadata: Dict[str, Dict],
    lang: str = "en",
    data_dir: str = ".",
    verify_images: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build judge input records from metadata.
    Keep only samples with all required fields and matching lang.
    
    Returns: (records, verification_stats)
    """
    records = []
    stats = {
        "checked": 0,
        "source_exists": 0,
        "edited_exists": 0,
        "both_exist": 0,
        "missing_source": 0,
        "missing_edited": 0,
        "missing_both": 0,
    }
    
    for sample_id, meta in metadata.items():
        # Filter by language
        if meta.get("lang") != lang:
            continue
        
        # Require all fields
        required_fields = ["prompt", "orig_path", "edited_path"]
        if not all(k in meta for k in required_fields):
            continue
        
        # Normalize paths to forward slashes
        source_rel = normalize_path(meta["orig_path"])
        edited_rel = normalize_path(meta["edited_path"])
        
        # Verify images exist if requested
        if verify_images:
            stats["checked"] += 1
            source_full = Path(data_dir) / source_rel
            edited_full = Path(data_dir) / edited_rel
            
            source_ok = source_full.exists()
            edited_ok = edited_full.exists()
            
            if source_ok:
                stats["source_exists"] += 1
            else:
                stats["missing_source"] += 1
            
            if edited_ok:
                stats["edited_exists"] += 1
            else:
                stats["missing_edited"] += 1
            
            if source_ok and edited_ok:
                stats["both_exist"] += 1
            else:
                # Skip this sample if images missing
                continue
        
        # Build judge input record
        record = {
            "id": sample_id,
            "source_image": source_rel,
            "edited_image": edited_rel,
            "instruction_en": meta["prompt"],
        }
        records.append(record)
    
    return records, stats


def deterministic_split(
    records: List[Dict],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """Split records deterministically into train/val/test."""
    random.seed(seed)
    shuffled = sorted(records, key=lambda x: x["id"])  # Stable sort first
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]
    
    return train, val, test


def save_dataset(records: List[Dict], output_path: str) -> None:
    """Save records as JSONL."""
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def save_summary(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
    output_dir: str,
) -> None:
    """Save dataset summary statistics."""
    summary = {
        "total_samples": len(train) + len(val) + len(test),
        "splits": {
            "train": {
                "count": len(train),
            },
            "val": {
                "count": len(val),
            },
            "test": {
                "count": len(test),
            },
        },
    }
    
    with open(Path(output_dir) / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Build English-only judge input dataset for dual-judge LLM labeling."
    )
    
    # Input source options
    input_group = parser.add_argument_group("Input source (choose one)")
    input_group.add_argument(
        "--metadata",
        default=None,
        help="Path to local metadata.jsonl (default: data/sample/metadata.jsonl if --use_hf_loader not set)",
    )
    input_group.add_argument(
        "--use_hf_loader",
        action="store_true",
        help="Load directly from HuggingFace dataset instead of local metadata",
    )
    input_group.add_argument(
        "--repo_id",
        default="Legend2727/xLingual-picobanana-12k",
        help="HuggingFace repo ID (used with --use_hf_loader, default: Legend2727/xLingual-picobanana-12k)",
    )
    input_group.add_argument(
        "--local_dir",
        default=None,
        help="Local directory with HF snapshot (used with --use_hf_loader)",
    )
    
    # Processing options
    parser.add_argument(
        "--lang",
        default="en",
        help="Language filter (default: en)",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/english_judge_input",
        help="Output directory for train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Val split ratio (default 0.15)",
    )
    parser.add_argument(
        "--data_dir",
        default=".",
        help="Directory containing image files (for verification only)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that all image files exist before building dataset",
    )
    
    args = parser.parse_args()
    
    # Determine input source
    if args.use_hf_loader:
        print("[1] Loading from HuggingFace dataset...")
        metadata = load_from_hf_dataset(repo_id=args.repo_id, local_dir=args.local_dir)
        print(f"    Loaded {len(metadata)} records from HF dataset")
    else:
        # Use local metadata
        metadata_path = args.metadata or "data/sample/metadata.jsonl"
        print("[1] Loading local metadata...")
        metadata = load_metadata(metadata_path)
        print(f"    Loaded {len(metadata)} metadata records")
    
    if not metadata:
        print("[!] No metadata records. Exiting.")
        return
    
    # Build judge input records
    print(f"[2] Building judge input records (lang={args.lang})...")
    if args.verify:
        print(f"    (with image verification against {args.data_dir})")
    
    records, verify_stats = build_judge_input_records(
        metadata, lang=args.lang, data_dir=args.data_dir, verify_images=args.verify
    )
    print(f"    Built {len(records)} judge input records")
    
    if args.verify:
        print(f"[2b] Verification results:")
        print(f"    Total rows checked: {verify_stats['checked']}")
        print(f"    Source images exist: {verify_stats['source_exists']}")
        print(f"    Edited images exist: {verify_stats['edited_exists']}")
        print(f"    Both images exist: {verify_stats['both_exist']}")
        print(f"    Missing source: {verify_stats['missing_source']}")
        print(f"    Missing edited: {verify_stats['missing_edited']}")
        broken = verify_stats['missing_source'] + verify_stats['missing_edited'] - verify_stats.get('missing_both', 0)
        print(f"    Rows kept (both images exist): {verify_stats['both_exist']}")
    
    if not records:
        print("[!] No records passed filters. Exiting.")
        return
    
    # Deterministic split
    print("[3] Splitting train/val/test...")
    train, val, test = deterministic_split(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"    train={len(train)}, val={len(val)}, test={len(test)}")
    
    # Save splits
    print("[4] Saving splits...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_dataset(train, str(output_dir / "train.jsonl"))
    print(f"    Saved: {output_dir / 'train.jsonl'}")
    
    save_dataset(val, str(output_dir / "val.jsonl"))
    print(f"    Saved: {output_dir / 'val.jsonl'}")
    
    save_dataset(test, str(output_dir / "test.jsonl"))
    print(f"    Saved: {output_dir / 'test.jsonl'}")
    
    # Save summary
    print("[5] Saving dataset summary...")
    save_summary(train, val, test, args.output_dir)
    print(f"    Saved: {output_dir / 'dataset_summary.json'}")
    
    print("\n[OK] Build complete.")


if __name__ == "__main__":
    main()
