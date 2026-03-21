#!/usr/bin/env python3
"""
build_english_classifier_dataset.py
Phase 5: Export clean English-only dataset for tri-input classifier.

Input:  metadata.jsonl, labels.jsonl (sample data)
Output: artifacts/english_classifier/{train,val,test}.jsonl + dataset_summary.json

Each output record:
  - id, source_image, edited_image, instruction_en, adherence_label, taxonomy_labels
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import random


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


def load_labels(labels_path: str) -> Dict[str, Dict]:
    """Load labels.jsonl; return dict indexed by id."""
    records = {}
    with open(labels_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records[row["id"]] = row
    return records


def build_classifier_records(
    metadata: Dict[str, Dict],
    labels: Dict[str, Dict],
    lang: str = "en",
    data_dir: str = ".",
    verify_images: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build classifier records from metadata and labels.
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
        
        # Require label exists
        if sample_id not in labels:
            print(f"[SKIP] {sample_id}: no label found")
            continue
        
        label = labels[sample_id]
        
        # Require all fields
        required_meta = ["prompt", "orig_path", "edited_path"]
        required_label = ["adherence", "error_types"]
        
        if not all(k in meta for k in required_meta):
            print(f"[SKIP] {sample_id}: missing metadata field(s)")
            continue
        
        if not all(k in label for k in required_label):
            print(f"[SKIP] {sample_id}: missing label field(s)")
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
                if not source_ok:
                    print(f"[SKIP] {sample_id}: source image missing: {source_full}")
                if not edited_ok:
                    print(f"[SKIP] {sample_id}: edited image missing: {edited_full}")
                continue
        
        # Build classifier record
        record = {
            "id": sample_id,
            "source_image": source_rel,
            "edited_image": edited_rel,
            "instruction_en": meta["prompt"],
            "adherence_label": label["adherence"],
            "taxonomy_labels": label["error_types"],
        }
        records.append(record)
    
    return records, stats


def deterministic_split(
    records: List[Dict],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """Split records deterministically by id into train/val/test."""
    random.seed(seed)
    
    # Sort by id for reproducibility
    records_sorted = sorted(records, key=lambda r: r["id"])
    
    # Shuffle (deterministic with seed)
    shuffled = records_sorted.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    
    return train, val, test


def compute_class_counts(records: List[Dict]) -> Dict[str, Any]:
    """Compute adherence and taxonomy class counts."""
    adherence_counts = Counter(r["adherence_label"] for r in records)
    
    # Taxonomy: multi-label, count each label occurrence
    taxonomy_counts = Counter()
    for r in records:
        for label in r["taxonomy_labels"]:
            taxonomy_counts[label] += 1
    
    return {
        "adherence": dict(adherence_counts),
        "taxonomy": dict(taxonomy_counts),
    }


def save_dataset(records: List[Dict], output_path: str) -> None:
    """Save records to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def save_summary(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
    output_dir: str,
) -> None:
    """Save dataset_summary.json with counts and split info."""
    summary = {
        "total_samples": len(train) + len(val) + len(test),
        "splits": {
            "train": {
                "count": len(train),
                "classes": compute_class_counts(train),
            },
            "val": {
                "count": len(val),
                "classes": compute_class_counts(val),
            },
            "test": {
                "count": len(test),
                "classes": compute_class_counts(test),
            },
        },
    }
    
    output_path = Path(output_dir) / "dataset_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Build English-only classifier dataset."
    )
    parser.add_argument(
        "--metadata",
        default="data/sample/metadata.jsonl",
        help="Path to metadata.jsonl",
    )
    parser.add_argument(
        "--labels",
        default="data/annotations/labels.jsonl",
        help="Path to labels.jsonl",
    )
    parser.add_argument(
        "--data_dir",
        default=".",
        help="Directory containing image files (default: current directory)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language filter (default: en)",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/english_classifier",
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
        "--verify",
        action="store_true",
        help="Verify that all image files exist before building dataset",
    )
    
    args = parser.parse_args()
    
    # Load data
    print("[1] Loading metadata...")
    metadata = load_metadata(args.metadata)
    print(f"    Loaded {len(metadata)} metadata records")
    
    print("[2] Loading labels...")
    labels = load_labels(args.labels)
    print(f"    Loaded {len(labels)} label records")
    
    # Build classifier records
    print(f"[3] Building classifier records (lang={args.lang})...")
    if args.verify:
        print(f"    (with image verification against {args.data_dir})")
    
    records, verify_stats = build_classifier_records(
        metadata, labels, lang=args.lang, data_dir=args.data_dir, verify_images=args.verify
    )
    print(f"    Built {len(records)} classifier records")
    
    if args.verify:
        print(f"[3b] Verification results:")
        print(f"    Total checked: {verify_stats['checked']}")
        print(f"    Source images exist: {verify_stats['source_exists']}")
        print(f"    Edited images exist: {verify_stats['edited_exists']}")
        print(f"    Both exist: {verify_stats['both_exist']}")
        print(f"    Missing source: {verify_stats['missing_source']}")
        print(f"    Missing edited: {verify_stats['missing_edited']}")
        print(f"    Missing both: {verify_stats['missing_both']}")
    
    if not records:
        print("[!] No records. Exiting.")
        return
    
    # Deterministic split
    print("[4] Splitting train/val/test...")
    train, val, test = deterministic_split(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"    train={len(train)}, val={len(val)}, test={len(test)}")
    
    # Save splits
    print("[5] Saving splits...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_dataset(train, str(output_dir / "train.jsonl"))
    print(f"    Saved: {output_dir / 'train.jsonl'}")
    
    save_dataset(val, str(output_dir / "val.jsonl"))
    print(f"    Saved: {output_dir / 'val.jsonl'}")
    
    save_dataset(test, str(output_dir / "test.jsonl"))
    print(f"    Saved: {output_dir / 'test.jsonl'}")
    
    # Save summary
    print("[6] Saving dataset summary...")
    save_summary(train, val, test, args.output_dir)
    print(f"    Saved: {output_dir / 'dataset_summary.json'}")
    
    print("\n[OK] Build complete.")


if __name__ == "__main__":
    main()
