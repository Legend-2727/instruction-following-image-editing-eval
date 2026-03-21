#!/usr/bin/env python3
"""
run_dual_judge.py

Run dual-judge (Judge A and Judge B) on English classifier dataset.
Produces candidate labels for human review.

Usage (dry-run with mock predictions):
  python scripts/run_dual_judge.py \
    --input artifacts/english_classifier/train.jsonl \
    --output artifacts/judges/train_with_judges.jsonl \
    --mock

Usage (real VLM judges - requires Qwen2.5-VL and images):
  python scripts/run_dual_judge.py \
    --input artifacts/english_classifier/train.jsonl \
    --output artifacts/judges/train_with_judges.jsonl \
    --data_dir . \
    --judge_a qwen_adherence \
    --judge_b qwen_taxonomy
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ── Mock judges for dry-run / testing ────────────────────────────────────────
def mock_judge_a(instruction: str, seed: int = 0) -> Dict[str, Any]:
    """Mock Judge A: adherence + taxonomy predictions."""
    random.seed(seed)
    adherence = random.choice(["Success", "Partial", "No"])
    
    # Judge A also predicts taxonomy (0-2 errors)
    all_errors = [
        "Wrong Object", "Missing Object", "Extra Object", "Wrong Attribute",
        "Spatial Error", "Style Mismatch", "Over-editing", "Under-editing",
        "Artifact / Quality Issue", "Ambiguous Prompt", "Failed Removal",
    ]
    num_errors = random.randint(0, 2)
    taxonomy = random.sample(all_errors, num_errors)
    
    confidence = random.uniform(0.5, 1.0)
    return {
        "adherence": adherence,
        "taxonomy": taxonomy,
        "confidence": confidence,
        "reasoning": f"Judge A: {adherence}",
    }


def mock_judge_b(instruction: str, seed: int = 1) -> Dict[str, Any]:
    """Mock Judge B: adherence + taxonomy predictions."""
    random.seed(seed)
    
    # Judge B predicts taxonomy (0-3 errors)
    all_errors = [
        "Wrong Object", "Missing Object", "Extra Object", "Wrong Attribute",
        "Spatial Error", "Style Mismatch", "Over-editing", "Under-editing",
        "Artifact / Quality Issue", "Ambiguous Prompt", "Failed Removal",
    ]
    num_errors = random.randint(0, 3)
    taxonomy = random.sample(all_errors, num_errors)
    
    # Infer adherence from taxonomy: no errors -> Success, else -> Partial/No
    if not taxonomy:
        adherence = "Success"
    else:
        adherence = random.choice(["Partial", "No"])
    
    confidence = random.uniform(0.5, 1.0)
    return {
        "adherence": adherence,
        "taxonomy": taxonomy,
        "confidence": confidence,
        "reasoning": f"Judge B: {len(taxonomy)} errors",
    }


# ── Judges using VLM (placeholder; would use vlm_evaluator.py) ──────────────
# Global cache for VLM judge instance (loaded once, reused for all samples)
_vlm_judge_cache = {}


def judge_with_vlm(
    judge_name: str,
    data_dir: str,
    source_image_path: str,
    edited_image_path: str,
    instruction: str,
    allow_missing: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    Real VLM judge using QwenVLMJudge.
    Returns (judgment_dict, status) where status is one of:
    - 'real': successful VLM judgment
    - 'mock': fallback to mock (PIL missing, import error, etc.)
    - 'parse_failed': JSON parsing failed in real mode
    - 'image_error': image loading failed
    
    judgment_dict includes parse_failed=True/False field.
    """
    if not HAS_PIL:
        logger.warning("[!] PIL not available; falling back to mock")
        judgment = mock_judge_a(instruction) if judge_name == "qwen_a" else mock_judge_b(instruction)
        judgment["parse_failed"] = False
        return judgment, "mock"
    
    try:
        from utils.vlm_evaluator import QwenVLMJudge
    except ImportError:
        logger.warning("[!] vlm_evaluator not available; falling back to mock")
        judgment = mock_judge_a(instruction) if judge_name == "qwen_a" else mock_judge_b(instruction)
        judgment["parse_failed"] = False
        return judgment, "mock"
    
    # Determine prompt type based on judge name
    prompt_type = "failure_focused" if "b" in judge_name.lower() else "conservative"
    
    # Initialize judge once, reuse for all samples
    judge_id = f"judge_{judge_name}"
    if judge_id not in _vlm_judge_cache:
        logger.info(f"[*] Initializing {judge_name} (prompt_type={prompt_type})...")
        _vlm_judge_cache[judge_id] = QwenVLMJudge(
            model_id="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cuda",
            prompt_type=prompt_type,
        )
    
    judge = _vlm_judge_cache[judge_id]
    
    # Load images
    try:
        src_img_full = Path(data_dir) / source_image_path
        edit_img_full = Path(data_dir) / edited_image_path
        
        if not src_img_full.exists() or not edit_img_full.exists():
            logger.warning(f"[!] Images not found: {source_image_path} or {edited_image_path}")
            if allow_missing:
                judgment = mock_judge_a(instruction) if judge_name == "qwen_a" else mock_judge_b(instruction)
                judgment["parse_failed"] = False
                return judgment, "mock"
            else:
                return {}, "image_error"
        
        orig_img = Image.open(src_img_full)
        edit_img = Image.open(edit_img_full)
    except Exception as e:
        logger.warning(f"[!] Error loading images: {e}")
        if allow_missing:
            judgment = mock_judge_a(instruction) if judge_name == "qwen_a" else mock_judge_b(instruction)
            judgment["parse_failed"] = False
            return judgment, "mock"
        else:
            return {}, "image_error"
    
    # Call VLM judge
    try:
        vlm_output = judge.evaluate(orig_img, edit_img, instruction)
        
        # Check if parsing failed
        if vlm_output.get("parse_failed", False):
            logger.warning(f"[!] Parse failed for {judge_name} on sample")
            return vlm_output, "parse_failed"
        
        # Normalize output
        return {
            "adherence": vlm_output.get("adherence"),
            "taxonomy": vlm_output.get("error_types", []),
            "confidence": vlm_output.get("confidence", 0.5),
            "reasoning": vlm_output.get("raw_response", "")[:100],
            "parse_failed": False,
        }, "real"
    except Exception as e:
        logger.exception(f"[!] VLM evaluation exception for {judge_name}: {e}")
        if allow_missing:
            judgment = mock_judge_a(instruction) if judge_name == "qwen_a" else mock_judge_b(instruction)
            judgment["parse_failed"] = False
            return judgment, "mock"
        else:
            return {"parse_failed": True, "adherence": None, "taxonomy": [], "confidence": None}, "runtime_error"


# ── Agreement computation ─────────────────────────────────────────────────────
def compute_taxonomy_agreement(tax_a: List[str], tax_b: List[str]) -> bool:
    """Taxonomy agreement: do A and B predict the same error types?"""
    return set(tax_a) == set(tax_b)


def judge_sample(
    sample: Dict[str, Any],
    mock: bool = False,
    data_dir: str = ".",
    judge_a_type: str = "qwen_a",
    judge_b_type: str = "qwen_b",
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Run Judge A and Judge B on a sample.
    Returns (output_record, status) where status is one of:
    - 'real': both judges succeeded with real VLM
    - 'mock': mock judges used
    - 'skipped': image missing or image_error occurred
    - 'partial_parse_failed': one judge parse-failed; record still created with nulls for failed judge
    - 'both_parse_failed': both judges parse-failed
    - 'runtime_error': other exception occurred
    """
    sample_id = sample["id"]
    instruction = sample["instruction_en"]
    
    if mock:
        # Deterministic mock predictions for testing
        judge_a_out = mock_judge_a(instruction, seed=hash(sample_id) % 1000)
        judge_b_out = mock_judge_b(instruction, seed=(hash(sample_id) + 1) % 1000)
        judge_a_out["parse_failed"] = False
        judge_b_out["parse_failed"] = False
        judge_a_status = "mock"
        judge_b_status = "mock"
        overall_status = "mock"
    else:
        # Real VLM judges
        judge_a_out, judge_a_status = judge_with_vlm(
            judge_a_type,
            data_dir,
            sample["source_image"],
            sample["edited_image"],
            instruction,
            allow_missing=False,
        )
        judge_b_out, judge_b_status = judge_with_vlm(
            judge_b_type,
            data_dir,
            sample["source_image"],
            sample["edited_image"],
            instruction,
            allow_missing=False,
        )
        
        # If either judge had image_error or parse_failed that should skip
        if judge_a_status == "image_error" or judge_b_status == "image_error":
            return None, "skipped"
        
        # Check for parse failures — still create record but with nulls
        a_parse_failed = judge_a_status == "parse_failed"
        b_parse_failed = judge_b_status == "parse_failed"
        
        if a_parse_failed or b_parse_failed:
            # Still continue to build record with available data
            if a_parse_failed and b_parse_failed:
                overall_status = "both_parse_failed"
            else:
                overall_status = "partial_parse_failed"
        else:
            overall_status = "real"
    
    # Extract predictions from each judge, handling None/missing values
    judge_a_parse_failed = judge_a_out.get("parse_failed", False)
    judge_a_adherence = judge_a_out.get("adherence") if not judge_a_parse_failed else None
    judge_a_taxonomy = judge_a_out.get("taxonomy", []) if not judge_a_parse_failed else []
    judge_a_confidence = judge_a_out.get("confidence") if not judge_a_parse_failed else None
    
    judge_b_parse_failed = judge_b_out.get("parse_failed", False)
    judge_b_adherence = judge_b_out.get("adherence") if not judge_b_parse_failed else None
    judge_b_taxonomy = judge_b_out.get("taxonomy", []) if not judge_b_parse_failed else []
    judge_b_confidence = judge_b_out.get("confidence") if not judge_b_parse_failed else None
    
    # Compute agreement metrics (only when both parsed successfully)
    if not judge_a_parse_failed and not judge_b_parse_failed:
        adherence_agreement = (judge_a_adherence == judge_b_adherence)
        taxonomy_agreement = compute_taxonomy_agreement(judge_a_taxonomy, judge_b_taxonomy)
        overall_agreement = adherence_agreement and taxonomy_agreement
        mean_confidence = (judge_a_confidence + judge_b_confidence) / 2.0 if judge_a_confidence and judge_b_confidence else None
    else:
        # If parse failed, no agreement/confidence
        adherence_agreement = None
        taxonomy_agreement = None
        overall_agreement = False
        mean_confidence = None
    
    # Auto-accept only if both judges succeeded and agree
    auto_accept = (
        overall_agreement 
        and mean_confidence is not None 
        and mean_confidence >= 0.8
    )
    
    # Build output record
    output = dict(sample)  # Keep original fields
    output.update({
        "judge_a_adherence": judge_a_adherence,
        "judge_a_taxonomy": judge_a_taxonomy,
        "judge_a_confidence": float(judge_a_confidence) if judge_a_confidence is not None else None,
        "judge_a_parse_failed": judge_a_parse_failed,
        "judge_a_raw": str(judge_a_out.get("reasoning", "")),
        "judge_b_adherence": judge_b_adherence,
        "judge_b_taxonomy": judge_b_taxonomy,
        "judge_b_confidence": float(judge_b_confidence) if judge_b_confidence is not None else None,
        "judge_b_parse_failed": judge_b_parse_failed,
        "judge_b_raw": str(judge_b_out.get("reasoning", "")),
        "adherence_agreement": adherence_agreement,
        "taxonomy_agreement": taxonomy_agreement,
        "overall_agreement": overall_agreement,
        "mean_confidence": float(mean_confidence) if mean_confidence is not None else None,
        "auto_accept_candidate": auto_accept,
        "review_status": "pending",
        "judge_mode": overall_status,  # Track: 'real', 'mock', 'partial_parse_failed', 'both_parse_failed', etc.
    })
    
    return output, overall_status



# ── Dataset and statistics ──────────────────────────────────────────────────
def save_judged_dataset(
    records: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Save judged records to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def save_summary_stats(
    stats: Dict[str, Any],
    output_path: str,
) -> None:
    """Save summary statistics to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)



def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load English classifier dataset JSONL."""
    records = []
    with open(dataset_path) as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def compute_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute detailed summary statistics with separate error tracking."""
    total = len(records)
    adherence_agreement_count = sum(1 for r in records if r.get("adherence_agreement") is True)
    taxonomy_agreement_count = sum(1 for r in records if r.get("taxonomy_agreement") is True)
    overall_agreement_count = sum(1 for r in records if r.get("overall_agreement") is True)
    auto_accept = sum(1 for r in records if r.get("auto_accept_candidate"))
    
    # Count judge modes and failure types
    real_judged = sum(1 for r in records if r.get("judge_mode") == "real")
    mock_judged = sum(1 for r in records if r.get("judge_mode") == "mock")
    partial_parse_failed = sum(1 for r in records if r.get("judge_mode") == "partial_parse_failed")
    both_parse_failed = sum(1 for r in records if r.get("judge_mode") == "both_parse_failed")
    runtime_error = sum(1 for r in records if r.get("judge_mode") == "runtime_error")
    
    # Count parse failures at judge level
    judge_a_parse_failed = sum(1 for r in records if r.get("judge_a_parse_failed", False))
    judge_b_parse_failed = sum(1 for r in records if r.get("judge_b_parse_failed", False))
    
    judge_a_adherence_counts = {}
    for r in records:
        adh = r.get("judge_a_adherence")
        if adh is not None:  # Only count successful parses
            judge_a_adherence_counts[adh] = judge_a_adherence_counts.get(adh, 0) + 1
    
    return {
        "total_samples": total,
        "real_judged_count": real_judged,
        "mock_count": mock_judged,
        "partial_parse_failed_count": partial_parse_failed,
        "both_parse_failed_count": both_parse_failed,
        "runtime_error_count": runtime_error,
        "judge_a_parse_failed_count": judge_a_parse_failed,
        "judge_b_parse_failed_count": judge_b_parse_failed,
        "adherence_agreement_count": adherence_agreement_count,
        "adherence_agreement_rate": f"{100.0 * adherence_agreement_count / real_judged:.1f}%" if real_judged > 0 else "N/A",
        "taxonomy_agreement_count": taxonomy_agreement_count,
        "taxonomy_agreement_rate": f"{100.0 * taxonomy_agreement_count / real_judged:.1f}%" if real_judged > 0 else "N/A",
        "overall_agreement_count": overall_agreement_count,
        "overall_agreement_rate": f"{100.0 * overall_agreement_count / real_judged:.1f}%" if real_judged > 0 else "N/A",
        "auto_accept_candidates": auto_accept,
        "auto_accept_rate": f"{100.0 * auto_accept / real_judged:.1f}%" if real_judged > 0 else "N/A",
        "judge_a_adherence_distribution": judge_a_adherence_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run dual-judge pipeline on English classifier dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input dataset JSONL (from build_english_classifier_dataset.py)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL with judge predictions",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Optional: save summary stats to JSON (e.g., artifacts/judges/summary.json)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock (deterministic random) judges for dry-run",
    )
    parser.add_argument(
        "--data_dir",
        default=".",
        help="Directory containing images (for real judges)",
    )
    parser.add_argument(
        "--judge_a",
        default="qwen_a",
        help="Judge A type (default: qwen_a)",
    )
    parser.add_argument(
        "--judge_b",
        default="qwen_b",
        help="Judge B type (default: qwen_b)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N samples (0 = no limit)",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"[1] Loading dataset from {args.input}...")
    records = load_dataset(args.input)
    print(f"    Loaded {len(records)} records")
    
    if args.limit > 0:
        records = records[: args.limit]
        print(f"    Limited to {len(records)} records")
    
    # Run judges
    print(f"[2] Running dual-judge pipeline (mock={args.mock})...")
    judged = []
    skipped_count = 0
    for i, sample in enumerate(records, 1):
        if i % max(1, len(records) // 10) == 0 or i == 1:
            print(f"    {i}/{len(records)}...")
        
        output, status = judge_sample(
            sample,
            mock=args.mock,
            data_dir=args.data_dir,
            judge_a_type=args.judge_a,
            judge_b_type=args.judge_b,
        )
        
        if output is None:
            # Skipped due to image error or missing images
            skipped_count += 1
        else:
            # Include successful judgments and parse failures
            judged.append(output)
    
    if skipped_count > 0:
        print(f"    Skipped: {skipped_count} samples (images missing in real mode)")
    
    # Save
    print(f"[3] Saving judged dataset...")
    save_judged_dataset(judged, args.output)
    print(f"    Saved: {args.output}")
    
    # Stats
    print(f"[4] Summary statistics:")
    stats = compute_stats(judged)
    stats["skipped_missing_image_count"] = skipped_count
    stats["total_processed"] = len(records)
    
    for key, val in stats.items():
        print(f"    {key}: {val}")
    
    # Save summary if requested
    if args.summary:
        print(f"[5] Saving summary statistics...")
        save_summary_stats(stats, args.summary)
        print(f"    Saved: {args.summary}")
    
    print(f"\n[OK] Dual-judge pipeline complete.")


if __name__ == "__main__":
    main()
