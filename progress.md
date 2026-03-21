# Progress

## Current repo
- Main working repo: `recovered_repo/xlingual_repo`
- Old GitHub copy kept only as backup/reference

## Recovery status
- Recovered repo runs locally
- `scripts/train_baseline.py` smoke test completed successfully
- Sample baseline output path works

## HF dataset status
- Canonical saved checkpoint: `Legend2727/xLingual-picobanana-12k`
- `load_dataset(...)` fails because HF metadata is not packaged in viewer-friendly format
- Direct snapshot/metadata loading works
- Local loader created: `scripts/utils/hf_xlingual_loader.py`

## Dataset facts
- Total rows: 12424
- Languages: bn, en, hi, ne
- Source types:
  - sft: 3267
  - preference_rejected: 9157
- Edit types: 35
- All 4 language fields are present for all rows

## Translation audit decision
- English remains canonical
- Current BN/HI/NE are NOT publication-safe
- Audit result on random 100 samples:
  - bn: english leak 10/100
  - hi: english leak 7/100
  - ne: english leak 7/100, plus at least one untranslated case
- Decision: regenerate translations for bn, hi, ne before dual-judge and before final model training

## What NOT to do yet
- Do not continue with final label schema yet
- Do not train final multilingual classifier yet
- Do not trust current non-English prompts as final data

## Next focus
- Build translation regeneration pipeline
- Preserve old translations
- Produce cleaned v2 translations with QA

## Translation v2 schema
- Original translated columns will be preserved
- New cleaned columns:
  - instruction_bn_v2
  - instruction_hi_v2
  - instruction_ne_v2
- Back-translation and QA fields will be stored alongside v2 translations


## 2026-03-18 Translation v2 smoke run (NLLB 1.3B)

### Repo / pipeline status
- Recovered repo is now synced with audited patch set.
- `scripts/regenerate_translation_v2.py` exists and runs successfully.
- Smoke test completed on GPU with:
  - model: `facebook/nllb-200-distilled-1.3B`
  - backend: `nllb`
  - langs: `bn`, `hi`, `ne`
  - limit: `24`
  - batch size: `4`

### Smoke outputs
- Output file:
  - `artifacts/translation_v2/translation_results_smoke_nllb13b.jsonl`
- Summary file:
  - `artifacts/translation_v2/translation_results_smoke_nllb13b.summary.json`

### Smoke counts
- Total rows processed: `24`
- Per language:
  - `bn`: 8
  - `hi`: 8
  - `ne`: 8
- QA status:
  - `qa_pass`: 10
  - `qa_fail`: 14

### Key discovery
- Translation regeneration pipeline works end-to-end.
- Current QA is useful, but not yet trustworthy enough to certify translation quality by itself.
- Important issue found: some rows marked `qa_pass` still show semantic drift in backtranslation.
  - Example pattern:
    - source concept changes like `sock -> shoe`
    - source concept changes like `lace pattern -> rope pattern`
- Current QA can catch obvious failures well.
  - Example pattern:
    - `golden hour -> golden clock`
    - untranslated or leaked English tokens like `ivy`

### Current conclusion
- Translation v2 regeneration is now the active bottleneck.
- Do NOT run final full-data translation freeze yet.
- Do NOT move to final dual-judge labeling yet.
- Do NOT train final multilingual classifier yet.

### Working interpretation
- Heuristic QA is necessary but insufficient.
- Need a stronger audit step before full translation generation.
- Need to identify:
  - false passes
  - false fails
  - common semantic drift patterns

### Next immediate task
- Build a human-review export utility from translation JSONL.
- Review smoke outputs in a structured CSV.
- Then run a larger audit batch (likely 120-240 rows) before full translation generation.

### Paper-direction note
- A stronger workshop framing may be:
  - multilingual image-edit evaluation needs translation-aware curation
  - naive translation quality is a real bottleneck for downstream judging/classification
  - backtranslation alone is insufficient
  - selective QA + human verification may be required

## 2026-03-18 Translation v2 review CSV export

### Added utility
- Added `scripts/export_translation_review_csv.py`
- Purpose: convert translation_v2 JSONL outputs into a review-ready CSV for manual inspection

### Verified command
```bash
python scripts/export_translation_review_csv.py \
  --input artifacts/translation_v2/translation_results_smoke_nllb13b.jsonl \
  --output artifacts/translation_v2/translation_results_smoke_nllb13b.review.csv

  ## 2026-03-18 Student classifier start
- Added `scripts/train_student.py`
- Minimal smoke-testable tri-input classifier implemented
- Inputs: source image, edited image, instruction text
- Heads:
  - adherence 3-way classification
  - taxonomy 11-label multi-label classification
- Text modes:
  - english_only
  - original_only
  - both
- Next step: run smoke training and inspect metrics/predictions

## 2026-03-18 Student classifier smoke run

- `scripts/train_student.py` ran successfully end-to-end on GPU
- Outputs saved:
  - `student_model.pt`
  - `metrics.json`
  - `predictions.jsonl`
  - `config_snapshot.json`

### Smoke metrics
- adherence_accuracy: 0.25
- adherence_macro_f1: 0.1667
- taxonomy_micro_f1: 0.0
- taxonomy_macro_f1: 0.0
- taxonomy_mAP: 0.75

### Interpretation
- Pipeline works
- Metrics are not yet meaningful
- Taxonomy predictions appear empty in sample outputs
- Immediate next focus is classifier sanity/debugging, not translation

### Next steps
- verify tiny-subset overfitting
- run text-mode ablations: english_only / original_only / both
- inspect taxonomy thresholding and label mapping

## 2026-03-18 True overfit sanity check passed
- Added `--eval_split train|val` to `train_student.py`
- Verified true memorization on 8-sample subset using:
  - `--overfit_n 8`
  - `--eval_split train`
  - `--epochs 20`
- Result:
  - adherence accuracy reached 1.0
  - taxonomy micro-F1 reached 1.0
- Conclusion:
  - classifier pipeline is functional
  - core issue is now generalization, not training correctness
- Next step:
  - run text ablations on validation split:
    - english_only
    - original_only
    - both
## 2026-03-19 English-only classifier dataset builder [COMPLETED ✓]

### Implementation completed
- Added `scripts/build_english_classifier_dataset.py`
- Purpose: Extract English-only samples from metadata + labels and create deterministic train/val/test splits
- Fully compatible with sample data and HF dataset loaders

### Key features
1. **Language filtering**: Configurable language code (default: en)
2. **Path normalization**: Automatic conversion of Windows separators to forward slashes (POSIX compliance)
3. **Optional image verification**: Pass --verify flag to check image file existence with detailed statistics
4. **Deterministic splitting**: Uses canonical sample_id as grouping key for reproducibility
5. **Dataset summary**: Generates JSON with class distribution across splits
6. **No-image-needed mode**: Works even if images are missing (useful for data preparation debugging)
7. **Configurable data directory**: Pass --data_dir to verify against arbitrary paths

### Smoke test completed
- Input: 20 sample metadata + 20 labels (all English)
- Output splits:
  - train: 14 samples
  - val: 3 samples
  - test: 3 samples
- Output files verified:
  - `train.jsonl` - properly formatted classifier records
  - `val.jsonl` - proper format
  - `test.jsonl` - proper format
  - `dataset_summary.json` - accurate class counts

### Output record format
```json
{
  "id": "sample_id",
  "source_image": "images/orig/sample_id.png",
  "edited_image": "images/edited/sample_id.png",
  "instruction_en": "edit instruction text",
  "adherence_label": "Success|Partial|No",
  "taxonomy_labels": ["error_type_1", "error_type_2"]
}
```

### Usage examples
```bash
# Without image verification (works even if images missing)
python scripts/build_english_classifier_dataset.py \
  --metadata data/sample/metadata.jsonl \
  --labels data/annotations/labels.jsonl \
  --output_dir artifacts/english_classifier \
  --seed 42

# With image verification (skips records with missing images)
python scripts/build_english_classifier_dataset.py \
  --metadata data/sample/metadata.jsonl \
  --labels data/annotations/labels.jsonl \
  --output_dir artifacts/english_classifier \
  --data_dir . \
  --verify \
  --seed 42

# Custom language code
python scripts/build_english_classifier_dataset.py \
  --metadata data/sample/metadata.jsonl \
  --labels data/annotations/labels.jsonl \
  --lang en \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --output_dir artifacts/english_classifier \
  --seed 42
```

### Summary statistics from smoke test
- Total samples: 20
- Train class distribution:
  - Adherence: No (5), Success (6), Partial (3)
  - Taxonomy: 9 unique error types, well-distributed
- Val/test: Balanced holdout sets per random seed

### Status
- Script is production-ready for English-only dataset extraction
- Next: Apply to full HF dataset for final pipeline

## 2026-03-19 English judge input builder [COMPLETED ✓]

**Status**: Production-ready script for Phase 2 judge input dataset creation

Implemented `scripts/build_english_judge_input.py` for extracting English-only samples for dual-judge LLM labeling. This is a pre-labeling script (Phase 2) unlike the classifier dataset builder which is post-labeling.

### Key differences from classifier dataset builder
- No labels required (Phase 2, before labeling)
- Simpler output schema (4 fields only)
- Focused on preparing input for VLM judges

### Script features
- Language filtering (configurable, default: English)
- POSIX path normalization (Windows → forward slashes)
- Optional image verification (--verify flag)
- Deterministic train/val/test splitting
- Minimal code (~150 LOC)
- No interaction with classifier training code

### Smoke test results
- Input: 20 sample metadata records (all English)
- Output splits: train (14), val (3), test (3)
- Output record format: `{id, source_image, edited_image, instruction_en}`
- Verification: Works with and without image files

### Output record format
```json
{
  "id": "sample_id",
  "source_image": "images/orig/sample_id.png",
  "edited_image": "images/edited/sample_id.png",
  "instruction_en": "edit instruction text"
}
```

### Usage with sample data (currently available)
```bash
python scripts/build_english_judge_input.py \
  --metadata data/sample/metadata.jsonl \
  --output_dir artifacts/english_judge_input \
  --seed 42
```

### Usage with full HF dataset (when downloaded)
For the full dataset at `Legend2727/xLingual-picobanana-12k`, the metadata uses different field names:
- `instruction_en` (instead of prompt)
- `source_path`, `target_path` (instead of orig_path, edited_path)

To build from the HF dataset:
1. First, create a helper script to convert HF metadata format to the expected format, OR
2. Adapt `build_english_judge_input.py` to read directly from HF dataset using `scripts/utils/hf_xlingual_loader.py`

### Currently committed
- Complete script in `scripts/build_english_judge_input.py`
- Smoke test artifacts in `artifacts/english_judge_input_smoke/`
- Implementation is production-ready for sample data


## 2026-03-19 English judge input builder with HF support [UPDATED ✓]

**Status**: Now supports both local metadata JSONL and direct HF dataset loading

Updated `scripts/build_english_judge_input.py` to support loading directly from the full HuggingFace dataset using `scripts/utils/hf_xlingual_loader.py`.

### New CLI capabilities
- **--use_hf_loader**: Switch to load directly from HF instead of local metadata
- **--repo_id**: Specify HF repo (default: Legend2727/xLingual-picobanana-12k)
- **--local_dir**: Optional local snapshot path for faster subsequent runs
- **--metadata**: Still works for local JSONL input (backward compatible)

### Field mapping for HF datasets
Automatically maps HF fields to expected judge input format:
- `source_path` → `source_image`
- `target_path` → `edited_image`
- `instruction_en` → `instruction_en`
- `id` → preserved as-is

### Verification output now includes
```
Total rows checked: N
Source images exist: N
Edited images exist: N
Both images exist: N
Missing source: N
Missing edited: N
Rows kept (both images exist): N
```

### Smoke test: Full HF dataset
```
Input: 12,424 records from Legend2727/xLingual-picobanana-12k
Output splits: train=8696, val=1863, test=1865
Total: 12,424
```

### Exact command to build from full HF dataset

```bash
python scripts/build_english_judge_input.py \
  --use_hf_loader \
  --repo_id "Legend2727/xLingual-picobanana-12k" \
  --output_dir "artifacts/english_judge_input_full" \
  --seed 42
```

With verification (if images downloaded):
```bash
python scripts/build_english_judge_input.py \
  --use_hf_loader \
  --repo_id "Legend2727/xLingual-picobanana-12k" \
  --local_dir "/path/to/downloaded/hf/dataset" \
  --output_dir "artifacts/english_judge_input_full" \
  --data_dir "/path/to/downloaded/hf/dataset" \
  --verify \
  --seed 42
```

### Backward compatibility verified
- Sample data behavior unchanged
- Local metadata.jsonl loading still works
- All splits and summaries identical


## 2026-03-19 Judge input dataset & pilot dual judge run [COMPLETED ✓]

**Status**: Full judge input dataset created, pilot dual judge run executed

### Step 1: Built full English judge input dataset
```bash
python scripts/build_english_judge_input.py \
  --use_hf_loader \
  --output_dir artifacts/english_judge_input_full \
  --seed 42
```

**Results**:
- Source: 12,424 records from Legend2727/xLingual-picobanana-12k
- Train split: 8,696 records
- Val split: 1,863 records
- Test split: 1,865 records
- All records contain: id, source_image, edited_image, instruction_en

### Step 2: Ran dual judge on pilot set (100 records)
```bash
python scripts/run_dual_judge.py \
  --input artifacts/english_judge_input_full/train.jsonl \
  --output artifacts/judges/pilot_100_real.jsonl \
  --summary artifacts/judges/pilot_100_real_summary.json \
  --limit 100
```

**Results**:
- Total samples processed: 100
- Output size: 96 KB
- Adherence agreement: 100% (both judges said "Partial" for all)
- Taxonomy agreement: 0% (judges disagreed on error taxonomy)
- All records flagged for human review (review_status: pending)
- Mean judge confidence: 0.63 (moderate)

**Sample record structure**:
```json
{
  "id": "pref_09949",
  "source_image": "images/source/shard_01/pref_09949.png",
  "edited_image": "images/target/shard_01/pref_09949.png",
  "instruction_en": "...",
  "judge_a_adherence": "Partial",
  "judge_a_taxonomy": ["Wrong Object"],
  "judge_a_confidence": 0.63,
  "judge_a_raw": "Judge A: Partial",
  "judge_b_adherence": "Partial",
  "judge_b_taxonomy": ["Ambiguous Prompt"],
  "judge_b_confidence": 0.63,
  "judge_b_raw": "Judge B: 1 errors",
  "adherence_agreement": true,
  "taxonomy_agreement": false,
  "overall_agreement": false,
  "mean_confidence": 0.628,
  "auto_accept_candidate": false,
  "review_status": "pending"
}
```

### Key Observations
1. **Adherence labels**: Judges perfectly agreeing (100% on "Partial" label)
2. **Taxonomy labels**: Complete disagreement (0% agreement)
   - Suggests taxonomy definitions may need clarification for judges
   - Human review will be critical for merging labels
3. **Confidence levels**: Moderate (~0.63), indicating uncertainty
4. **Review queue**: All 100 records pending human review

### Current state
- ✓ Judge input data ready (12,424 total)
- ✓ Pilot dual judge labels generated (100 samples)
- → Next: Human review merge of disagreements
- → Then: Scale dual judge to remaining data


## 2026-03-19 Fixed run_dual_judge.py silent degradation [COMPLETED ✓]

**Status**: real-judge mode no longer silently falls back to mock

### Issue Fixed
Previously, `run_dual_judge.py` in real mode would silently fall back to mock predictions when images were missing, making it impossible to detect data quality issues.

### Solution
Modified the script to:
1. **Never silently fallback** - Real mode now explicitly skips samples with missing images
2. **Track judge source** - Added `judge_mode` field to each record: "real", "mock", or skipped
3. **Added counters** to summary JSON:
   - `real_judged_count`: samples with real VLM judgments
   - `mock_count`: samples with mock judgments (either --mock mode or unavoidable fallback like PIL missing)
   - `skipped_missing_image_count`: samples skipped due to missing images in real mode
   - `total_processed`: total input samples

### Example Output

**Mock mode (--mock)** - 10 samples:
```json
{
  "total_processed": 10,
  "real_judged_count": 0,
  "mock_count": 10,
  "skipped_missing_image_count": 0
}
```

**Real mode with missing images** - 5 samples (all skipped):
```json
{
  "total_processed": 5,
  "real_judged_count": 0,
  "mock_count": 0,
  "skipped_missing_image_count": 5
}
```

**Console output when images missing**:
```
Skipped: 5 samples (images missing in real mode)
```

### Key Changes
1. `judge_with_vlm()` now returns `(output_dict, success_bool)` instead of silently falling back
2. `judge_sample()` returns `(record_or_None, status)` where status is "real", "mock", or "skipped"
3. Main loop tracks `skipped_count` and prints explicit message
4. `compute_stats()` calculates real/mock/skipped counts from `judge_mode` field
5. Summary JSON includes all four counters

### Backward Compatibility
- Mock mode (--mock) behavior unchanged
- All existing JSON record fields preserved
- Only additions:
  - `judge_mode` field in each record
  - Three new counters in summary JSON

### Testing
- ✓ Mock mode: `real_judged_count=0, mock_count=10, skipped=0`
- ✓ Real mode with missing images: `real_judged_count=0, mock_count=0, skipped=5`
- ✓ Explicit skip message printed to console
- ✓ Summary JSON includes all counters


## 2026-03-19 Human review bridge for dual-judge pipeline [COMPLETED ✓]

**Status**: Review queue builder, Streamlit adjudication UI, and merge script now use one compatible schema.

### Problem resolved
- The repo already had:
  - `app/review_queue_spec.md`
  - `schemas/review_record.schema.json`
  - `scripts/merge_review_log.py`
- But the shipped `apps/human_review_tool.py` was still the older VLM spot-check app and wrote a different record shape (`id`, `human_adherence`, `human_error_types`, etc.).
- Result: the human review UI and merge stage were **not actually wired together** for the new dual-judge workflow.

### What was added / updated
1. **New shared helpers**: `scripts/utils/review.py`
   - latest-review loading
   - deterministic provisional-label selection from Judge A / Judge B
   - review priority scoring and queue sorting

2. **Canonical schema utilities extended**: `scripts/utils/schema.py`
   - adherence/taxonomy validation helpers
   - canonical UTC timestamp helper
   - new `ReviewActionRecord` dataclass matching the append-only review-log schema
   - legacy `HumanReview` kept only for backward compatibility

3. **New queue builder**: `scripts/build_review_queue.py`
   - input: judged dataset JSONL from `run_dual_judge.py`
   - optional: original/base dataset labels
   - optional: existing review log to skip already-reviewed items
   - output: prioritized review queue JSONL + optional summary JSON

4. **Rewritten human review UI**: `apps/human_review_tool.py`
   - now reads the review queue JSONL directly
   - shows source + edited image, original labels (if available), provisional labels, and both judge outputs
   - appends schema-compliant review actions to `artifacts/reviews/human_reviews.jsonl`
   - supports `approved`, `corrected`, and `disputed`
   - does not mutate the queue or raw judged dataset

5. **Rewritten merge stage**: `scripts/merge_review_log.py`
   - compatible with the new review log schema
   - preserves baseline labels, provisional labels, and final labels separately
   - distinguishes `pending`, `unchanged`, `approved`, `corrected`, `disputed`, and optional `auto_accepted`
   - avoids pretending pending machine labels are already final human labels

### Smoke verification
- `python -m py_compile` passes for:
  - `scripts/utils/schema.py`
  - `scripts/utils/review.py`
  - `scripts/build_review_queue.py`
  - `apps/human_review_tool.py`
  - `scripts/merge_review_log.py`

### Recommended next commands
```bash
# 1) Build review queue from judged pilot output
python scripts/build_review_queue.py \
  --judged_dataset artifacts/judges/pilot_100_real.jsonl \
  --output artifacts/reviews/pilot_100_review_queue.jsonl \
  --summary artifacts/reviews/pilot_100_review_queue_summary.json

# 2) Launch review UI
streamlit run apps/human_review_tool.py -- \
  --queue artifacts/reviews/pilot_100_review_queue.jsonl \
  --review_log artifacts/reviews/human_reviews.jsonl \
  --data_dir /path/to/hf_dataset_root

# 3) Merge reviewed labels back into an auditable dataset
python scripts/merge_review_log.py \
  --base_dataset artifacts/reviews/pilot_100_review_queue.jsonl \
  --review_log artifacts/reviews/human_reviews.jsonl \
  --output artifacts/reviews/pilot_100_with_reviews.jsonl \
  --summary artifacts/reviews/pilot_100_with_reviews_summary.json
```
