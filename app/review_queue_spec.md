# Review Queue Specification

## Overview

A **review item** is one image-edit sample shown to an annotator for quality control. The HF Space UI loads review items from a queue, displays judge predictions, accepts human corrections, and appends the result to an append-only review log.

## Review Item Structure

Each review item contains:

### Display Information (Read-Only)
```
- id: unique sample identifier (e.g., "85160_3")
- source_image: path to original image
- edited_image: path to edited image
- instruction_en: English description of the requested edit
- original_adherence_label: initial label (from dataset)
- original_taxonomy_labels: initial error types (from dataset)
```

### Judge Predictions (Read-Only, Context)
```
- judge_a_adherence: Judge A's adherence prediction
- judge_a_taxonomy: Judge A's error type list
- judge_a_confidence: Judge A's confidence (0.0-1.0)
- judge_a_raw: Judge A's reasoning text
- judge_b_adherence: Judge B's adherence prediction
- judge_b_taxonomy: Judge B's error type list
- judge_b_confidence: Judge B's confidence (0.0-1.0)
- judge_b_raw: Judge B's reasoning text
- adherence_agreement: boolean (A and B agree on adherence?)
- taxonomy_agreement: boolean (A and B agree on error types?)
- overall_agreement: boolean (both agree?)
- mean_confidence: average confidence of both judges
```

### Editable Fields (Annotator Input)
```
- final_adherence: corrected adherence label (Success|Partial|No)
- final_taxonomy: corrected error type list (array, may be empty)
- notes: optional reviewer comment or explanation
```

### Metadata Added by UI
```
- reviewer_id: HF username or assigned reviewer ID
- action_type: "corrected" | "approved" | "disputed"
- timestamp_utc: ISO 8601 timestamp when submitted
```

## Review Action Output

When a reviewer submits, the UI appends **one** review log record:

```json
{
  "sample_id": "85160_3",
  "previous_labels": {
    "adherence": "No",
    "taxonomy": ["Wrong Object"]
  },
  "updated_labels": {
    "adherence": "Partial",
    "taxonomy": ["Wrong Object", "Spatial Error"]
  },
  "reviewer_id": "reviewer_001",
  "timestamp_utc": "2026-03-19T16:30:45.123456Z",
  "action_type": "corrected",
  "notes": "Judge said 'No' but edit mostly follows instruction",
  "source": "hf-space-v1"
}
```

## Multi-User Safety

### Concurrent Editing
- **No locks needed.** Each reviewer submits independently; review logs are append-only.
- Multiple annotators can review the same sample; last update wins deterministically (by timestamp).
- Conflicts resolved during merge (see `merge_review_log.py`).

### Optimistic Locking (Optional)
- UI can fetch current `mean_confidence` and `overall_agreement` before showing the item.
- If another reviewer has already submitted corrections since this item was loaded, UI can notify: "New review submitted for this sample. Reload to see updated judge predictions."
- This is optional; append-only design works without it.

### Queue Status Tracking (Optional)
- Sample becomes "under review" when loaded by a reviewer (soft flag, no enforcement).
- Sample becomes "reviewed" when submission is appended to log.
- If two reviewers submit simultaneously, both records are logged; merge logic takes the latest by timestamp.

## Review Item Lifecycle

1. **Load**: UI loads sample (id, images, instruction, judge outputs, original labels)
2. **Display**: Show all read-only fields + judge predictions
3. **Edit**: Annotator modifies final_adherence, final_taxonomy, notes
4. **Validate**: Check that final labels are valid (adherence in enum, taxonomy in approved list)
5. **Submit**: UI appends review action to review log JSONL (does NOT modify base dataset)
6. **Confirm**: Show "Review saved" + offer next item from queue

## Queue Filtering & Sorting

Suggested priority order for the review queue:

1. **High priority** (recommended for first pass):
   - `overall_agreement == false` (judges disagree; need human decision)
   - `mean_confidence < 0.7` (low confidence; need verification)
   - Rare error types or languages (stratified sampling)

2. **Medium priority**:
   - `adherence_agreement == false` (disagreement on main label)
   - `mean_confidence 0.7-0.8` (moderate confidence)

3. **Low priority**:
   - `overall_agreement == true` (strong agreement; can spot-check)
   - `mean_confidence >= 0.8` (high confidence)

## Error Taxonomy (Canonical)

Reviewers must choose from (or approve empty):

```
- "Wrong Object"
- "Missing Object"
- "Extra Object"
- "Wrong Attribute"
- "Spatial Error"
- "Style Mismatch"
- "Over-editing"
- "Under-editing"
- "Artifact / Quality Issue"
- "Ambiguous Prompt"
- "Failed Removal"
```

## Adherence Labels (Canonical)

Reviewers must choose exactly one:

```
- "Success"
- "Partial"
- "No"
```

## Integration Points

- **Input**: Load from artifacts/english_classifier/train.jsonl + judges output
- **Output**: Append to artifacts/reviews/human_reviews.jsonl
- **Merge**: Use `merge_review_log.py` to combine with dataset
- **Next**: Feed merged dataset to student classifier training

## Example Walkthrough

**Scenario**: Reviewer sees sample 85160_3

1. UI loads from judge outputs:
   - Judge A: Partial, confidence 0.65
   - Judge B: No, confidence 0.70
   - agreement: false, overall: false

2. Reviewer reads instruction: "let the man wear a suit and tie"

3. Reviewer examines source and edited images

4. Reviewer changes labels:
   - adherence: No → Partial
   - taxonomy: [Wrong Object] → [Wrong Object, Spatial Error]
   - notes: "Judge said No but edit mostly follows instruction. Man has suit but is slightly off-center."

5. UI appends to review log:
   ```json
   {
     "sample_id": "85160_3",
     "previous_labels": {"adherence": "No", "taxonomy": ["Wrong Object"]},
     "updated_labels": {"adherence": "Partial", "taxonomy": ["Wrong Object", "Spatial Error"]},
     "reviewer_id": "reviewer_001",
     "timestamp_utc": "2026-03-19T16:30:45.123456Z",
     "action_type": "corrected",
     "notes": "Judge said No but edit mostly follows instruction. Man has suit but is slightly off-center.",
     "source": "hf-space-v1"
   }
   ```

6. Later, `merge_review_log.py` reads this log and creates merged dataset with final_adherence = "Partial".

## Notes for Space Implementation

- Do **not** write to the base dataset JSONL files.
- Do **not** overwrite previous reviews.
- **Always append** new entries to the append-only review log.
- **Validate all inputs** against the canonical taxonomy and adherence sets.
- **Track reviewer_id** (HF username or custom ID).
- **Include notes** for later error analysis.
- **Timestamp everything** in UTC ISO 8601 format.
