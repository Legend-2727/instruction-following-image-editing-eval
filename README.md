# xLingual Instruction-Following Image Editing Evaluation

This repository is being developed into a publishable multilingual evaluation and modeling pipeline for instruction-following image editing.

## Goal

Build a compact, reproducible tri-input classifier that predicts:

- Adherence: `Success | Partial | No`
- Error taxonomy: 11-label multi-label output

from:

- Source image
- Edited image
- Instruction text (original language + English translation)

Target outcome:

- Strong multilingual behavior (English + low-resource languages)
- Verified labels (dual judge + human corrections)
- Reproducible experiments and paper-ready analysis

## Current Status (March 2026)

Project maturity by stage:

- Stage 0 (audit + repair): mostly complete
- Stage 1 (HF dataset recovery + validation): in progress, core pieces available
- Stage 2 (dual-judge labeling): partially implemented
- Stage 3 (human review + correction merge): partially implemented
- Stage 4 (student classifier training): implemented and smoke-tested
- Stage 5 (full evaluation + paper tables): not complete

High-level reality check:

- Legacy Checkpoint 2 editor/judge flow exists, but should not be treated as fully verified final pipeline.
- Canonical dataset source is Hugging Face: `Legend2727/xLingual-picobanana-12k`.
- Translation quality for non-English prompts is currently a major bottleneck.

## What Has Been Completed

### A. Baseline / legacy pipeline (Checkpoint 1)

Implemented and runnable:

- Sample dataset builder
- Manual labeling tool
- CLIP embedding extraction
- Logistic regression baseline trainer
- Failure analysis utilities

Smoke checks have been run successfully for baseline scripts.

### B. Repo audit and recovery work

Completed items include:

- Linux/path compatibility fixes (forward-slash path handling added in key scripts)
- README/script mismatch repair in codebase (review app filename aligned to `human_review_tool.py` in current code)
- Recovery of missing utilities and modular helpers
- Addition of pipeline scripts for multilingual direction and student training

### C. HF dataset recovery and facts

Recovered/validated facts from current workflow:

- Total rows: `12424`
- Languages present: `bn`, `en`, `hi`, `ne`
- Source types:
  - `sft`: `3267`
  - `preference_rejected`: `9157`
- Edit types: `35`

Loader support exists in:

- `scripts/utils/hf_xlingual_loader.py`

### D. Translation v2 regeneration groundwork

Implemented:

- Translation regeneration script
- Translation review CSV export script
- Smoke outputs and summaries saved in `artifacts/translation_v2/`

Current conclusion:

- Regeneration pipeline works end-to-end.
- QA heuristics help, but semantic drift still appears in some `qa_pass` cases.
- Non-English translations are not yet publication-safe without stronger review.

### E. Student classifier implementation

Implemented:

- `scripts/train_student.py` with tri-input setup
- Dual-head prediction:
  - 3-way adherence
  - 11-label taxonomy
- Text modes:
  - `english_only`
  - `original_only`
  - `both`

Verification status:

- End-to-end smoke run completed.
- True overfit sanity check on tiny subset passed (memorization reached expected near-perfect behavior on train subset).
- Interpretation: pipeline is functional; current challenge is generalization quality.

### F. English-only dataset builder

Implemented:

- `scripts/build_english_classifier_dataset.py`

Features include:

- Deterministic train/val/test split
- Group-safe sample handling with canonical id usage
- Optional image verification
- Path normalization for cross-platform compatibility
- Machine-readable summary outputs

## Forward Tasks (Prioritized)

### Priority 1: Finalize translation quality gate

- Run larger translation audit batches (beyond smoke)
- Quantify false-pass / false-fail QA patterns
- Add stronger semantic checks before freezing v2 translations
- Lock cleaned columns:
  - `instruction_bn_v2`
  - `instruction_hi_v2`
  - `instruction_ne_v2`

### Priority 2: Complete dual-judge labeling pipeline

- Finalize judge A + judge B normalization outputs
- Persist raw outputs, normalized labels, confidence, agreement, and merged labels
- Ensure disagreement/low-confidence routing to review queue

### Priority 3: Human review and correction merge

- Strengthen review app workflow
- Capture reviewer id + timestamp + rationale
- Keep raw judge labels separate from corrected labels
- Merge corrected labels into auditable annotation artifacts (no silent overwrite)

### Priority 4: Train multilingual student model on verified labels

- Train on group-aware splits by canonical sample id
- Evaluate language-wise and source-type-wise slices
- Compare text configurations:
  - original only
  - English only
  - both

### Priority 5: Evaluation and paper artifacts

Required outputs:

- Adherence accuracy and macro-F1
- Taxonomy micro-F1, macro-F1, and mAP
- Per-language performance tables
- Performance by `source_type` and `edit_type`
- Teacher-human and student-human agreement summaries
- Ablation results and reproducible export tables

## Non-Negotiable Rules for Ongoing Work

1. One canonical taxonomy source used by scripts, UI, and reports.
2. Always preserve original instruction, English translation, and language code.
3. Never leak translated variants across train/val/test (group-aware split only).
4. Write metadata paths as POSIX relative paths.
5. Keep raw judge labels, human corrections, and merged labels as separate artifacts.
6. Every stage must pass a deterministic smoke test before scale-up.
7. Every stage must produce explicit artifacts: config snapshot, machine-readable outputs, logs, and summaries.

## Important Files and Entry Points

Core scripts:

- `scripts/train_baseline.py`
- `scripts/analyze_failures.py`
- `scripts/regenerate_translation_v2.py`
- `scripts/export_translation_review_csv.py`
- `scripts/build_english_classifier_dataset.py`
- `scripts/run_dual_judge.py`
- `scripts/train_student.py`

Utility modules:

- `scripts/utils/hf_xlingual_loader.py`
- `scripts/utils/schema.py`
- `scripts/utils/review.py`

Review tools:

- `apps/human_review_tool.py`
- `apps/label_tool.py`

Project planning references:

- `AGENTS.md`
- `progress.md`
- `TRAIN_STUDENT_SANITY_CHECK_GUIDE.md`

## Minimal Verification Commands

Run from repo root:

```bash
python -m py_compile $(find scripts apps -name '*.py')
python scripts/train_baseline.py --data data/sample --labels data/annotations/labels.jsonl --out runs/baseline_smoke
python scripts/analyze_failures.py --data data/sample --labels data/annotations/labels.jsonl --out runs/baseline_smoke_analysis
```

## Recommended Next Execution Order

1. Run translation v2 larger audit batch and review CSV.
2. Freeze translation v2 only after quality gate passes.
3. Run full dual-judge labeling with agreement metadata.
4. Complete human correction merge pipeline.
5. Train multilingual student on verified labels.
6. Export paper tables and ablations.

## Notes

- This repo should be treated as a research codebase first, not a demo-only project.
- The strongest paper story is reliable multilingual benchmark + trustworthy labeling + compact student classifier + rigorous analysis.
