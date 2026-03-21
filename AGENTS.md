# AGENTS.md

## Mission
This repository should become a **publishable multilingual tri-input classifier** for instruction-following image editing.

The core task is:
- input: **source image + edited image + instruction**
- output A: **adherence** = `Success | Partial | No`
- output B: **11-label error taxonomy** (multi-label)
- target behavior: work on **English + low-resource languages** with strong verification and reproducibility.

Treat this as a **research codebase** and not only as a demo repo.

## What the current repository already has
The repo already contains a working **Checkpoint 1 baseline**:
- sample dataset builder
- manual labeling tool
- CLIP embedding extraction
- logistic regression baseline trainer
- prompt/error analysis

The repo also contains **partial Checkpoint 2 code**:
- edit generation script
- VLM judge script
- VLM analysis script
- human review UI

However, Checkpoint 2 is **not fully verified** and the repo does **not yet implement the final multilingual classifier pipeline** that the project actually needs.

## Important reality check
Do **not** assume the README is fully accurate.

Known issues in the current zip snapshot:
1. `data/sample/metadata.jsonl` stores Windows-style path separators (`images\\orig\\...`), which breaks image loading on Linux/macOS.
2. The README refers to `apps/human_review.py`, but the actual file in the repo is `apps/human_review_tool.py`.
3. `config/eval_config.yaml` exists but is not wired into the scripts.
4. Cross-lingual support is mostly placeholder right now:
   - `translations.csv` is a stub in the sample data.
   - `text_encoder.py` exists but is not used by the training pipeline.
5. The human review tool writes a sidecar JSONL but does **not** push corrections back into a dataset revision or annotation store.
6. The taxonomy and terminology need one canonical source of truth.
7. `data/eval/` is empty in this zip, so Checkpoint 2 should be treated as partially implemented code, not as a verified pipeline.

## Canonical post-step-4 dataset
The project should treat the Hugging Face dataset `Legend2727/xLingual-picobanana-12k` as the canonical dataset after translation/upload.

Assume this dataset is the source of truth for the multilingual pipeline and recover missing code around it rather than re-deriving earlier steps unless necessary for verification.

## Non-negotiable design rules
1. **One canonical taxonomy.**
   Store adherence labels and error labels in one place only.
   Every script, UI, judge, training module, and report must import from that source.

2. **Keep both texts.**
   For multilingual examples, preserve:
   - original-language instruction
   - English translation
   - language code

3. **Group-aware data splits.**
   Never split translated variants of the same underlying sample across train/val/test.
   Split by a stable canonical sample id.

4. **POSIX relative paths only.**
   All metadata paths written to disk or hub must use forward slashes.
   Add a path-normalization helper and use it everywhere.

5. **Human-reviewed labels are gold.**
   Keep raw judge labels, corrected human labels, and merged labels separate.
   Never silently overwrite raw labels.

6. **Smoke tests before full runs.**
   Any pipeline stage must pass a very small deterministic smoke test before a larger run.

7. **No hidden state.**
   Every major stage must save machine-readable artifacts:
   - config snapshot
   - JSONL/CSV outputs
   - logs
   - summary stats
   - plots or tables where appropriate

8. **Paper-first discipline.**
   Every implemented stage should make the eventual paper stronger:
   reproducibility, ablations, verification, and failure analysis matter more than flashy demos.

## Stronger research framing
The strongest paper is **not**:
- “we translated prompts and trained a classifier.”

The stronger paper is:
1. a multilingual benchmark/dataset for edit-faithfulness error classification,
2. a dual-judge + human verification pipeline for scalable taxonomy labeling,
3. a compact multilingual tri-input classifier that is cheaper/faster than VLM judging,
4. analysis across language, edit type, and source quality (`sft` vs `preference_rejected`).

## Recommended pipeline direction
Prefer the following end-to-end direction.

### Stage 0 — Audit and repair the existing repo
Goals:
- make the current repo honest and runnable on Linux
- repair obvious mismatches
- avoid expensive runs until basics work

Required work:
- normalize metadata paths
- fix README/file-name mismatches
- wire config usage or remove misleading claims
- add small validation utilities
- verify Checkpoint 1 with the bundled sample data

Minimum verification:
- `python -m py_compile` over `scripts/` and `apps/`
- smoke test `train_baseline.py` with bundled sample artifacts
- smoke test `analyze_failures.py`

### Stage 1 — Recover the lost code around the uploaded HF dataset
Do **not** spend time rebuilding the full translation/upload pipeline unless necessary.
Recover only what is needed to continue from the uploaded dataset.

Add or recover:
- dataset contract validator
- loader for `Legend2727/xLingual-picobanana-12k`
- split builder (group-aware)
- translation-audit / metadata-audit utilities
- dataset statistics report

Expected outputs:
- verified dataset manifest
- split files
- audit report with counts by language, `source_type`, and `edit_type`

### Stage 2 — Dual-judge labeling for taxonomy supervision
Judge primarily in **English** using the English translation field, while preserving the original-language prompt for later evaluation.

Recommended label sources:
- Judge A: strong open-source VLM judge
- Judge B: second VLM or API judge
- Human review for disagreement, low-confidence, and stratified spot checks

Store for every example:
- raw response from each judge
- normalized adherence label
- normalized taxonomy labels
- judge confidence
- judge agreement status
- merged label decision

Prioritize review queue sampling by:
- judge disagreement
- low confidence
- rare language
- rare error type
- rare `edit_type`

### Stage 3 — Human review UI and dataset-backed corrections
The review app can be Streamlit or Gradio.
A Hugging Face Space is acceptable, but persistence must be auditable.

Do **not** directly mutate the raw dataset in place.
Prefer one of:
- a separate annotations dataset repo,
- a dedicated `annotations/` directory committed back to the hub,
- a branch/revision-specific correction log.

The review tool must support:
- showing source image, edited image, instruction, language, and source type
- showing both judge outputs and merged output
- changing adherence and taxonomy labels
- writing correction metadata with reviewer id and timestamp

### Stage 4 — Train the multilingual student classifier
The student model should be a fast classifier, not a huge VLM by default.

Recommended default architecture:
- frozen or lightly tuned **vision encoder** for source and edited images
- multilingual **text encoder** for instructions
- fusion features using source, edit, difference, abs-difference, and text
- two prediction heads:
  - adherence head (3-way)
  - taxonomy head (11-label multi-label)

Recommended losses:
- weighted cross-entropy for adherence
- BCE-with-logits or focal loss for taxonomy
- optional confidence/agreement weighting from teacher labels

Recommended training setup:
- group split by canonical sample id
- stratify by language and `source_type` where possible
- report per-language and macro metrics
- track calibration, not just accuracy/F1

### Stage 5 — Evaluation and paper artifacts
At minimum, report:
- adherence accuracy / macro-F1
- taxonomy micro-F1 / macro-F1 / mAP
- per-language performance
- performance on `sft` vs `preference_rejected`
- performance by edit type
- teacher-vs-human agreement
- student-vs-human agreement on the gold subset

Mandatory ablations:
1. original-language text only
2. English translation only
3. both original + English text
4. single-judge labels vs dual-judge labels
5. with vs without human-corrected data
6. old CLIP baseline vs improved multilingual model

## Modeling guidance
### Default student model
Prefer a **compact multimodal classifier** over full VLM fine-tuning for the main result.

A strong practical recipe is:
- image features from a better encoder than the current CLIP baseline
- multilingual text features from a multilingual encoder
- simple fusion MLP
- multi-task training

This is the best balance of:
- speed
- reproducibility
- low GPU cost
- easier paper ablations

### Optional stronger baseline
If time remains, add one stronger but costlier baseline:
- VLM classification baseline with LoRA or prompted inference

But do not let this block the main compact-model pipeline.

### Legacy editor code
`generate_edits.py` and `editor_model.py` are legacy evaluation assets from the older idea of benchmarking an editor.

They may still be useful for sanity checks or extra experiments, but **they are not the main path** for the final classifier paper.
Because Pico-Banana already provides edited outputs and bad examples, you do not need to generate new edits to make the paper viable.

## File and code quality rules
- Keep new code in a consistent structure and avoid one-off scripts that duplicate logic.
- Prefer adding small reusable helpers over repeating parsing logic.
- Every new CLI should have:
  - `--config`
  - `--limit` or a smoke-test friendly flag
  - deterministic seed support
  - machine-readable output
- Add docstrings and concise comments only where they help.
- Update README only after the code has been verified.

## Suggested new modules or scripts
Use the existing `scripts/` layout unless there is a strong reason to refactor.
Likely additions:
- `scripts/validate_xlingual_dataset.py`
- `scripts/build_group_splits.py`
- `scripts/dual_judge_label.py`
- `scripts/merge_human_reviews.py`
- `scripts/train_student.py`
- `scripts/evaluate_student.py`
- `scripts/export_paper_tables.py`
- `scripts/sync_reviews_to_hub.py`
- `apps/review_space.py` or improved `human_review_tool.py`

## Verification checklist for every serious change
Before considering a phase complete, ensure:
- code compiles
- a tiny smoke test runs
- output files are written
- logs or summaries are readable
- no path separator bugs remain
- no train/val leakage through translated duplicates

## Commands worth preserving
Small baseline checks that should continue to work:

```bash
python -m py_compile $(find scripts apps -name '*.py')
python scripts/train_baseline.py --data data/sample --labels data/annotations/labels.jsonl --out runs/baseline_smoke
python scripts/analyze_failures.py --data data/sample --labels data/annotations/labels.jsonl --out runs/baseline_smoke_analysis
```

## When asked to help
When a user asks for work in this repo:
1. First determine which stage the request belongs to.
2. Prefer the smallest change that advances the pipeline toward the paper.
3. Run a smoke test before proposing a full run.
4. Keep outputs and verification explicit.
5. If the request conflicts with the paper-first plan, explain the tradeoff and steer back to the publishable path.

## Most important instruction
The final deliverable is **not just working code**.
It is a **credible workshop-paper pipeline** with:
- a validated multilingual dataset,
- trustworthy labels,
- a strong compact classifier,
- clear verification,
- reproducible experiments,
- and analysis that reviewers will respect.
