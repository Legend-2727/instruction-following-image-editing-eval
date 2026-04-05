# Evidence-Grounded Multilingual Taxonomy Evaluation for Instruction-Following Image Editing

This repository packages a multilingual ML project for evaluating instruction-following image edits from a tri-input prediction task:

- input: source image + edited image + instruction
- output A: adherence = `Success | Partial | No`
- output B: 11-label error taxonomy

The broader repo goal remains adherence plus taxonomy, but the recovered evidence-grounded `v1` / `v2` / `v3` lineage that is preserved here is the taxonomy-focused branch and is the current best release path.

The current public branch reflects the later English/Hindi/Bangla workstream, not the older editor-demo README that originally shipped with this repo snapshot. It keeps:

- the earlier lightweight baseline as a real baseline
- multilingual dataset/split utilities and review tooling
- open-VLM benchmarking utilities and supporting workflow code
- evidence-grounded artifact bundles for `v1`, `v2`, and `v3`
- the recovered evidence-grounded lineage notebook and reusable wrapper scripts

`evidence_grounded_taxonomy_eval_v3` is the current best overall model artifact in this repository and is the only release-designated model bundle.

## Project Scope

This project evaluates whether an edited image follows a text instruction, with a multilingual taxonomy-focused framing.

Supported languages in this public branch:

- English
- Hindi
- Bangla

Current inference framing:

- multilingual single-prompt inference
- one prompt language per prediction
- image pair + prompt used jointly for taxonomy prediction in the recovered evidence-grounded path
- broader repo framing still retains adherence as part of the overall evaluation task

## Taxonomy

Canonical adherence labels:

- `Success`
- `Partial`
- `No`

Canonical taxonomy labels:

- `Wrong Object`
- `Missing Object`
- `Extra Object`
- `Wrong Attribute`
- `Spatial Error`
- `Style Mismatch`
- `Over-editing`
- `Under-editing`
- `Artifact / Quality Issue`
- `Ambiguous Prompt`
- `Failed Removal`

The single source of truth for these labels is [scripts/utils/schema.py](scripts/utils/schema.py).

## Data Protocol

The public branch uses original-level splitting first and language expansion second.

1. Build splits by original sample ID.
2. Keep translated variants of the same original in the same split.
3. Expand each original into language-specific training/evaluation rows at runtime.
4. Run multilingual single-prompt inference by selecting one prompt language for a source/edited pair.

This avoids leakage across language variants and matches the training/evaluation utilities already present in the repo.

## What Is Included

- Checkpoint 1 lightweight baseline code:
  - sample dataset builder
  - manual labeling UI
  - CLIP feature extraction
  - logistic-regression baseline trainer
  - failure analysis
- Multilingual dataset-building and split utilities for the later branch
- Human review queue and merge tooling for auditable label correction
- Open-VLM benchmarking utilities and prompt files from the later workflow
- Recovered evidence-grounded notebook source:
  - `notebooks/evidence_grounded_taxonomy_eval_lineage_v1_to_v3.ipynb`
- Recovered evidence-grounded scripts:
  - `scripts/train_evidence_grounded_taxonomy.py`
  - `scripts/finetune_evidence_grounded_taxonomy_v2.py`
  - `scripts/finetune_evidence_grounded_taxonomy_v3.py`
  - `scripts/infer_evidence_grounded_taxonomy.py`
  - `scripts/evaluate_evidence_grounded_benchmark.py`
  - `scripts/run_demo_inference_evidence_grounded.py`
- Organized evidence-grounded report bundles for `v1`, `v2`, and `v3`
- Final public model bundle: `evidence_grounded_taxonomy_eval_v3`
- Preferred model host: `https://huggingface.co/Legend2727/evidence_grounded_taxonomy_eval_v3`

## Primary Operational Path

The notebook is preserved as lineage/source history. The primary operational path for this repo is now the cleaned script interface:

- `scripts/train_evidence_grounded_taxonomy.py`
- `scripts/finetune_evidence_grounded_taxonomy_v2.py`
- `scripts/finetune_evidence_grounded_taxonomy_v3.py`
- `scripts/infer_evidence_grounded_taxonomy.py`
- `scripts/evaluate_evidence_grounded_benchmark.py`
- `scripts/run_demo_inference_evidence_grounded.py`

The recovered evidence-grounded path emits taxonomy predictions, matching the tracked `v1` / `v2` / `v3` artifacts and manifests under `reports/evidence_grounded/`.

## Recovery Status

The evidence-grounded lineage notebook has now been imported into the repo and promoted into reusable scripts. The recovered code path covers:

- `v1` training
- `v2` fine-tuning
- `v3` fine-tuning
- live inference wrappers
- benchmark evaluation wrappers
- a small demo-output path

What is still not claimed:

- full end-to-end retraining verification inside this repo
- packaged image data inside git
- fully offline runtime without the required model dependencies/backbone availability

## Model Progression

Evidence-grounded progression preserved in this repo:

| Version | Role | Benchmark micro-F1 | Benchmark macro-F1 supported | Full-test micro-F1 | Full-test macro-F1 supported | Status |
|---|---|---:|---:|---:|---:|---|
| `v1` | first evidence-grounded artifact bundle | 0.6557 | 0.3421 | 0.4326 | 0.2337 | history |
| `v2` | improved evidence-grounded artifact bundle | 0.6652 | 0.3565 | 0.4659 | 0.2666 | history |
| `v3` | current best overall artifact bundle | 0.6652 | 0.3555 | 0.4835 | 0.2813 | release |

Why `v3` is the release candidate:

- it is the strongest overall public artifact in this branch
- it improves the full-test metrics over `v1` and `v2`
- it is the intended final model artifact for this repo snapshot

`v1` and `v2` remain in the repo as experiment history, not as competing public releases.

## Baselines and Benchmarking

The earlier lightweight baseline still matters and remains part of the public project story. It provides:

- a small, reproducible tri-input baseline
- a lower-cost reference point against the later evidence-grounded system
- a transparent starting point for outsiders reading the repo

The repo also preserves the open-VLM benchmarking path and the human-review utilities that support it:

- [scripts/vlm_judge.py](scripts/vlm_judge.py)
- [scripts/run_dual_judge.py](scripts/run_dual_judge.py)
- [scripts/analyze_vlm_results.py](scripts/analyze_vlm_results.py)
- [apps/human_review_tool.py](apps/human_review_tool.py)

Those components are kept because benchmark-against-VLM work is part of the completed branch history, even though the repo’s main public release story is now the evidence-grounded taxonomy evaluator.

## Repository Layout

```text
.
├── docs/                                  # public project docs and migration notes
├── reports/
│   ├── evidence_grounded/
│   │   ├── v1/                            # evidence-grounded history bundle
│   │   ├── v2/                            # evidence-grounded history bundle
│   │   └── v3/                            # current best report bundle
│   └── demo_inference/                    # small inspectable v3 demo outputs
├── models/
│   └── released/
│       └── evidence_grounded_taxonomy_eval_v3/
│                                           # release placeholder; large weights not tracked in git
├── experiments/
│   └── history/                           # history placeholders for older evidence-grounded weights
├── notebooks/
│   └── evidence_grounded_taxonomy_eval_lineage_v1_to_v3.ipynb
│                                           # recovered source notebook from the local/Kaggle workflow
├── apps/
│   ├── label_tool.py
│   └── human_review_tool.py
├── scripts/
│   ├── train_baseline.py
│   ├── analyze_failures.py
│   ├── build_final_multilingual_taxonomy_dataset.py
│   ├── split_final_taxonomy_dataset.py
│   ├── train_stage1_multilingual_binary.py
│   ├── train_stage2_multilingual_taxonomy.py
│   ├── train_stage2_multilingual_taxonomy_heavy.py
│   ├── train_stage2_taxonomy_only_heavy.py
│   ├── train_evidence_grounded_taxonomy.py
│   ├── finetune_evidence_grounded_taxonomy_v2.py
│   ├── finetune_evidence_grounded_taxonomy_v3.py
│   ├── infer_evidence_grounded_taxonomy.py
│   ├── evaluate_evidence_grounded_benchmark.py
│   ├── run_demo_inference_evidence_grounded.py
│   ├── run_dual_judge.py
│   ├── vlm_judge.py
│   └── utils/
└── data/
    └── final/
        └── splits/                        # original-level split artifacts
```

## Setup

Create a Python environment and install the repo dependencies:

```bash
python -m venv .venv
# Git Bash
source .venv/Scripts/activate
# PowerShell
# .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU-backed work, install the appropriate PyTorch build for your CUDA setup before or after `requirements.txt` as needed.

## Quick Demo

Artifact-backed demo replay writes a small set of inspectable JSON outputs without requiring the image corpus or live checkpoint execution:

```bash
python scripts/run_demo_inference_evidence_grounded.py --mode artifact
```

The checked-in sample outputs currently live under `reports/demo_inference/`:

- `reports/demo_inference/demo_predictions.jsonl`
- `reports/demo_inference/pref_02339_en.json`
- `reports/demo_inference/pref_02339_hi.json`
- `reports/demo_inference/pref_02339_bn.json`

Those checked-in samples are replayed from the released `v3` benchmark predictions and are meant for format inspection on GitHub. This cleanup environment did not have a local image root available, so the sample files are artifact-backed rather than fresh live inference runs.

If you have local images and the local `v3` checkpoint in `models/released/evidence_grounded_taxonomy_eval_v3/best_model.pt`, you can also run live inference:

```bash
python scripts/infer_evidence_grounded_taxonomy.py \
  --image-root /path/to/image_root \
  --source-path images/source/shard_00/example.png \
  --target-path images/target/shard_00/example.png \
  --instruction "Remove the object from the left side of the image." \
  --lang en
```

## Weights and Artifact Policy

This repo now distinguishes between tracked report artifacts and untracked large model binaries.

Current release status:

- release-designated model: `evidence_grounded_taxonomy_eval_v3`
- local release bundle prepared under `models/released/evidence_grounded_taxonomy_eval_v3/`
- published model host: `https://huggingface.co/Legend2727/evidence_grounded_taxonomy_eval_v3`
- local bundle remains the canonical upload source for future release refreshes

Tracked in git:

- manifests
- metrics
- thresholds
- label maps
- per-class benchmark tables
- training histories
- public-facing docs

Optional local-only imports when present:

- raw prediction exports from the original root evidence-grounded folders
- auxiliary evidence CSV exports from the original root evidence-grounded folders

Not tracked in git:

- large `.pt` model checkpoints

Expected local location for the final release weight:

- `models/released/evidence_grounded_taxonomy_eval_v3/best_model.pt`

See:

- [docs/model_release_policy.md](docs/model_release_policy.md)
- [docs/model_download_and_usage.md](docs/model_download_and_usage.md)
- [docs/artifact_contracts.md](docs/artifact_contracts.md)
- [models/released/evidence_grounded_taxonomy_eval_v3/README.md](models/released/evidence_grounded_taxonomy_eval_v3/README.md)

## Legacy and Historical Components

Not everything in this repo is part of the current public release story.

Historical/legacy items are still kept because they are useful reference material:

- the older editor-generation and VLM-judge evaluation path
- migration notes from the recovery process
- older experiment outputs that are no longer the main release path

See [docs/legacy_components.md](docs/legacy_components.md) for a precise breakdown.

## Additional Docs

- [docs/evidence_grounded_model.md](docs/evidence_grounded_model.md)
- [docs/evidence_grounded_repro.md](docs/evidence_grounded_repro.md)
- [docs/evidence_grounded_recovery_gaps.md](docs/evidence_grounded_recovery_gaps.md)
- [docs/project_status.md](docs/project_status.md)
- [docs/model_release_policy.md](docs/model_release_policy.md)
- [docs/model_download_and_usage.md](docs/model_download_and_usage.md)
- [docs/artifact_contracts.md](docs/artifact_contracts.md)
- [docs/legacy_components.md](docs/legacy_components.md)
- [docs/repo_migration_status.md](docs/repo_migration_status.md)
- [docs/lfs_cleanup_instructions.md](docs/lfs_cleanup_instructions.md)
- [docs/github_lfs_remote_cleanup.md](docs/github_lfs_remote_cleanup.md)
