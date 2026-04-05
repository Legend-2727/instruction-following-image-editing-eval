# Artifact Contracts

## Purpose

This document explains the lightweight public artifact contract used by this repository after the cleanup of the current tree.

The main principle is:

- keep report/result metadata in git
- keep large model weights out of normal git history

## Evidence-Grounded Report Bundle

Each organized evidence-grounded report directory under `reports/evidence_grounded/<version>/` may contain:

- `artifact_manifest.json`
  - human-readable machine metadata for the bundle
- `metrics.json`
  - primary metric summary recovered from the local artifact folder
- `label_map.json`
  - taxonomy index-to-label mapping
- `best_thresholds.json`
  - original threshold artifact
- `best_thresholds_by_label.json`
  - optional derived convenience file when the original thresholds were stored as an unlabeled array
- `dataset_summary.json`
  - optional dataset summary when it existed in the recovered artifact folder
- `per_class_test_benchmark.csv`
  - per-class benchmark metrics
- `predictions_test_benchmark.jsonl`
  - optional prediction export for the benchmark split when it was migrated into git
- `train_history*.json`
  - epoch-level training history
- `evidence_*.csv`
  - optional evidence feature previews/benchmark exports where present

Some recovered bundles only track the compact summary/report files in git and keep additional raw exports in the ignored root import folders referenced by `artifact_manifest.json`.

## Release Bundle Contract

Release-model metadata lives under:

- `models/released/evidence_grounded_taxonomy_eval_v3/`

That directory is intentionally lightweight in git and should contain:

- `README.md`
  - explains the expected local weight location and usage assumptions

Optional local-only file:

- `best_model.pt`
  - not tracked in git
  - expected local weight path for the public release model

## History Bundle Contract

History-only model placeholders live under:

- `experiments/history/evidence_grounded_taxonomy_eval_v1/`
- `experiments/history/evidence_grounded_taxonomy_eval_v2/`

These are not public releases. They exist to preserve the existence of older variants and document where local weights would belong if present.

## Canonical Taxonomy Ordering

Whenever a threshold array or unlabeled vector artifact needs ordering, this repository uses the canonical label order from `scripts/utils/schema.py`:

1. `Wrong Object`
2. `Missing Object`
3. `Extra Object`
4. `Wrong Attribute`
5. `Spatial Error`
6. `Style Mismatch`
7. `Over-editing`
8. `Under-editing`
9. `Artifact / Quality Issue`
10. `Ambiguous Prompt`
11. `Failed Removal`

This ordering is used to derive convenience metadata such as `label_map.json` and `best_thresholds_by_label.json` for recovered bundles that were not fully self-describing.
