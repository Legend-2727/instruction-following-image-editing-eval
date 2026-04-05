# Project Status

## Public Branch Summary

This repository now reflects the later completed public branch centered on:

- English, Hindi, and Bangla support
- multilingual single-prompt inference
- broader adherence-plus-taxonomy project framing
- recovered evidence-grounded `v1` / `v2` / `v3` lineage focused on the 11-label taxonomy path
- a retained lightweight baseline
- open-VLM benchmarking workflow support
- evidence-grounded artifact progression through `v1`, `v2`, and `v3`

`evidence_grounded_taxonomy_eval_v3` is the current best overall model artifact in this repo.

## What Is In Scope

- baseline code and failure analysis
- multilingual dataset/split utilities already present in the repo
- human review queue and merge tooling
- organized report bundles for the evidence-grounded variants
- recovered evidence-grounded notebook and wrapper scripts
- release metadata for the final public model bundle

## What Is Still Limited

The evidence-grounded source notebook is now present and the main v1/v2/v3 code path has been recovered into reusable scripts.

What is still not claimed:

- a fully re-run, end-to-end retraining verification in this cleaned repo
- packaged image data inside git
- a fully offline runtime bundle with all pretrained backbone dependencies vendored

## Current Best Model

Release-designated model:

- `evidence_grounded_taxonomy_eval_v3`

History-only experiment bundles:

- `evidence_grounded_taxonomy_eval` (`v1`)
- `evidence_grounded_taxonomy_eval_v2` (`v2`)

## Metrics Snapshot

Best overall public artifact metrics:

- `v3` benchmark micro-F1: `0.6652`
- `v3` benchmark macro-F1 supported: `0.3555`
- `v3` full-test micro-F1: `0.4835`
- `v3` full-test macro-F1 supported: `0.2813`

These numbers come from the recovered local artifact bundle and are preserved in the organized report directories under `reports/evidence_grounded/`.

## Practical Reading Order

1. Read [README.md](../README.md).
2. Inspect `reports/evidence_grounded/v3/`.
3. Read [docs/model_release_policy.md](model_release_policy.md).
4. Read [docs/artifact_contracts.md](artifact_contracts.md).
5. Use the recovered evidence-grounded scripts for training, evaluation, and inference.

## Honest Limitation

The repo now includes the recovered notebook and first-class wrapper scripts, but the cleaned script path has not yet been validated by a fresh full retraining run in this environment.
