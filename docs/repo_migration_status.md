# Repo Migration Status

## Goal Of This Cleanup

This cleanup now aligns the repository with the later completed public branch and includes the recovered evidence-grounded source notebook plus reusable wrapper scripts.

The intended public story is now:

- English, Hindi, Bangla multilingual evaluation
- earlier lightweight baseline retained
- open-VLM benchmarking retained
- evidence-grounded `v1` / `v2` / `v3` progression preserved
- `v3` marked as the current best overall model artifact

## What Changed In The Public Packaging

- the README was rewritten around the actual later branch
- public docs were added for status, release policy, artifact contracts, legacy scope, and LFS cleanup
- evidence-grounded artifacts are organized into `reports/`, `models/`, and `experiments/`
- old current-tree LFS-tracked checkpoint artifacts are removed from the repo tree
- the evidence-grounded lineage notebook is now stored under `notebooks/`
- notebook-grounded v1/v2/v3 wrapper scripts are now part of `scripts/`

## What Was Not Done

- no git history rewrite
- no expansion of the public story to Nepali or a different dataset contract
- no invented architecture beyond what could be grounded in the recovered notebook

## Current Honest State

The repo is now cleaner and more publishable, and the main missing-source gap has been closed.

What still remains:

- the recovered code has not been re-run end-to-end in this sandboxed environment
- the repo still does not ship the image corpus or large checkpoints in normal git
- the demo sample outputs in git are artifact-backed inspection files, not fresh live inference runs from this sandbox
