# Evidence-Grounded Recovery Gaps

The main notebook source has been recovered, but a few limitations still remain.

## What Was Recovered Safely

- the evidence-grounded lineage notebook
- the v1 training loop
- the v2 fine-tuning loop
- the v3 fine-tuning loop
- the live inference wrapper
- the benchmark evaluation wrapper
- a demo-output path

## Remaining Gaps

- The recovered script path has not been validated by a fresh end-to-end retraining run in this sandbox.
- The repo still does not include the image corpus, so live execution needs an external image root.
- The sandbox used for this recovery does not include the full ML runtime stack needed to execute training or live inference here.
- The original v2/v3 artifact folders did not contain complete self-description, so the repo still relies on derived `label_map.json` and `best_thresholds_by_label.json` files for clarity.
- The release checkpoint is still intentionally excluded from normal git history.

## Demo Output Caveat

The checked-in demo result files are based on replaying released v3 benchmark predictions against the cleaned test split metadata.

That makes them faithful to the released artifact outputs, but they are not a fresh live inference run from this sandboxed environment.
