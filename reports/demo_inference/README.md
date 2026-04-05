# Demo Inference Samples

This directory contains small, inspectable `v3` demo outputs.

Included sample files:

- `demo_predictions.jsonl`
- `pref_02339_en.json`
- `pref_02339_hi.json`
- `pref_02339_bn.json`

These checked-in samples were materialized from the released `v3` benchmark prediction artifact using the cleaned test split metadata. They are artifact-backed replay files, not fresh live inference runs.

The cleanup environment did not have a local image root available, so no honest live sample inference run was added in this step.

The demo script is:

- `scripts/run_demo_inference_evidence_grounded.py`

It supports:

- `--mode artifact`
  - replay tracked metadata plus released prediction artifacts
- `--mode live`
  - run live inference if you have a local image root and the local `v3` checkpoint

Output fields in the sample JSON files include:

- `id`
- `lang`
- `instruction`
- `source_path`
- `target_path`
- `gold_labels`
- `pred_labels`
- `probabilities`
- `thresholds`
- `demo_mode`
