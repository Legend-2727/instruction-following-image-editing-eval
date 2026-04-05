# Evidence-Grounded Model Path

This repo now includes the recovered source notebook and a cleaned script path for the evidence-grounded multilingual taxonomy evaluator.

## Source Notebook

Recovered notebook:

- `notebooks/evidence_grounded_taxonomy_eval_lineage_v1_to_v3.ipynb`

This notebook was imported from the earlier local/Kaggle workflow file:

- `../notebookc95272763e.ipynb`

It is now the source of truth for the recovered evidence-grounded lineage in this repo.

## Supported Protocol

The recovered code keeps the notebook protocol aligned to the public branch:

- languages: `en`, `hi`, `bn`
- single-prompt inference
- source image + edited image + one instruction
- split by original ID before multilingual expansion
- canonical taxonomy ordering from `scripts/utils/schema.py`
- taxonomy-only prediction for the recovered `v1` / `v2` / `v3` lineage

## Recovered Script Mapping

Shared utility module:

- `scripts/utils/evidence_grounded_taxonomy.py`

Versioned training path:

- `scripts/train_evidence_grounded_taxonomy.py`
  - notebook-grounded v1 training and export
- `scripts/finetune_evidence_grounded_taxonomy_v2.py`
  - notebook-grounded v2 fine-tuning from v1
- `scripts/finetune_evidence_grounded_taxonomy_v3.py`
  - notebook-grounded v3 fine-tuning from v2 or v1 fallback

Inference and evaluation:

- `scripts/infer_evidence_grounded_taxonomy.py`
- `scripts/evaluate_evidence_grounded_benchmark.py`
- `scripts/run_demo_inference_evidence_grounded.py`

## Model Structure

The recovered implementation follows the notebook architecture:

- vision backbone: `google/vit-base-patch16-224-in21k`
- text backbone: `xlm-roberta-base`
- prompt-conditioned grounding over source and target patch features
- evidence signals including:
  - `src_presence`
  - `tgt_presence`
  - `correspondence_max`
  - `correspondence_mean`
  - `local_change_score`
  - `global_change_score`
  - `outside_change_score`
  - `pred_op_type`
- taxonomy prediction over the fused evidence vector

## Artifact Layout

Recovered scripts write to the cleaned repo layout rather than the older checkpoint directories:

- v1 reports: `reports/evidence_grounded/v1/`
- v2 reports: `reports/evidence_grounded/v2/`
- v3 reports: `reports/evidence_grounded/v3/`
- v1 weights: `experiments/history/evidence_grounded_taxonomy_eval_v1/best_model.pt`
- v2 weights: `experiments/history/evidence_grounded_taxonomy_eval_v2/best_model.pt`
- v3 weights: `models/released/evidence_grounded_taxonomy_eval_v3/best_model.pt`

For compatibility with the existing local workspace, the scripts also fall back to the original root import folders when those local artifacts still exist.

## Threshold Semantics

Threshold handling matches the recovered artifact contract:

- v1 threshold file is label-keyed
- v2 and v3 threshold files are array-based in canonical taxonomy order
- the recovered scripts also emit `best_thresholds_by_label.json` for v2/v3 to make that ordering explicit

## Demo Outputs

Small inspectable demo outputs live under:

- `reports/demo_inference/`

Those files are intended for quick GitHub inspection of the v3 output format. They are artifact-backed replay outputs unless you run the demo script in live mode with local images and the local `v3` checkpoint.
