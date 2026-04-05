# Model Download And Usage

## Current Status

The release-designated model for this repository is:

- `evidence_grounded_taxonomy_eval_v3`

This cleanup prepared a local release bundle under:

- `models/released/evidence_grounded_taxonomy_eval_v3/`

Published model host:

- `https://huggingface.co/Legend2727/evidence_grounded_taxonomy_eval_v3`

The local release bundle remains the source for future refresh uploads.

## Local Bundle Contents

Tracked metadata files:

- `models/released/evidence_grounded_taxonomy_eval_v3/README.md`
- `models/released/evidence_grounded_taxonomy_eval_v3/artifact_manifest.json`
- `models/released/evidence_grounded_taxonomy_eval_v3/metrics.json`
- `models/released/evidence_grounded_taxonomy_eval_v3/label_map.json`
- `models/released/evidence_grounded_taxonomy_eval_v3/best_thresholds.json`
- `models/released/evidence_grounded_taxonomy_eval_v3/best_thresholds_by_label.json`

Local-only large file:

- `models/released/evidence_grounded_taxonomy_eval_v3/best_model.pt`

`best_model.pt` is intentionally ignored by git.

## Run Local Inference

With a local image root and the local release bundle in place:

```bash
python scripts/infer_evidence_grounded_taxonomy.py \
  --image-root /path/to/image_root \
  --checkpoint models/released/evidence_grounded_taxonomy_eval_v3/best_model.pt \
  --thresholds models/released/evidence_grounded_taxonomy_eval_v3/best_thresholds.json \
  --source-path images/source/shard_00/example.png \
  --target-path images/target/shard_00/example.png \
  --instruction "Remove the object from the left side of the image." \
  --lang en
```

The recovered evidence-grounded path is EN / HI / BN only and emits taxonomy predictions.

## Hugging Face Publication

Canonical host:

- `https://huggingface.co/Legend2727/evidence_grounded_taxonomy_eval_v3`

Refresh workflow for maintainers:

```bash
huggingface-cli login
huggingface-cli upload-large-folder Legend2727/evidence_grounded_taxonomy_eval_v3 models/released/evidence_grounded_taxonomy_eval_v3 --repo-type model
```

Notes:

- `huggingface-cli` is available in the cleanup environment.
- this repo was published successfully to the canonical host above.
- the prepared bundle is self-contained for upload: weight, thresholds, label map, metrics, and release README live in the same directory.

## GitHub Release Fallback

If Hugging Face publication is not possible, the fallback is:

1. create a GitHub Release
2. upload `models/released/evidence_grounded_taxonomy_eval_v3/best_model.pt` as a release asset
3. keep the JSON metadata files in the repository tree
4. point users back to this repo for code and evaluation reports

## Demo Outputs

Inspectable demo outputs live under:

- `reports/demo_inference/`

In this cleanup step they remain artifact-backed replay outputs because no local image root was available for a live sample run.
