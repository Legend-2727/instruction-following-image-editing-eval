# Released Model Bundle: evidence_grounded_taxonomy_eval_v3

This directory is the local release bundle for the final model artifact in this repository.

Release status:

- release-designated model: yes
- tracked large weights in git: no
- preferred external host: Hugging Face model repo
- current cleanup status: local bundle prepared, manual upload still required

## Local Bundle Contents

Tracked metadata files in this directory:

- `README.md`
- `artifact_manifest.json`
- `metrics.json`
- `label_map.json`
- `best_thresholds.json`
- `best_thresholds_by_label.json`

Local-only large file:

- `best_model.pt`

`best_model.pt` is intentionally ignored by git. During cleanup, the local checkpoint was copied here from the older root import folder so this directory can serve as the upload-ready bundle.

## Use This Bundle Locally

Example live inference command:

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

## Manual Hugging Face Publication

The cleanup environment had Hugging Face CLI available but no local Hub token, so upload was not executed automatically.

Typical maintainer workflow:

```bash
huggingface-cli login
huggingface-cli repo create <namespace>/evidence_grounded_taxonomy_eval_v3 --exist-ok
huggingface-cli upload-large-folder <namespace>/evidence_grounded_taxonomy_eval_v3 models/released/evidence_grounded_taxonomy_eval_v3 --repo-type model
```

## Related Report Bundle

Tracked evaluation/report files also live under:

- `reports/evidence_grounded/v3/`

That report bundle contains:

- `artifact_manifest.json`
- `metrics.json`
- `label_map.json`
- `best_thresholds.json`
- `best_thresholds_by_label.json`
- `per_class_test_benchmark.csv`
- `train_history_v3.json`

If the original local import folder is still available, it may also contain additional raw exports such as prediction JSONL files that were intentionally not migrated into normal git tracking.

## Current Recovery Status

The repo now includes the recovered lineage notebook and wrapper scripts for the evidence-grounded path. What is still not claimed is a fresh end-to-end retraining verification inside this cleaned repo.
