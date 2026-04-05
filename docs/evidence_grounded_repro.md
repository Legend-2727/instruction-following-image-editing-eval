# Evidence-Grounded Repro

This repo now contains a notebook-grounded evidence-grounded code path, but it should be treated as a recovered research path rather than a fully re-run and fully verified clean-room reimplementation.

## Required Inputs

For live training or inference you need:

- the original-level split JSONL files under `data/final/splits/` or equivalent
- an image root containing paths such as `images/source/...` and `images/target/...`
- the required Python ML dependencies
- access to the backbone model ids used by the notebook

The repo does not ship the image corpus in git.

## Notebook To Script Mapping

Imported notebook:

- `notebooks/evidence_grounded_taxonomy_eval_lineage_v1_to_v3.ipynb`

Original imported filename:

- `../notebookc95272763e.ipynb`

Recovered wrappers:

- `scripts/train_evidence_grounded_taxonomy.py`
- `scripts/finetune_evidence_grounded_taxonomy_v2.py`
- `scripts/finetune_evidence_grounded_taxonomy_v3.py`
- `scripts/infer_evidence_grounded_taxonomy.py`
- `scripts/evaluate_evidence_grounded_benchmark.py`
- `scripts/run_demo_inference_evidence_grounded.py`

## Example Commands

Train v1:

```bash
python scripts/train_evidence_grounded_taxonomy.py \
  --image-root /path/to/image_root \
  --splits-dir data/final/splits
```

Fine-tune v2:

```bash
python scripts/finetune_evidence_grounded_taxonomy_v2.py \
  --image-root /path/to/image_root \
  --splits-dir data/final/splits
```

Fine-tune v3:

```bash
python scripts/finetune_evidence_grounded_taxonomy_v3.py \
  --image-root /path/to/image_root \
  --splits-dir data/final/splits
```

Evaluate a recovered checkpoint:

```bash
python scripts/evaluate_evidence_grounded_benchmark.py \
  --version v3 \
  --image-root /path/to/image_root \
  --splits-dir data/final/splits
```

Run live inference:

```bash
python scripts/infer_evidence_grounded_taxonomy.py \
  --image-root /path/to/image_root \
  --source-path images/source/shard_00/example.png \
  --target-path images/target/shard_00/example.png \
  --instruction "Remove the object from the left side of the image." \
  --lang en
```

Materialize demo outputs:

```bash
python scripts/run_demo_inference_evidence_grounded.py --mode artifact
```

If local images are available, the demo script can also be pointed at a real image root and used in live mode.

The recovered evidence-grounded lineage emits taxonomy predictions, matching the released `v1` / `v2` / `v3` report bundles.

## What Is Direct Recovery Vs Wrapper Logic

Directly grounded in the notebook:

- the v1 training loop
- the v2 fine-tuning loop
- the v3 fine-tuning loop
- the evidence-grounded model architecture
- the taxonomy/evidence signals
- the threshold-tuning logic

Lightly cleaned wrapper logic:

- CLI argument handling
- cleaned repo output directories
- fallback path resolution between cleaned paths and local import folders
- artifact replay mode for small checked-in demo outputs

## Demo Result Files

Small inspectable demo files are stored in:

- `reports/demo_inference/`

These are useful for checking the output format quickly without opening large report bundles. The checked-in samples are artifact-backed replay files, while live mode requires the local image corpus and the local `v3` checkpoint.
