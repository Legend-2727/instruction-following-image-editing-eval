# Model Release Policy

## Release Decision

Only one model is treated as the public release artifact in this repository:

- `evidence_grounded_taxonomy_eval_v3`

This is the current best overall artifact bundle for the completed English/Hindi/Bangla branch.

Preferred publication strategy:

- keep GitHub lightweight
- publish the final weight bundle on Hugging Face
- keep this GitHub repo focused on code, docs, manifests, reports, and demo outputs

## History Policy

The following variants remain available as experiment history only:

- `evidence_grounded_taxonomy_eval` (`v1`)
- `evidence_grounded_taxonomy_eval_v2` (`v2`)

They are kept for comparison and provenance, not as parallel public releases.

## Large Weight Policy

Large `.pt` checkpoints are intentionally not tracked in normal git history.

Why:

- they are too large for a clean source repository
- the repo no longer depends on Git LFS in the current tree
- public git history should keep only lightweight metadata and result files

Expected local path for the final release weight:

- `models/released/evidence_grounded_taxonomy_eval_v3/best_model.pt`

Current cleanup status:

- the local release bundle has been prepared under `models/released/evidence_grounded_taxonomy_eval_v3/`
- the local `best_model.pt` was found in the older root import folder and copied into the release directory
- Hugging Face upload was not executed from the cleanup environment because no local Hub token was available

## What Is Tracked In Git

- report manifests
- metrics
- thresholds
- label maps
- benchmark tables
- prediction exports
- training history JSON files
- public-facing documentation

## What Is Not Tracked In Git

- large model binaries such as `best_model.pt`

## Maintainer Guidance

If you have the final `v3` weight locally:

1. Place it at `models/released/evidence_grounded_taxonomy_eval_v3/best_model.pt`.
2. Keep it untracked.
3. Prefer publishing the prepared release directory to a Hugging Face model repo.
4. If Hugging Face publication is not available, use an external artifact host or a GitHub Release asset instead.

If you do not have the weight locally:

- the repository still remains useful because the tracked report bundle and manifests document the model outputs and evaluation state.

See `docs/model_download_and_usage.md` for exact local usage and manual publication commands.
