# Legacy Components

## Not Legacy

The following are still part of the public project story:

- the lightweight baseline
- multilingual split/dataset utilities already present in the repo
- review queue and merge tooling
- evidence-grounded report bundles

## Historical Or Legacy Paths

The repo still contains older code paths that are useful as history or reference, but are not the main release story.

### Older editor-generation / judge pipeline

These files reflect the earlier branch centered on generating edits and judging them:

- `scripts/generate_edits.py`
- `scripts/vlm_judge.py`
- `scripts/analyze_vlm_results.py`
- `scripts/utils/editor_model.py`
- `config/eval_config.yaml`

They are preserved because they remain informative for the branch history and open-VLM benchmarking workflow, but they are not the main public entry point anymore.

### Recovery-era notes

These repo notes are still present as internal historical context, not as primary public documentation:

- `AUDIT_2026-03-18.md`
- `progress.md`
- `TRAIN_STUDENT_SANITY_CHECK_GUIDE.md`
- `COPILOT_SETUP.md`

### Removed current-tree artifacts

Old tracked checkpoint directories from earlier stages were removed from the current tree during cleanup because they were stale model artifacts, not source code:

- `checkpoints/stage2_taxonomy/`
- `checkpoints/stage2_taxonomy_only_heavy/`
- `runs/stage1_binary_multilingual/`

Their removal does not claim the underlying ideas are invalid. It only reflects the current public packaging policy for this repo.
