# Copilot setup for this repository

This repo works best when you use **both**:
1. `AGENTS.md` at the repo root for always-on project context.
2. `.github/agents/xlingual-edit-eval-recovery.agent.md` for the dedicated recovery / implementation workflow.

## Recommended setup

### Option A — Best setup
Use the normal Copilot agent **plus** the custom recovery agent.

- Keep `AGENTS.md` in the repo root.
- Keep the custom agent file in `.github/agents/`.
- Commit both files to the branch you are using.
- Open Copilot Chat in VS Code.
- Select the custom agent `xlingual-edit-eval-recovery` when you want multi-step repo work.
- Switch back to the normal agent for quick one-off questions.

### Option B — Minimal setup
If you do not want a custom agent yet, keep only `AGENTS.md`.
That still gives Copilot repository-wide instructions and is enough for everyday coding.

## How to prompt Copilot in this repo
Use **phase-based prompts**.
Do not ask Copilot to do the whole project in one shot.

Good examples:

- `Phase 0 only: audit this repo, list concrete bugs, and patch only the highest-value blockers.`
- `Phase 0 only: normalize metadata paths and make the sample pipeline run on Linux.`
- `Phase 1 only: build a loader and validator for Legend2727/xLingual-picobanana-12k.`
- `Phase 2 only: implement dual-judge labeling on a 32-sample smoke subset.`
- `Phase 3 only: make the human review app write corrected labels to a separate annotation log.`
- `Phase 4 only: implement the compact multilingual student model and run a smoke train.`
- `Generate a paper-ready experiment matrix and ablation plan for this repo.`

## How to keep Copilot from drifting
Ask for one of these in the prompt:
- `do not start a full run; use a smoke test first`
- `keep changes minimal and testable`
- `show the exact files you plan to edit before editing`
- `preserve AGENTS.md rules`
- `treat the HF dataset as canonical after step 4`

## Practical advice
- Start each new session by telling Copilot which phase you want.
- If Copilot starts rewriting the whole repo, stop it and narrow the prompt.
- If Copilot tries to center the project around InstructPix2Pix generation, redirect it: the main paper should be the multilingual classifier, not legacy editor benchmarking.
- Always ask for a smoke test command before a large run.
- Ask it to export machine-readable outputs after every stage.

## Suggested first three Copilot tasks
1. `Phase 0 only: audit the repo and fix cross-platform path bugs.`
2. `Phase 1 only: create a validator and split builder for the uploaded xLingual PicoBanana dataset.`
3. `Phase 2 only: implement a dual-judge schema and save merged labels plus agreement metadata.`
