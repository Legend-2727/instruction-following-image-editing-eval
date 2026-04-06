# GitHub LFS Remote Cleanup

## Scope

This document is for maintainers only.

It separates:

- safe current-tree cleanup that is already done
- optional remote/history cleanup that would rewrite history

## Already Done In The Current Tree

The current branch already reflects the no-large-weights-in-git policy:

- `.gitattributes` no longer enables Git LFS for model binaries
- `.gitignore` keeps release/history `.pt` files out of normal git tracking
- stale current-tree checkpoint artifacts were removed from `checkpoints/` and `runs/`
- the release-designated `v3` checkpoint now lives locally under `models/released/evidence_grounded_taxonomy_eval_v3/best_model.pt` and stays untracked

None of those current-tree changes rewrote history.

## What Still Remains Optional

Old GitHub LFS objects may still exist in repository history and may still count toward remote storage until maintainers do a separate history-rewrite maintenance pass.

This is optional and should be treated as a deliberate repo-maintenance operation, not a normal feature change.

## Recommended Safety Rules

Before any history rewrite:

1. make a full backup or fresh mirror clone
2. coordinate with anyone who has open branches or forks
3. expect all collaborators to re-clone after the rewrite
4. do not run these commands from your normal working clone

## Preflight Checklist

Do not proceed past preflight unless every item below is true:

- GitHub is reachable from the current environment
- `git lfs version` works
- `git filter-repo --version` works
- you can authenticate to the GitHub remote used by `origin`
- you can create and destroy a disposable mirror clone outside the normal working repo
- you have confirmed that the current working tree is clean and pushed
- maintainers understand that the final push will be `--force --mirror`
- maintainers have separately confirmed that the Hugging Face model repo is out of scope for this operation

Useful preflight checks:

```bash
git remote -v
git ls-remote origin
git lfs version
git filter-repo --version
git lfs ls-files
```

Non-destructive history checks you can run before the rewrite:

```bash
git rev-list --all -- checkpoints/stage2_taxonomy
git rev-list --all -- checkpoints/stage2_taxonomy_only_heavy
git rev-list --all -- runs/stage1_binary_multilingual
git log --all --name-only -- '*.pt' '*.pth' '*.bin' '*.safetensors'
```

## Example Mirror-Clone Workflow

Create a disposable mirror clone:

```bash
git clone --mirror <REPO_URL> instruction-following-image-editing-eval.git
cd instruction-following-image-editing-eval.git
```

Fetch all LFS objects before rewriting:

```bash
git lfs fetch --all
```

Export historical LFS pointers back into normal git objects for the old large-file patterns:

```bash
git lfs migrate export --everything --include="*.pt,*.pth,*.bin,*.safetensors"
```

Optionally remove old stale artifact paths from all history as well:

```bash
git filter-repo \
  --path-glob 'checkpoints/stage2_taxonomy/**' \
  --path-glob 'checkpoints/stage2_taxonomy_only_heavy/**' \
  --path-glob 'runs/stage1_binary_multilingual/**' \
  --invert-paths
```

Then aggressively prune rewritten objects:

```bash
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

Force-push only after maintainer review:

```bash
git push --force --mirror origin
```

## After The Rewrite

Expect follow-up maintainer work:

- tell collaborators to re-clone
- re-check `.gitattributes` and `.gitignore`
- verify GitHub Releases or external weight links still point to the intended `v3` artifact
- verify that GitHub LFS billing/storage drops after old refs are gone

If GitHub continues to report old LFS storage after the rewrite and remote refs are gone, maintainers may need to open a GitHub support ticket for server-side cleanup.

## Post-Cleanup Validation Checklist

After the force-push, verify all of the following:

- collaborators have been told to re-clone
- `git lfs ls-files` is empty in a fresh clone unless new LFS usage was intentionally introduced later
- stale history paths no longer appear in history lookups
- the current branch still contains the cleaned docs, scripts, reports, and release metadata
- the local-only `best_model.pt` is still not tracked in git
- GitHub Releases and the Hugging Face model page still point to the intended `v3` bundle

Suggested validation commands in a fresh clone:

```bash
git lfs ls-files
git log --all --name-only -- '*.pt' '*.pth' '*.bin' '*.safetensors'
git log --all --name-only -- 'checkpoints/stage2_taxonomy' 'checkpoints/stage2_taxonomy_only_heavy' 'runs/stage1_binary_multilingual'
git status --short --branch
```

## Recommendation

Prefer leaving history alone unless GitHub LFS storage pressure or policy requires the rewrite. The current tree is already safe for ongoing code and documentation work.
