# LFS Cleanup Instructions

## What Was Done In The Current Tree

The current tree no longer relies on Git LFS.

During cleanup:

- stale LFS tracking rules were removed from `.gitattributes`
- old tracked checkpoint artifact directories were removed from the working tree
- large release/history weights were treated as local-only files, not normal git content

## What Was Not Done

No git history rewrite was executed.

That means old LFS history may still exist in the repository history outside the current tree.

## Optional Maintainer Follow-Up

If you later want to clean historical LFS objects from the full repository history, do it in a separate maintenance operation with backups and coordination.

Typical follow-up workflow:

1. clone a fresh mirror of the repo
2. inspect current LFS usage
3. export or rewrite historical LFS pointers if desired
4. validate every branch and tag
5. force-push only after maintainer review

Possible tools maintainers often evaluate for that kind of task:

- `git lfs migrate export`
- `git filter-repo`

Those tools are intentionally not run by this cleanup step.

## Recommendation

Treat historical LFS cleanup as a separate repository-maintenance project, not as part of normal model/documentation updates.

For an exact maintainer-facing remote cleanup procedure with concrete commands, see `docs/github_lfs_remote_cleanup.md`.
