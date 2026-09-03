# RoK4 Repository Instructions

## Baseline Documentation

Before creating or pushing a commit, tag, or branch that establishes or changes a trained-policy baseline:

1. Update the `Policy Baselines` table at the top of `README.md`.
2. Verify that the run name, checkpoint filename, and code commit all represent the same training configuration.
3. Record Isaac Sim, MuJoCo Sim2Sim, and hardware Sim2Real validation separately. Never infer an unperformed
   validation stage from training success, branch ancestry, or another checkpoint.
4. Keep the current development baseline separate from previous validated references.
5. Update the relevant RST sources, regenerate their HTML and PDF outputs, and update `CHANGELOG.md`.
6. Inspect `git status`, branch upstreams, and the decorated branch graph before pushing.
7. Do not merge, rebase, fast-forward, force-update, or otherwise move any branch other than the current branch
   without the user's explicit approval. Pushing several branch names is not approval to make their commits equal.
8. Run focused tests and `git diff --check`, then review the final staged file list before committing.

These checks apply even when a documentation-only commit records a checkpoint trained from an earlier code commit.
In that case, record the training code commit and the later documentation or validation commit separately.
