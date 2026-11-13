# Packaging manifest revision

Date: 2026-11-13

## Deliverable

`manifest/build_manifest.py` is a complete replacement for the published builder.
Its SHA-256 is
`ad9db7c95c1b44d941d79fda49424a2a78b66d48018546e1d2c858cd9179a239`.
The file was prepared and tested in scratch only. No Windows repository, remote
repository, index, commit, or `.gitattributes` file was changed.

## What CI requires

The published workflow compiles `src` and `tests`, imports `common` and `dataset`,
runs `tests/test_hrc_port.py`, and checks every entry of `manifest/MANIFEST.json`
against the first 16 hexadecimal characters of SHA-256 over its file bytes.
It does not regenerate the manifest or test inventory completeness. Its passing
status would not establish end-to-end reproduction of a competition submission.

## Changes

- File discovery uses NUL-separated `git ls-files --cached` results. Untracked
  work is excluded, and Git-ignored paths are excluded even if already tracked.
- The default scope retains the original top-level Python files under `src`,
  `tests`, and `servidor`, the seven fixed core files, and the builder itself.
- `--include-research-docs` also includes already-tracked, non-ignored text and
  code under `research` and `docs`. This optional scope assumes the caller has
  reviewed those tracked files; the builder does not classify secrets by content.
- The output manifest excludes itself. Missing tracked files, symbolic links,
  and paths escaping the repository are rejected before replacing the manifest.
- Hashes use binary reads with no newline normalization. JSON output is
  deterministic and the manifest replacement is atomic.
- Metadata records version 1.1.0, date 2026-11-13, and historical selected
  S1109.csv, public score 0.060, submission ID 58145080. Model reproduction remains
  explicitly pending. The previous H1 verdict is retained as a dated historical
  statement, without a new validation claim.

## Verification

`verify_revision.py` ran successfully in a temporary Git fixture with zero
commits. The receipt records checks for tracked/untracked/ignored paths, the
default and expanded scopes, Unicode and spaces, manifest self-exclusion, CRLF
preservation, deterministic output, unchanged index and attributes, and rejection
of missing files and external symlinks. The fixture also applied the published
CI hash condition successfully. This tests the replacement builder, not the real
repository's complete CI workflow.

An independent static review found no blocking issue. Its scope qualification
was that the optional tracked document inventory still needs a content review.

## Safe integration sequence for the repository owner

1. Copy this replacement to `manifest/build_manifest.py` in the actual repository.
2. Review and stage only the intended source and documentation changes. Newly
   added files must be tracked before the builder can include them. Keep
   `.gitattributes` unchanged, including its `* -text` rule.
3. Run `python manifest/build_manifest.py` for the historical scope, or add
   `--include-research-docs` after reviewing the expanded tracked inventory.
4. Review and stage the resulting `manifest/MANIFEST.json`. Verify that the
   staged versions of every listed file match the working-tree bytes that were
   hashed. Rebuild the manifest if any listed file changes afterward.
5. Run the workflow's existing checks before publishing. A fixture result is not
   a substitute for those checks on the intended commit.

No replacement MANIFEST.json was generated here: the unpublished Windows
worktree contains changes that this scratch checkout does not contain.
