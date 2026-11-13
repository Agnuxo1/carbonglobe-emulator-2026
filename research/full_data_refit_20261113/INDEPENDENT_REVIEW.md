# Independent full-refit review

Date: 2026-11-13

Reviewed source: `full_data_refit_20261113.py`

SHA-256:
`73a96a2d93fca61833e419e4efac6a09563ca8c5f75910b05937b2405ff35fcb`

## Finding

No blocking implementation defect was identified in the requested static review.
The source was not edited. This finding does not claim that production training
or the complete synthetic Torch/LightGBM smoke has run.

## Reviewed behavior

- `finalize` requires completion receipts and weight hashes for exactly 32 models,
  then copies an explicit artifact allowlist into a separate inference bundle.
  Training arrays, feature caches, and public inputs are not copied into it.
- `read_bundle` checks the code and primitive identities, plan hash, fixed
  configuration, model ordering, declared durations, and completion receipts.
  Weight and receipt hashes are checked for the requested model families.
- `load_new_mlp` validates the new checkpoint contract and the exact float32
  scaler-array hashes before restoring the architecture with a strict state load.
- `predict_auxiliary` preserves 128-site scenario batching, the frozen decoder,
  and float64 seed averaging followed by float32 conversion. Specialist inference
  preserves the 635-column schema, saved tree counts, normalized residual anchor,
  and nonnegative clamp. The fixed family mixture accumulates in float64.
- Prediction reads only the bundled weights and metadata plus the declared
  public inputs. It does not invoke an old engine verifier or read train targets.
  The portable inference budget has no competition-date cutoff. Mission GPU
  execution delegates to the canonical GPU ownership context.
- CSV export validates all 185,130 ordered unique IDs and finite, nonnegative
  predictions, preserves the declared `.9g` format, and retains outputs alongside
  an explicit failed gate if expected hashes differ.
- The synthetic smoke is marked as synthetic, exercises isolated copied code and
  weights after deleting its original run directory, and verifies that the
  production reader rejects synthetic checkpoints. The command-line failure
  guard does not write a receipt into a pre-existing output directory rejected
  before ownership is established.

The production CPU phase uses the declared Linux address-space cap and
per-iteration budget checks. The GPU and inference paths retain elapsed-time and
resident-memory checks at their numerical boundaries. This static review does
not measure real runtime or peak memory.

## Executed representation check

`check_feature_parity.py` compared the frozen original specialist module against
the reviewed refit using Python 3.12.14 and NumPy 2.3.5. All seven input cases
produced byte-identical float32 climate feature arrays, with maximum absolute
difference zero. Cases covered random float32, a wide dynamic range, float64,
noncontiguous views, Fortran layout, constant profiles, and linear profiles.

The 635 feature names and their order, the 519 climate dimensions, and the BAD/GOOD
index sets were also exactly equal. Neither Torch nor LightGBM was imported.
`feature_parity_receipt.json` records source hashes and case output hashes.

## Remaining runtime evidence

The full synthetic CPU training and isolated prediction replay still require an
environment with Torch and LightGBM. No heavyweight dependency was installed in
scratch for this review. The executed NumPy test does not replace that smoke or a
saved-weight prediction replay on the intended production backend.
