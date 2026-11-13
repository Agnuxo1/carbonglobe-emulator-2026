# Full-data refit implementation

## Current verification status

The new implementation is complete in source form. Python compilation, CLI help and the NumPy-only mathematical/contract smoke pass in the scratch environment. The complete synthetic CPU training/reload smoke is implemented but **has not run**: neither installed scratch Python has PyTorch or LightGBM, and the Windows/clone connection is unavailable. No production preparation, fitting, finalization, inference or submission has been launched by this implementation.

The exact reviewed `PROPOSED_FIXED_LENGTHS.json` exists on Windows and the clone with SHA-256 `06d4beef87f9880e8afe493dff1029d5cbf7a28ad7dd5ab0661eca2972cd5358`. Its bytes are not available in this scratch backup and have not been reconstructed. Restore that exact file when access returns. Preparation and the full smoke reject any different digest.

The original proposal is preserved as [PROPOSAL.md](PROPOSAL.md), SHA-256 `92fed6402e6b16f32a9b1bfbbec25db6917c1a02671ce368d1b96dbb161ceab0`.

## Files and numerical contract

The new runner is `src/full_data_refit_20261113.py`. It imports the unchanged inference-only helper `src/carbon_inference.py`, verified against SHA-256 `8d03ba50841828ca27e20661b9a02aa6aaeb034409defd6db3245a03f4b6fe5f`. It does not import or modify either old training engine.

- Auxiliary MLPs: two new fits, seeds 20261113 and 20261114, exactly 4 and 16 epochs, original float32 scalers, original 116 features, batch 2048, AdamW and `CosineAnnealingLR(T_max=30)`.
- Age specialists: 30 new CPU boosters, original 635 features and requested parameters, exactly the 30 tune-selected durations totaling 32,128 trees. No validation set, early-stopping callback or parameter selection is used. A booster that cannot produce its complete tree count fails the gate.
- All 54,152 historical sites are training sites. The runner computes no new holdout or OOF metric. Training loss is diagnostic only and does not select a neural checkpoint.
- The empirical affine decoder, age anchor, target scales and final nonnegative clamps are unchanged. LightGBM features do not receive the MLP scaler or its [-8,8] clamp.
- The MLP seed mean accumulates in float64 and returns float32. Specialists retain float64. The fixed mixture is `0.5 * A.astype(float64) + 0.5 * G`, retained as float64 before the existing `.9g` CSV format.

Every new checkpoint identifies this refit runner and its new plan by SHA-256. Old source hashes are recorded as reused mathematical provenance, never as the producer of the new trained weights.

## Verification commands

From `/srv/ai-data-250-a/ieee-carbon`, after synchronizing the new source and the exact fixed-length manifest:

```bash
.venv/bin/python src/full_data_refit_20261113.py smoke-math \
  --receipt research/full_data_refit_20261113/checks/math_smoke_new.json

.venv/bin/python src/full_data_refit_20261113.py smoke \
  --fixed-lengths research/full_data_refit_20261113/PROPOSED_FIXED_LENGTHS.json \
  --output research/full_data_refit_20261113/cpu_smoke
```

The full smoke fits 30 synthetic boosters for all 32,128 fixed iterations and two synthetic MLPs for 4/16 epochs, using CPU only. It verifies reloads, original scaler precision, scheduler values, exact optimizer-step counts and tree counts. It then copies the inference bundle to an isolated directory, removes the synthetic training directory, reproduces all three full 185,130-row CSVs twice, checks byte-exact CPU replay, tests the deliberate SHA-failure path, and confirms that an invalid mission UUID is rejected before CUDA initialization. Synthetic artifacts are explicitly marked and rejected by the production reader. Temporary synthetic inputs and weights are removed after the smoke; the receipt remains.

These checks, once executed, will still not certify production GPU fitting or production prediction reproduction.

## Production phases — coordinated start remains pending

The commands are provided for a later coordinated run. First preserve and deliver the existing candidate. Then run each phase explicitly; none of these commands automatically starts another training phase.

```bash
.venv/bin/python src/full_data_refit_20261113.py prepare \
  --root /srv/ai-data-250-a/ieee-carbon \
  --fixed-lengths research/full_data_refit_20261113/PROPOSED_FIXED_LENGTHS.json \
  --output research/full_data_refit_20261113/run

CUDA_VISIBLE_DEVICES="" .venv/bin/python src/full_data_refit_20261113.py train-cpu \
  --root /srv/ai-data-250-a/ieee-carbon \
  --output research/full_data_refit_20261113/run

CUDA_VISIBLE_DEVICES=GPU-d60d12af-82b8-9913-7a5e-1177e774dafa \
.venv/bin/python src/full_data_refit_20261113.py train-gpu \
  --root /srv/ai-data-250-a/ieee-carbon \
  --output research/full_data_refit_20261113/run

.venv/bin/python src/full_data_refit_20261113.py finalize \
  --output research/full_data_refit_20261113/run
```

Preparation copies the exact original scalers and verifies source artifacts; it does not fit models. The CPU phase uses four threads, a 6 GiB address-space cap and a cumulative 45-minute budget, and rejects an accidental torch import. The GPU phase uses the canonical reserved-device lease and `/srv/ai-data-250-a/ieee-carbon/gpu1_ieee.lock`, with a cumulative five-minute budget and a 6 GiB resident-host-memory check. Both training phases retain the mission training cutoff. Completed models can resume only with unchanged code, plan, environment and hashes. An unclean resource ledger or an orphaned weight file fails instead of being silently reset or overwritten.

Finalization requires all 32 completed weights and reload receipts. It creates `run/inference_bundle/` with an explicit artifact inventory: models, records, scalers, plan, fixed-length provenance and the two source files. Training caches and public input data are excluded. The existing source studies remain unchanged.

## Independent inference

The bundle requires only the five prepared public files whose hashes are recorded in its manifest: the two `ssp*_test_y0.npy` files, the two `test_xann_ssp*.npy` files and `sample_submission.csv`. The existing verified official-data preparation pipeline supplies these files. Original training arrays, original reference checkpoints and original study directories are not required.

On the research machine, after the GPU owner releases the canonical lock:

```bash
CUDA_VISIBLE_DEVICES=GPU-d60d12af-82b8-9913-7a5e-1177e774dafa \
.venv/bin/python src/full_data_refit_20261113.py predict \
  --bundle research/full_data_refit_20261113/run/inference_bundle \
  --inputs data \
  --output research/full_data_refit_20261113/reference_predictions \
  --family all --mode mission --device cuda:0
```

Create the exact expected-SHA map from that reference receipt, then perform a separate replay:

```bash
.venv/bin/python -c 'import json,pathlib; p=pathlib.Path("research/full_data_refit_20261113/reference_predictions"); r=json.loads((p/"reproduction_receipt.json").read_text()); (p/"expected_sha256.json").write_text(json.dumps({k:v["sha256"] for k,v in r["outputs"].items()},indent=2)+"\n")'

CUDA_VISIBLE_DEVICES=GPU-d60d12af-82b8-9913-7a5e-1177e774dafa \
.venv/bin/python src/full_data_refit_20261113.py predict \
  --bundle research/full_data_refit_20261113/run/inference_bundle \
  --inputs data \
  --output research/full_data_refit_20261113/replayed_predictions \
  --family all --mode mission --device cuda:0 \
  --expected-sha256-json research/full_data_refit_20261113/reference_predictions/expected_sha256.json \
  --reference-dir research/full_data_refit_20261113/reference_predictions
```

Portable CPU inference from inside a transferred `inference_bundle/`, with the prepared public input directory next to it:

```bash
python code/full_data_refit_20261113.py predict \
  --bundle . --inputs ../data --output ../portable_cpu_predictions \
  --family all --mode portable --device cpu
```

Portable mode also accepts a CUDA device such as `cuda:0` on an organizer's machine. It has no calendar cutoff and no fixed GPU UUID. The reserved research GPU requires mission mode and its canonical lock. `--family aux_shape` and `--family state_shape_climate_delta` reproduce either family separately; the specialist-only path requires CPU. `--family blend` exports only the fixed mixture, and `--family all` exports the two families and their mixture.

For a second replay, `--expected-sha256-json` accepts a JSON object mapping exactly the exported family names to their complete CSV SHA-256 values. `--reference-dir` additionally records prediction-to-prediction numerical differences. A failed SHA gate returns exit code 2 and retains the generated CSVs and difference report. No tolerance is relaxed to pass a byte-exact gate. CPU and GPU results are not assumed byte-identical across backends.

All output directories must be fresh. Prediction output must be outside the input directory and outside the model bundle. The runner performs no network access, Kaggle operation, publication or final-submission selection.
