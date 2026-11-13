# Research handoff — November 13, 2026

This branch preserves reviewed research tools prepared while the reference computer was unavailable. It is an implementation checkpoint, not a final model release. The complete restored-model inference and full-data refit have not been executed from this branch.

## Prepared work and actual verification

| Item | Implemented behavior | Verification completed |
| --- | --- | --- |
| `src/full_data_refit_20261113.py` | Separate preparation, CPU fitting, GPU fitting, packaging and independent prediction for two neural models and 30 age/target boosters | Compilation, NumPy contract checks and independent parity against the frozen climate-feature implementation; real training smoke remains pending |
| `src/carbon_inference.py` | Frozen feature, decoder, anchor, model and CSV primitives; portable and reserved-device execution modes | Previously measured real-GPU replay for the three original transfer ensembles; new refit weights do not exist yet |
| `research/reproducibility_20261113/run_verified_replays.py` | Two bounded saved-weight inferences, schema and literal byte comparison, and a separate historical-prediction comparison | Eighteen synthetic checks, including two full-format 185,130-row fixtures; no emulator or GPU execution in those checks |
| `manifest/build_manifest.py` | Inventory only Git-tracked, non-ignored files and hash their exact bytes | Isolated Git fixture covering tracked/untracked/ignored paths, CRLF, Unicode paths, CI hash compatibility and atomic failure behavior |

The dated evidence receipts beside the source files identify the precise tested source hashes. Synthetic and mathematical checks do not certify the pending production computations.

## Full-data refit contract

The original [proposal](full_data_refit_20261113/PROPOSAL.md) is preserved byte for byte as a historical preregistration. Its opening status describes the moment it was written; the implementation status is recorded in this handoff and the implementation notes.

The refit adds all 54,152 historical sites, preserves the original feature scalers, uses exactly four epochs for seed 20261113 and sixteen for seed 20261114, and retains `CosineAnnealingLR(T_max=30)`. The 30 climate-aware age specialists retain their original requested parameters and their fixed, previously selected durations, totaling 32,128 boosting iterations. The family mixture is fixed at 0.5/0.5. No new source validation score or public-score-fitted coefficient is part of this experiment.

The exact `PROPOSED_FIXED_LENGTHS.json` remains on the reference systems and must be synchronized before the real smoke or preparation step. Its SHA-256 is `06d4beef87f9880e8afe493dff1029d5cbf7a28ad7dd5ab0661eca2972cd5358`. Its missing bytes have not been reconstructed or guessed. The original source checkpoints and their manifests are also required by preparation; the resulting inference bundle will contain its own scalers and weights and will not require those training references.

Production fitting remains gated on a real CPU smoke with PyTorch and LightGBM, independent review, resource checks and preservation of the existing candidate package. The available scratch Python environments lack those two libraries, so the complete synthetic training/save/reload smoke has been implemented but not run there.

## Restored candidate verification

The selected historical submission is `S1109.csv`, submission ID 58145080, with an observed public score of 0.060. Composition from its archived predictions reproduces the historical CSV, but that is separate from reproduction from saved model weights. The recovered neural models differ from the historical predictions, including after a controlled seed-2 trial with the original PyTorch runtime. A full reconstructed prediction must therefore be classified and evaluated from its actual output, not assigned the historical score.

The frozen restored package is held outside Git. Its release-manifest SHA-256 is `7a646394e23d81377f4385fdbcec8d9382095902eb941a2062cb21159f71d14c`. The wrapper must remain outside that package. When the reference machine is available, execute it with the reference interpreter, all public test caches and the protected historical CSV. Allow 1,800 seconds per inference and approximately 4,000 seconds for the complete supervising service. The inference child owns the canonical GPU lock; an additional lock around the wrapper would conflict with it.

The gate must establish all of the following before a new Kaggle upload: immutable package and inputs, two successful saved-weight executions, 185,130 correctly ordered IDs, finite nonnegative predictions, identical CSV bytes, and a separately recorded historical comparison. The comparison error measures differences between predictions; it is not a predictive MSE against hidden labels.

## Repository integration

This branch was based on the published `master` commit `48e11014622e0233146639f0b4c8d06bd88cb612`. The reference Windows checkout also has unpublished work, including the previously prepared public snapshot. Fetch and integrate this branch without replacing that work or rewriting its frozen source files. Regenerate `manifest/MANIFEST.json` after staging the intended public files, using `--include-research-docs` when research evidence should be inventoried. Preserve the existing `* -text` Git attributes because their exact-byte policy is part of the recorded hashes.

Competition data, model weights, private session records, service credentials, account details and private consultation records are not included in this branch. Final code/weight archives, the technical report and the organizer form remain separate delivery steps after verification and final candidate selection.

## Attribution for the new research

The feature and inference primitives were extracted without numerical changes from this project's frozen experiment runners. The empirical affine test decoder follows the public [CarbonGlobe: from 0.142 to 0.060](https://www.kaggle.com/code/lucasbruzzone/carbonglobe-from-0-142-to-0-060) notebook by Lucas Bruzzone, published under Apache 2.0. The decoder is an empirical hypothesis rather than an organizer-certified encoding specification. The age-specialist direction was informed by the public [EcoPhysNet](https://www.kaggle.com/code/robinamirbahar/ecophysnet-physics-guided-carbon-cycle-emulator) notebook; this refit does not copy its implementation or claim a verified score for it. Benchmark and competition data retain their own terms; see the repository attribution and the [official CarbonGlobe project](https://github.com/zhwang0/carbon-globe).
