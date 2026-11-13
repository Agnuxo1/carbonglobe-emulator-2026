# Full-data refit proposal — November 13, 2026

**Status: proposed only. No refit implementation, training or submission has been performed.** Preserve the existing deliverable and all original run artifacts before beginning a separate refit.

## Fixed comparison

Train the same two model families on all **54,152 historical sites**, adding 21,662 sites to the original 32,490-site fit (a factor of 1.666728). The MLP sees 812,280 site-age rows. This tests adding the available historical training data with a previously selected training duration; it does not create new independent validation evidence.

Retain the original dated evaluation records and their limitations. The earlier audit partition was already reused and descriptive. Once all sites enter training, none of those source sites may be called a holdout or an OOF observation for the refit. Do not recompute a source “validation score” to choose the new models, training lengths or mixture weights.

The [proposed fixed-length manifest](PROPOSED_FIXED_LENGTHS.json) records the 32 source models, verified weight/record hashes, original tune-selected lengths, requested tree parameters and preserved evaluation-reference hashes. It contains no newly selected length or coefficient.

## Auxiliary-shape MLP: two fresh fits

| Seed | Original best_epoch (zero-based) | Fixed full-data epochs |
|---|---:|---:|
| 20261113 | 3 | 4 |
| 20261114 | 15 | 16 |

Initialize each fit from its original seed, not from the old trained weights. Preserve the 116 features, age anchor, residual target, nonnegative inference clamp, 512-wide/depth-3 MLP, LayerNorm, GELU, dropout 0.20, AdamW learning rate 0.0003 and weight decay 0.04, batch size 2,048, float16 autocast/GradScaler, gradient clipping and TF32 behavior.

**Keep CosineAnnealingLR T_max=30.** Run its original epoch-by-epoch schedule for exactly 4 or 16 epochs. Replacing T_max with the shorter duration would change the optimizer schedule and is excluded. Remove tune-driven early stopping and save the weights after the fixed final epoch; finite-value and resource guards remain.

For the narrowest data-addition comparison, preserve each source checkpoint's 116-dimensional feature_mean and feature_std rather than refitting the scaler. Keep the existing target scales 10.90645 and 3.39567. Preserve all feature formulas and the frozen community affine test decoder, without estimating new constants.

Fixed epochs mean more optimizer updates: approximately 397 batches per full-data epoch versus 238 originally. The comparison therefore includes the extra training work associated with seeing the extra sites; it is not a claim about dataset size at an equal gradient-update budget.

## Climate age specialists: 30 fresh fits

Preserve the 635 features (116 state/shape inputs plus 519 forcing summaries), per-age/per-target residual definitions, four CPU threads and every saved `parameters_requested` field. Retain the original seed formula. Fit each booster from scratch with its own fixed `num_boost_round`, without a validation set or early-stopping callback.

Use `parameters_requested`, not a copied `parameters_effective` dictionary that could carry the old 1,200-round override. Data-dependent tree splits and binning naturally change when additional training rows are included; feature formulas and hyperparameters remain fixed.

| Initial age | Height rounds | AGB rounds |
|---|---:|---:|
| 1 | 1191 | 1200 |
| 10 | 1194 | 1199 |
| 20 | 1158 | 1176 |
| 30 | 978 | 1196 |
| 50 | 1160 | 1194 |
| 70 | 1192 | 1200 |
| 100 | 897 | 1199 |
| 150 | 619 | 1200 |
| 200 | 415 | 1199 |
| 250 | 1198 | 1190 |
| 300 | 1101 | 1200 |
| 350 | 743 | 1199 |
| 400 | 494 | 1189 |
| 450 | 851 | 1186 |
| 500 | 1029 | 1181 |

Total: **32,128 fixed boosting iterations** across 30 models. Iterations at the original 1,200-round ceiling stay at that value; this proposal does not extrapolate or retune them.

## Predeclared family mean

Produce the two family outputs separately and one fixed combined candidate:

- `A`: equal mean of the two auxiliary MLP predictions, using the existing float64 accumulation followed by float32 conversion.
- `G`: the 30 age/target specialists assembled in the original sample order, retaining their existing float64 prediction arithmetic.
- `F = 0.5 * A.astype(float64) + 0.5 * G`: one fixed coefficient pair for every age, target, site and scenario; no intercept or fitted correction.

Apply the same frozen decoder to each family. Both family arrays use the same normalized height/AGB units. Combine arrays before CSV formatting, retain float64 for the final mixture, and use the existing nine-significant-digit export convention. Do not select or alter coefficients using the audit or public scores.

## Feasibility and execution estimate

The original 30 climate boosters required **692.67 seconds** of measured fitting. Removing the unused early-stopping tail gives a rough fixed-round equivalent of 672.81 seconds. Linear scaling by the training-site ratio suggests **18.69 minutes**; allow **20–25 minutes** for the CPU refit, feature access and reload checks. This is an estimate, not a measured full-data runtime. Retain the existing four-thread, 6 GiB and 45-minute bounded CPU resource contract.

The two original auxiliary fits completed between the prepared receipt at 14:15:06 UTC and the second completion at 14:16:29 UTC, including feature preparation, 30 executed epochs in total, tune checks and saved-model checks. The new fits total 20 full-data epochs, approximately 33.4 original-data epochs in row work, with no tune evaluation. With existing caches and the same GPU/runtime, **roughly 1–3 minutes** for training is plausible; reserve **five minutes** for GPU fitting and reload checks, plus several minutes for exports and reproduction. GPU use must wait for the current owner and acquire the reserved-device lock. No GPU work is part of this proposal.

The families can occupy their existing CPU/GPU resource envelopes separately. Keep the current verified runtime; changing it would add another variable to the data-addition comparison. Store all refit artifacts in a new directory and preserve the original reference outputs.

## Delivery gates and risks

Save all **32 actual model weights**, the MLP scaler arrays, full feature/anchor/decoder definitions, exact lengths and seeds, source data/record/code hashes and runtime metadata. Retain independent family inference paths and a separate deterministic mixture step. The inference package must generate all three outputs from weights and public test inputs without training, source target arrays or a competition-date cutoff.

The refit must record its own real training-code and plan hashes. Do not label newly trained weights as if they were produced by an unchanged original runner, and do not modify the frozen engines or their guards. A separate compatible reader or entrypoint may be needed for the new manifest. Require reload agreement and reference-backend CSV hash replay for both families and the mixture before treating the package as reproducible.

The main scientific risk is source-to-SSP shift: adding source sites may not correct a decoder or representation mismatch and may worsen SSP performance. Original tune-selected lengths may transfer imperfectly to a larger training set, but they remain fixed. Full-data training removes source holdout evidence; preserve the earlier metrics as dated references rather than treating training agreement as a new validation result. The half-and-half mixture may help or hurt and remains an untested predeclared candidate.
