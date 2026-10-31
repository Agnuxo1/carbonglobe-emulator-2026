# An unfitted anchor, a rollout-trained emulator, and a leaderboard-chosen blend: emulating ED year-40 forest state under scenario shift

**Technical report — IEEE BigData Cup 2026, AI Emulation Challenge (CarbonGlobe)**
Francisco Angulo de Lafuente (independent researcher, Madrid, Spain) · GitHub [@Agnuxo1](https://github.com/Agnuxo1)
Code: https://github.com/Agnuxo1/carbonglobe-emulator-2026 (MIT)
Version of this report: 2026-10-31. Competition open until 2026-11-16 05:00 UTC.

---

## 1. Result

**Public leaderboard 0.076 MSE, about rank 71 of 218 teams, against a field median of
0.092 and a leader at 0.037.** The submitted system is

> ŷ₄₀ = 0.7 × (age-shifted anchor) + 0.3 × (one-step emulator rolled out 40 years)

which beats both of its parts (0.094 and 0.166) and improves on the best unfitted
predictor by 19 %.

Alongside that we report a methodological result that we believe is the more transferable
contribution: **on this benchmark, held-out validation on the present-day distribution does
not rank models for the future-climate test set**, and the discrepancy scales with how hard
a system was fitted — from 0.94× for an unfitted constant to 6.05× for a gradient-boosted
direct regression. We quantify it with eight submissions used as calibration points.

## 2. Task and metric

For 6171 globally distributed sites and 15 seed forest ages, under SSP126 and SSP585
forcing, forecast the December state of year 40 for canopy height and aboveground biomass,
each divided by its global mean. Score: MSE over the 185 130 submitted values.

We first pinned down the target convention, because the public description leaves room for
doubt ("targets are annual"). The organizers' own tutorial indexes
`glob_Y{age}.npy[s, year, 11, :]` and states that December is used for Y in all
experiments; year 40 is index 40 of the 41 stored years, index 0 being the initial state.
We also verified that the two published submission scalers, 10.90645 and 3.39567, are
*exactly* `y_mean[0:2]` in the benchmark's own `data_stats.npz`. The official metric is
therefore reproducible offline without ambiguity, and we reproduce it for every number
in this report.

## 3. Method, by size of contribution

### 3.1 The age-shifted anchor (largest contribution, no fitted parameters)

The competition gives, for each test site, the initial state at **all 15 seed ages** — the
site's own state-versus-forest-age curve. A stand that begins at seed age *a* and grows for
40 years finishes at forest age *a + 40*, so the answer can be read off that curve by
interpolating it at age + 40.

This alone scores **0.094**, matching the median of a 218-team field with nothing trained.
Five interpolators (linear in age, linear in log-age, monotone PCHIP, cubic spline, and the
unweighted mean of the bracketing grid ages) span 0.173–0.186 on present-day data, so
interpolation accuracy is not the binding constraint; the anchor's residual is the genuine
effect of 40 years of altered climate.

### 3.2 A one-step emulator trained through the rollout

Per site *s*, age *a*, year *t* = 1…40, input
u*ₜ* = [annual-mean forcing of year *t* (136 drivers, with the 25 shifted ones as
within-record anomalies); December state of year *t*−1 (7); *a*/500], target the December
state of year *t*. Residual MLP: ŷₜ = y*ₜ₋₁* + f(uₜ), three hidden layers of 512 with GELU,
0.6 M parameters. Trained by backpropagating through the whole 40-step unroll with teacher
forcing annealed to zero, so the training objective *is* the rollout, with the loss weighted
towards the scored final year. Evaluation rolls the model out 40 years from y₀ alone.
Validation: contiguous 10° lon/lat blocks, plus a separate score over the warmest quintile
of held-out sites as a climate-extrapolation proxy.

**Local held-out rollout MSE at year 40: 0.0451.** Alone on the leaderboard: 0.166.

A physical clamp on the fed-back state is what makes this trainable beyond 8 epochs. Without
it, training diverged to NaN reproducibly, at exactly the epoch where teacher forcing
reaches zero while the one-cycle learning rate is still near its peak: the fed-back state
leaves the range the one-step map has ever seen, and forty compositions of an unconstrained
increment blow up. Clamping the fed-back state to the range attained in the training
trajectories, widened by 25 %, and deferring the teacher-forcing anneal to mid-training,
fixes it and improves the model on its own — 0.0515 → **0.0451** at the same seed, warm
subset 0.0860 → 0.0709.

### 3.3 The blend

`0.7 × anchor + 0.3 × emulator` scores **0.076**. A parabola through the three measured
leaderboard points (0.094 at w = 0, 0.076 at w = 0.3, 0.166 at w = 1) puts the optimum at
w = 0.309, so the submitted weight is already essentially optimal.

### 3.4 Also built, for the comparisons in §4

A LightGBM direct 40-year regression: one row per (site, seed age), target the scaled
year-40 value, features = seed age, the initial state, the anchor, the full 105-value y₀
curve, and 256 aggregated forcing columns selected by gain from 1904 candidates (record
mean, four decadal means, inter-annual sd, min, max, least-squares trend, four quarterly
climatologies, seasonal amplitude — per driver) after a covariate-shift filter.

## 4. Results

| system | local | leaderboard | rank |
|---|---|---|---|
| **anchor + neural emulator, w = 0.3** | 0.0801 | **0.076** | **~71** |
| age-shifted anchor, nothing fitted | 0.1799 | 0.094 | ~115 |
| anchor + mean of the two emulators, w = 0.3 | — | 0.082 | ~78 |
| anchor + climate delta | — | 0.109 | ~130 |
| anchor + per-age affine recalibration | 0.1342 | 0.137 | ~144 |
| neural one-step emulator alone | 0.0451 | 0.166 | ~164 |
| Holographic-Reservoir one-step emulator | 0.1808 | 0.245 | ~198 |
| LightGBM direct 40-year regression | 0.0425 | 0.257 | ~205 |
| persistence, y₄₀ = y₀ | 0.2962 | 0.285 | ~211 |
| per-age constant | 1.6455 | 1.542 | ~217 |

### 4.1 Why local validation cannot be used for selection here

| system | local | leaderboard | ratio |
|---|---|---|---|
| per-age constant | 1.6455 | 1.542 | 0.94 |
| persistence | 0.2962 | 0.285 | 0.96 |
| age-shifted anchor | 0.1799 | 0.094 | 0.52 |
| reservoir emulator (rollout) | 0.1808 | 0.245 | 1.35 |
| neural emulator (rollout) | 0.0451 | 0.166 | 3.68 |
| LightGBM direct regression | 0.0425 | 0.257 | 6.05 |

The three unfitted predictors transfer almost exactly, which rules out a format,
id-alignment or scaling error and confirms the test target variance matches the present-day
variance. What degrades is anything fitted against present-day validation, in proportion to
how hard it was fitted. The extreme case: the direct regression has the anchor among its
inputs, reaches 0.0425 locally by learning corrections to it, and lands at 0.257 — **2.7×
worse than one of its own input features**, having learned to stop trusting it.

Two consequences, both measured:

* **Rollout transfers about 4× better than the direct jump.** Learning the local
  year-to-year map (1.35×–3.68×) rather than the 40-year jump (6.05×) is what buys that: the
  local map is physics that does not depend on which scenario produced the forcing. This is
  the organizers' own recommended design and our measurements support it.
* **Not even a blend weight can be chosen locally.** Present-day held-out data puts the
  optimal anchor/emulator weight at w = 0.837 — almost the pure emulator — where the
  leaderboard puts it at 0.309. Trusting the local fit would have produced something near
  0.166 instead of 0.076.

### 4.2 The structural cause

**25 of the 136 drivers lie almost entirely outside the training range on the test**
(indices 8–31, plus static channel 135): more than 90 % of test rows fall outside the
1st–99th percentile band of the training values, and for some the 40-year trend sits up to
8.8 training-spans outside. Those are exactly the channels that separate SSP126 from SSP585.

**The training set contains exactly one forcing record per site.** There is no scenario
contrast anywhere in it, so the response of the ecosystem to a CO₂ or warming *level* it
has never seen is not identifiable from the training data; only the response to
within-record variability is. We therefore feed those 25 drivers as anomalies relative to
each site's own 40-year mean, which keeps the input in range while preserving the scenario
contrast.

## 5. Reproducibility

`reproduce.sh` in the repository downloads both datasets with the official Kaggle API,
rebuilds every derived array, reruns the ablation and writes every submission candidate.
One RTX 3090 and 24 GB of host RAM; arrays are memory-mapped and streamed, so peak RAM
stays under 2 GB. `tests/test_hrc_port.py` verifies that the batched GPU reservoir
reproduces the library's own single-sequence loop to 8 × 10⁻⁸. Neither the competition data
nor the CarbonGlobe benchmark is redistributed: both are licensed for non-commercial
academic use only.

## 6. Suggestions for the organizers

1. **Publish a validation split with scenario contrast.** The benchmark as shipped makes
   the central generalisation gap invisible offline: a model can look 6× better than it is.
   A second forcing record per training site — even a short, crude one — would let
   participants measure what the competition actually tests.
2. **Ship the test sites' coordinates.** Withholding them removes every spatial method from
   consideration, which seems unintended for a benchmark advertised on a 0.5° global grid.
3. **State which month the target is.** "Targets are annual" alongside monthly storage sent
   us to the tutorial source to settle it.

---

## 7. Observations and limitations

Every item here is a real measurement, reported in full; it is placed at the end because it
bounds the result rather than constituting it.

### 7.1 Hypothesis H1 — the Holographic-Reservoir — was rejected

[`holographic-reservoir`](https://pypi.org/project/holographic-reservoir/) is this author's
own library: a leaky echo-state reservoir with a sparse recurrent matrix rescaled to
spectral radius 0.9, SHA-256-derived diagonal entropy, and a trained linear readout. In the
emulator it supplies xₜ = (1−α)x*ₜ₋₁* + α·tanh(W_in uₜ + W x*ₜ₋₁*) with W_in and W taken
unchanged from the package, and a readout ŷₜ = W_out·[xₜ; xₜ²; uₜ; 1] solved in closed form
as a ridge problem whose normal equations are accumulated in float64 over all 32 491 200
teacher-forced samples. Only the readout is trained, so the entire training set is used, in
about 50 s. The batched GPU update is a port of the library's own single-sequence loop.

It was tested in the role that fits the task best — the recurrent core of the one-step
emulator — against the neural control of §3.2, same held-out sites, same metric, matched
wall-clock budget, three seeds per arm:

| arm | MSE year 40 (mean ± sd) | warm-site subset | min/seed |
|---|---|---|---|
| Holographic-Reservoir + ridge readout (N = 801, 3 leak rates, quadratic readout) | 0.1528 ± 0.0031 | 0.2384 | 5.4 |
| neural control (residual MLP on the rollout) | **0.0593 ± 0.0091** | 0.0996 | 2.8 |
| hybrid (control + reservoir state as 300 extra inputs) | 0.0733 ± 0.0059 | 0.1294 | 3.7 |

The reservoir is 2.58× the control's MSE on its own and the intervals do not overlap. Under
the later stabilised recipe the control–hybrid gap narrows from 1.24× to 1.04× (0.0467
against 0.0451), inside the control's seed spread, so about the reservoir as an extra input
we make only the weaker claim: **it does not help, and costs 33 % more compute.** An
independent three-way least-squares fit on held-out sites assigns it a weight of 0.016
against 0.806 for the neural emulator, pointing the same way. Within its own family a
quadratic readout helped (0.1808 → 0.1638 at N = 400) and width helped (0.1560 at N = 801);
multiple leak rates alone did not (0.1762 at N = 399 against 0.1808 at N = 400 single-rate).

Caveat in the other direction: the neural arms ran 8 epochs with the training loss still
falling, and their seed-to-seed spread (±0.0091) is large next to the control–hybrid gap.
The ablation orders the three arms reliably; it does not pin the size of that gap.

### 7.2 `pixelflow` was dropped, with a measured reason

It was the planned spatial-coupling component. The 6171 test sites cannot be located on the
0.5° grid: matching on the 7 time-invariant soil channels gives a nearest-neighbour distance
of 0.091 in standardised 7-space against 0.104 for the second nearest, with not a single
exact match across 6171 sites — that is the generic nearest-neighbour spacing for 54 152
points in 7 dimensions, i.e. no identification — and static channel 135 is itself out of the
training range. Without coordinates there is no neighbourhood to couple, so no spatial model
was built and none is claimed.

### 7.3 The delta-change construction did not pay off

Rolling the emulator out under the SSP forcing and again under a stationary-climate
counterfactual (the 25 shifted drivers' anomalies zeroed), then adding the difference to the
anchor, is the standard delta-change construction in climate-impact work and should cancel
the emulator's systematic bias. It scores **0.109**, worse than the anchor alone. The learned
delta does carry the right scenario ordering (mean +0.0202 under SSP585 against +0.0110 under
SSP126, per-site sd 0.124), so the construction behaves sensibly; the emulator's climate
response is simply not accurate enough for the difference to pay for itself.

### 7.4 Emulator diversity does not help the blend

Replacing the emulator side of the blend with the mean of the neural and reservoir
emulators, at the same total share of 0.3, scores **0.082** against 0.076 for the neural
emulator alone. Their test predictions correlate at 0.915, so the diversity is real; the
reservoir simply drags the mixture down even at a 15 % effective weight. This is a fourth
independent line pointing the same way as §7.1.

### 7.5 Fitting the anchor on present-day data hurts it

A 60-parameter per-age affine recalibration takes it from 0.094 to **0.137**.

### 7.6 What we could not measure, and the risks that remain

* **Selection risk.** With no local validation that predicts the leaderboard, model
  selection is done against the public leaderboard itself. The final choice therefore
  carries public-split selection risk against the private leaderboard.
* **The emulator is not at its ceiling.** With the stabilised recipe the training loss was
  still falling at 24 epochs; more width and more epochs should improve it further.
* **No independent audit.** The external audit channel the project's own protocol requires
  was unavailable (`codex exec` fails at account level for all nine models tried), so this
  work stands as *pending independent audit*, not approved. The leaderboard scores are
  platform facts; the ablation numbers are our own runs, recorded in `experiments/`.
* **The honest comparison.** The best *single* system in this repository has no fitted
  parameters, and every trained model here — the author's own technology included — scored
  worse than it on the real test set. The entry wins only by combining the unfitted anchor
  with a trained emulator at a weight the leaderboard, not the validation set, had to choose.
