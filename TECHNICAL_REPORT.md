# When held-out validation lies: emulating ED year-40 forest state under scenario shift

**Technical report — IEEE BigData Cup 2026, AI Emulation Challenge (CarbonGlobe)**
Francisco Angulo de Lafuente (independent researcher, Madrid, Spain) · GitHub [@Agnuxo1](https://github.com/Agnuxo1)
Code: https://github.com/Agnuxo1/carbonglobe-emulator-2026 (MIT)
Version of this report: 2026-10-30. Competition open until 2026-11-16 05:00 UTC.

---

## 1. Task and metric

For 6171 globally distributed sites and 15 seed forest ages, under SSP126 and SSP585
forcing, forecast the December state of year 40 for canopy height and aboveground
biomass, each divided by its global mean. Score: MSE over the 185 130 submitted values.

We first pinned down the target convention, because the public description leaves room
for doubt ("targets are annual"). The organizers' own tutorial indexes
`glob_Y{age}.npy[s, year, 11, :]` and states that December is used for Y in all
experiments; year 40 is index 40 of the 41 stored years, index 0 being the initial
state. We also verified that the two published submission scalers, 10.90645 and
3.39567, are *exactly* `y_mean[0:2]` in the benchmark's own `data_stats.npz`. The
official metric is therefore reproducible offline without ambiguity, and we reproduce
it for every number in this report.

## 2. The central difficulty: there is no scenario variation to learn from

Training data is the present-day CarbonGlobe benchmark: 54 152 sites, one 40-year
monthly record of 136 drivers per site, and for each of 15 seed ages the 41-year
trajectory of 7 ecosystem variables. The test is the same simulator driven by future
forcing.

Two measurements make the difficulty concrete.

**(a) A quarter of the drivers are out of range on the test.** For 25 of the 136
drivers (indices 8–31, plus static channel 135), more than 90 % of test rows fall
outside the 1st–99th percentile band of the training values; for some, the 40-year
trend sits up to 8.8 training-spans outside. These are precisely the channels that
separate SSP126 from SSP585. A tree model cannot extrapolate in them at all, and a
neural model extrapolates without evidence.

**(b) The training set contains exactly one forcing record per site.** There is no
scenario contrast anywhere in it. The response of the ecosystem to a CO₂ or warming
*level* it has never seen is therefore not identifiable from the training data; only
the response to within-record variability is. We consequently feed those 25 drivers as
anomalies relative to each site's own 40-year mean, which keeps the model's input in
range while preserving the scenario contrast.

**(c) The test sites cannot be located.** The competition ships the site axis as a bare
0–6170 index with no coordinates. Matching on the 7 time-invariant soil channels gives
a nearest-neighbour distance of 0.091 in standardised 7-space against 0.104 for the
second nearest, with not a single exact match across 6171 sites — that is the generic
nearest-neighbour spacing for 54 152 points in 7 dimensions, i.e. no identification.
Static channel 135 is itself out of the training range. We therefore have no latitude,
longitude or neighbourhood for any test site, which rules out every spatial model.

## 3. The strongest system has no fitted parameters

The competition gives, for each test site, the initial state at **all 15 seed ages**:
the site's own state-versus-forest-age curve. A stand that begins at seed age *a* and
grows for 40 years finishes at forest age *a + 40*. The answer can simply be read off
that curve:

> **age-shifted anchor** — interpolate the site's own y₀-versus-age curve at age + 40.

This scores **0.094** on the public leaderboard with nothing fitted, against a field
whose median over 218 teams is 0.092 and whose leader is at 0.037. We compared five
interpolators (linear in age, linear in log-age, monotone PCHIP, cubic spline, and the
unweighted mean of the bracketing grid ages); they span 0.173–0.186 on present-day
data, so interpolation accuracy is not the binding constraint. The anchor's residual is
the genuine effect of 40 years of altered climate.

## 4. Every trained model we built scored worse, and the gap grows with capacity

| system | local held-out | public leaderboard | ratio |
|---|---|---|---|
| per-age constant | 1.6455 | 1.542 | 0.94 |
| persistence, y₄₀ = y₀ | 0.2962 | 0.285 | 0.96 |
| age-shifted anchor | 0.1799 | **0.094** | **0.52** |
| Holographic-Reservoir one-step emulator, 40-year rollout | 0.1808 | 0.245 | 1.35 |
| LightGBM direct 40-year regression | 0.0425 | 0.257 | **6.05** |

Local validation here is held-out by contiguous 10° lon/lat blocks, so it is not naive
random splitting. Yet the ordering it produces is inverted on the test set.

The three unfitted predictors transfer almost exactly — which rules out a format, id
alignment or scaling error, and shows the test target variance matches the present-day
variance. What degrades is anything fitted against present-day validation, and it
degrades in proportion to capacity. The direct gradient-boosted regression is the worst
case: given the anchor among its inputs, it reaches 0.0425 locally by learning
corrections to it, and on the test it lands at 0.257 — **2.7× worse than one of its own
input features.** It learned to stop trusting the anchor.

A one-step autoregressive emulator, rolled out 40 years, degrades by 1.35× instead of
6×. Learning the local year-to-year map rather than the 40-year jump is what buys that:
the local map is physics that does not depend on which scenario produced the forcing.
This is the organizers' own recommended design, and our measurements support it.

## 5. Methods

**Direct regression (control for §4).** LightGBM, one row per (site, seed age), target
the scaled year-40 value. Features: seed age, the initial state, the age-shifted
anchor, the full 105-value y₀ curve, and 256 aggregated forcing columns selected by
gain from 1904 candidates (record mean, four decadal means, inter-annual sd, min, max,
least-squares trend, four quarterly climatologies, seasonal amplitude — per driver),
after discarding columns failing the covariate-shift filter of §2(a).

**One-step emulators.** Per site *s*, age *a*, year *t* = 1…40, input
u*ₜ* = [annual-mean forcing of year *t* (136, with the 25 shifted drivers as
within-record anomalies); December state of year *t*−1 (7); *a*/500], target the
December state of year *t*. Trained with teacher forcing, evaluated by rolling the
model out for 40 years from y₀ alone. Validation: the same 10° spatial blocks, plus a
separate score over the warmest quintile of held-out sites as a climate-extrapolation
proxy.

- *Holographic-Reservoir core.* xₜ = (1−α)x*ₜ₋₁* + α·tanh(W_in uₜ + W x*ₜ₋₁*), with
  W_in and W taken unchanged from the `holographic-reservoir` package (sparse recurrent
  matrix rescaled to spectral radius 0.9, SHA-256-derived diagonal entropy). Readout
  ŷₜ = W_out·[xₜ; xₜ²; uₜ; 1], solved in closed form as a ridge problem whose normal
  equations are accumulated in float64 over all 32 491 200 teacher-forced samples. Only
  the readout is trained, so the entire training set is used, in about 50 s.
- *Neural control.* Residual MLP, ŷₜ = y*ₜ₋₁* + f(uₜ), three hidden layers of 512 with
  GELU, 0.6 M parameters. Trained by backpropagating through the whole 40-step unroll
  with teacher forcing annealed to zero, so the objective is the rollout itself, with
  the loss weighted towards the scored final year.

The reservoir's batched GPU update is a port of the library's own single-sequence loop;
`tests/test_hrc_port.py` verifies agreement to 8 × 10⁻⁸.

## 6. Ablation: the contribution of the reservoir, measured

Three seeds per arm, same held-out sites, same metric, matched wall-clock budget.

| arm | MSE year 40 (mean ± sd) | warm-site subset | min/seed |
|---|---|---|---|
| Holographic-Reservoir + ridge readout (N = 801, 3 leak rates, quadratic readout) | 0.1528 ± 0.0031 | 0.2384 | 5.4 |
| **neural control** (residual MLP on the rollout) | **0.0593 ± 0.0091** | 0.0996 | 2.8 |
| hybrid (control + reservoir state as extra input) | 0.0733 ± 0.0059 | 0.1294 | 3.7 |

The reservoir is 2.58× the control's MSE on its own, and supplying its state to the
control as 300 extra inputs makes the control 1.24× *worse* while costing 28 % more
time. The intervals do not overlap. Within the reservoir family, a quadratic readout
helped (0.1808 → 0.1638 at N = 400) and width helped (0.1560 at N = 801); multiple leak
rates alone did not (0.1762 at N = 399 against 0.1808 at N = 400 single-rate).

**The reservoir hypothesis is rejected on this task.** What it does deliver is training
cost — a closed-form readout over 32.5 M samples in under a minute — and that is not
enough to compensate here. We report this because a negative result from a matched
ablation is worth more than a positive one from an unmatched comparison.

*Caveat we will not hide:* the neural arms ran 8 epochs with the training loss still
falling, and their seed-to-seed spread (±0.0091) is large next to the control-hybrid
gap. The ablation orders the three arms reliably; it does not pin the size of the gap
between the two neural arms.

## 7. Delta-change construction

Since the anchor supplies a well-calibrated level while the emulator generalises poorly
in absolute terms, we use the emulator only for the difference. Roll it out under the
SSP forcing; roll it out again with the 25 shifted drivers' anomalies set to zero, a
stationary-climate counterfactual; add the difference to the anchor:

> ŷ₄₀ = anchor + [ rollout(SSP forcing) − rollout(stationary counterfactual) ]

The emulator's systematic bias largely cancels in the difference, which is the standard
delta-change construction in climate-impact work. The learned delta carries the
expected scenario ordering: mean +0.0058 under SSP585 against −0.0009 under SSP126,
with a per-site standard deviation of 0.276, i.e. small in the mean and strongly
spatially structured. Its leaderboard score is pending at the time of writing.

## 8. What we would tell the organizers

1. **Publish a validation split with scenario contrast.** The benchmark as shipped
   makes the central generalisation gap invisible offline: a model can look 6× better
   than it is. A second forcing record per training site — even a short, crude one —
   would let participants measure what the competition actually tests.
2. **Ship the test sites' coordinates.** Withholding them removes every spatial method
   from consideration, which seems unintended for a benchmark advertised on a 0.5°
   global grid.
3. **Say which month the target is.** "Targets are annual" alongside monthly storage
   sent us to the tutorial source to settle it.

## 9. Honest summary

The best system we measured on this task has no fitted parameters: read each site's own
state-versus-age curve at age + 40, and score 0.094. Every trained model we built —
including the author's own reservoir-computing technology, tested in the role a
structured decision procedure selected for it — scored worse than that on the real test
set. The useful finding is the mechanism: on this benchmark, held-out validation on the
present-day distribution does not rank models for a future-climate test, and
rollout-based formulations degrade roughly four times less than direct end-of-horizon
regression.

### Reproducibility

`reproduce.sh` in the repository downloads both datasets with the official Kaggle API,
rebuilds every derived array, reruns the ablation and writes every submission
candidate. One RTX 3090 and 24 GB of host RAM; arrays are memory-mapped and streamed,
so peak RAM stays under 2 GB. Neither the competition data nor the CarbonGlobe
benchmark is redistributed: both are licensed for non-commercial academic use only.
