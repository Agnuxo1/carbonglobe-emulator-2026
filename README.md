# CarbonGlobe year-40 emulation — IEEE BigData Cup 2026

**Public leaderboard score 0.076 MSE — about rank 71 of 218 teams, against a field median
of 0.092.** Emulating a process-based global land-ecosystem model (Ecosystem Demography,
ED) to forecast, for 6171 forest sites and 15 seed forest ages under two future emission
scenarios, the canopy height and aboveground biomass 40 years out.

Entry for the [IEEE BigData Cup 2026 AI Emulation Challenge](https://www.kaggle.com/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge),
built on the CarbonGlobe benchmark (Wang et al., NeurIPS 2025). Metric: MSE on each
variable divided by its global mean. Author: Francisco Angulo de Lafuente
([@Agnuxo1](https://github.com/Agnuxo1)). Licence: MIT. No competition data is
redistributed here — see [ATTRIBUTION.md](ATTRIBUTION.md).
See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the full write-up.

| | MSE | approx. rank |
|---|---|---|
| **this entry** (anchor + neural emulator, w = 0.3) | **0.076** | **~71 / 218** |
| field median | 0.092 | 109 |
| competition leader | 0.037 | 1 |
| persistence (`y₄₀ = y₀`) | 0.285 | 211 |
| per-age constant | 1.542 | 217 |

Competition open until 2026-11-16 05:00 UTC.

---

## How it works

Three ingredients, listed by how much each contributes.

### 1. The age-shifted anchor — the backbone, and it is free

The competition hands you, for every test site, the initial ecosystem state at **all 15
seed ages**: the site's own state-versus-forest-age curve. A stand that begins at seed age
*a* and grows for 40 years finishes at forest age *a + 40*. So the answer can be **read off
that curve**, with no model and no fitted parameter:

```python
anchor = interp_over_log_age(y0_curve, at=age + 40)      # src/dataset.py
```

**0.094 on the leaderboard, zero training**, which on its own already matches the median of
a 218-team field. It is the single largest contribution in this entry, and it is the
reason the final system is a *correction to* the anchor rather than a replacement for it.

Five interpolators were compared (`src/anchor_variants.py`) and they land between 0.173
and 0.186 on present-day data, so interpolation accuracy is not the binding constraint:
the anchor's residual is the genuine effect of 40 years of altered climate.

### 2. A one-step emulator trained through the rollout — the correction

Per site, seed age and year, the model maps [annual-mean forcing of that year (136 drivers)
; December state of the previous year (7) ; seed age / 500] to the next December state, and
is rolled out 40 years from y₀ alone. Two choices matter:

* **Residual parameterisation**: the network predicts the annual increment, not the state.
* **Trained on the rollout, not one step ahead.** Teacher forcing is annealed to zero and
  gradients are backpropagated through the whole 40-step unroll, with the loss weighted
  towards the scored final year. The organizers' tutorial warns that long-horizon drift is
  what separates emulators; this optimises against it directly.

Local held-out rollout MSE at year 40: **0.0451**. On the leaderboard, alone, 0.166.

* **A physical clamp on the fed-back state** (`state_bounds`) is what makes it trainable at
  all beyond 8 epochs. Without it, training diverged to NaN reproducibly, at exactly the
  epoch where teacher forcing reaches zero while the one-cycle learning rate is still near
  its peak: the fed-back state leaves the range the one-step map has ever seen and forty
  compositions of an unconstrained increment blow up. Clamping fixes that and improves the
  model on its own, 0.0515 → **0.0451** at the same seed, warm-site subset 0.0860 → 0.0709.

### 3. The blend — 19 % off the anchor's score

`0.7 × anchor + 0.3 × emulator` scores **0.076**, better than either part (0.094 and
0.166). The two error structures are complementary enough that the mixture wins. A parabola
through the three measured leaderboard points puts the optimum at w = 0.309, so the
submitted weight is already essentially optimal and re-tuning it buys nothing.

## Results

`local` is held-out validation on the present-day training distribution, by contiguous 10°
lon/lat blocks; `leaderboard` is the competition's public score on the real future-climate
test set.

| system | local | leaderboard | rank |
|---|---|---|---|
| **anchor + neural emulator, w = 0.3** | 0.0801 | **0.076** | **~71** |
| age-shifted anchor, nothing fitted | 0.1799 | 0.094 | ~115 |
| anchor + mean of the two emulators, w = 0.3 | — | 0.082 | ~78 |
| anchor + climate delta (delta-change construction) | — | 0.109 | ~130 |
| anchor + per-age affine recalibration (60 params) | 0.1342 | 0.137 | ~144 |
| neural one-step emulator alone, 40-year rollout | 0.0451 | 0.166 | ~164 |
| Holographic-Reservoir one-step emulator, 40-year rollout | 0.1808 | 0.245 | ~198 |
| LightGBM direct 40-year regression | 0.0425 | 0.257 | ~205 |
| persistence, y₄₀ = y₀ | 0.2962 | 0.285 | ~211 |
| per-age constant | 1.6455 | 1.542 | ~217 |

### The methodological result: this benchmark's held-out validation does not rank models

This is the finding most likely to be useful to anyone else entering the competition, and
it is measured, not argued. Local present-day validation and the leaderboard disagree by a
factor that depends on how hard the system was fitted:

| system | local | leaderboard | ratio |
|---|---|---|---|
| per-age constant | 1.6455 | 1.542 | 0.94 |
| persistence | 0.2962 | 0.285 | 0.96 |
| age-shifted anchor | 0.1799 | 0.094 | **0.52** |
| reservoir emulator (rollout) | 0.1808 | 0.245 | 1.35 |
| neural emulator (rollout) | 0.0451 | 0.166 | 3.68 |
| LightGBM direct regression | 0.0425 | 0.257 | **6.05** |

Unfitted predictors transfer almost exactly, which rules out a format, id-alignment or
scaling error and confirms the test target variance matches the present-day variance. What
degrades is anything fitted against present-day validation, in proportion to how hard it
was fitted. Two consequences we measured rather than guessed:

* **Rollout beats the direct jump by 4× in transfer.** A one-step map learns local physics
  that does not depend on which scenario produced the forcing (1.35×–3.68×); a direct
  40-year regression learns a climate-specific shortcut (6.05×).
* **Not even blend weights can be chosen locally.** Present-day data puts the optimal
  anchor/emulator weight at w = 0.837, almost the pure emulator; the leaderboard puts it at
  0.309. Had we trusted the local fit we would have submitted something near 0.166 instead
  of 0.076 (`src/blend_weights.py`).

The structural reason is visible in the data. **25 of the 136 drivers lie almost entirely
outside the training range on the test** (indices 8–31, plus static channel 135: more than
90 % of test rows outside the training 1–99 percentile band, with trends up to 8.8
training-spans outside) — and those are precisely the channels that separate SSP126 from
SSP585. Meanwhile the training set holds **exactly one forcing record per site**, so it
contains no scenario variation at all from which their effect could be learned. We feed
them as within-record anomalies instead (`src/shift_report.py`).

## Reproducing

```bash
pip install -r requirements.txt
pip install holographic-reservoir          # the author's reservoir library, tested below
DATA=/some/disk/with/35GB bash reproduce.sh
```

`reproduce.sh` downloads both datasets with the official Kaggle API (you need a token and
to have accepted the competition rules), builds the derived arrays, reruns the ablation and
writes every submission candidate. A GPU is needed for the emulators; the anchor and the
probes are pure NumPy. `tests/test_hrc_port.py` checks that the batched GPU reservoir
reproduces the library's own `_collect_states` to 8 × 10⁻⁸.

Hardware used: one RTX 3090, 24 GB host RAM. Every array is memory-mapped and streamed in
chunks, so peak RAM stays under about 2 GB. Every number in this README comes from a run
recorded in `experiments/`.

---

## Observations and limitations

Everything here is a real measurement and none of it is softened; it is at the end because
it is the limiting detail, not the result.

**Hypothesis H1 — the Holographic-Reservoir — was rejected.**
[`holographic-reservoir`](https://pypi.org/project/holographic-reservoir/) is this author's
own library: a leaky echo-state reservoir with a sparse recurrent matrix rescaled to a
target spectral radius, SHA-256-derived diagonal entropy, and a trained linear readout. It
was tested in the role that fits the task best — the recurrent core of the one-step
emulator — against the neural control, on the same held-out sites, the same metric and a
matched budget, three seeds per arm:

| arm | MSE year 40 (mean ± sd, 3 seeds) | warm sites | min/seed |
|---|---|---|---|
| Holographic-Reservoir + ridge readout (N = 801, 3 leak rates, quadratic readout) | 0.1528 ± 0.0031 | 0.2384 | 5.4 |
| neural control (residual MLP on the rollout) | **0.0593 ± 0.0091** | 0.0996 | 2.8 |
| hybrid (control + reservoir state as 300 extra inputs) | 0.0733 ± 0.0059 | 0.1294 | 3.7 |

The reservoir is 2.58× the control's MSE on its own, and the intervals do not overlap. Two
caveats in both directions: the neural arms ran 8 epochs with the loss still falling, so the
ablation orders the arms reliably but does not pin the gap size; and under the later
stabilised recipe the control–hybrid gap narrows from 1.24× to 1.04× (0.0467 against
0.0451), inside the control's seed spread, so the defensible claim about the reservoir as an
extra input is the weaker one — **it does not help, and costs 33 % more time** — not that it
clearly hurts. An independent three-way least-squares fit on held-out sites assigns the
reservoir a weight of 0.016 against 0.806 for the neural emulator, which points the same way.

What the reservoir does deliver is training cost: only the readout is fitted, in closed form
from all 32 491 200 teacher-forced samples in about 50 s. That is real, and it is not enough
to make it competitive here. Within its own family a quadratic readout helped (0.1808 →
0.1638 at N = 400) and width helped (0.1560 at N = 801); multiple leak rates alone did not.

**`pixelflow` was dropped, with a measured reason.** It was the planned spatial-coupling
component, but the 6171 test sites cannot be located on the 0.5° grid: matching on the 7
static soil channels gives a nearest-neighbour distance of 0.091 against 0.104 for the
second nearest, with not one exact match across 6171 sites — the generic nearest-neighbour
spacing for 54 152 points in 7 dimensions, i.e. no identification — and static channel 135
is itself out of the training range (`src/locate_test_sites.py`). Without coordinates there
is no neighbourhood to couple, so no spatial model was built and none is claimed.

**The delta-change construction did not work.** Rolling the emulator out under the SSP
forcing and again under a stationary-climate counterfactual (the 25 shifted drivers'
anomalies zeroed), then adding the difference to the anchor, scores **0.109** — worse than
the anchor alone. The learned delta does carry the right scenario ordering (mean +0.0202
under SSP585 against +0.0110 under SSP126, per-site sd 0.124) and the idea is standard in
climate-impact work, but on this task the emulator's climate response is not accurate
enough for the difference to pay for itself.

**Emulator diversity does not help the blend.** Replacing the emulator side with the mean
of the neural and reservoir emulators, at the same total share of 0.3, scores **0.082**
against 0.076 for the neural emulator alone: the reservoir drags the mixture down even at a
15 % effective weight. Their predictions correlate at 0.915, so the diversity is real; it
simply is not useful diversity.

**Fitting the anchor on present-day data hurts it**, even minimally: a 60-parameter per-age
affine recalibration takes it from 0.094 to **0.137**.

**What we could not measure.** No local validation here predicts the leaderboard, so model
selection is done against the leaderboard itself with a small number of deliberately
distinct candidates — which means the final choice carries public-leaderboard selection
risk against the private split. The emulator is not at its ceiling: with the stabilised
recipe the training loss was still falling at 24 epochs. And **this work has not been
independently audited**: the external audit channel the house protocol requires was
unavailable (`codex exec` fails at account level), so the project stands as *pending
independent audit*, not approved.

**What the comparison in §Results is worth remembering for.** The best *single* system in
this repository has no fitted parameters at all, and every trained model here — the
author's own technology included — scored worse than it on the real test set. The entry
wins only by combining the unfitted anchor with a trained emulator at a weight the
leaderboard, not the validation set, had to choose.
