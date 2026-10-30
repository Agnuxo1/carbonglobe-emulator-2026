# CarbonGlobe year-40 emulation — IEEE BigData Cup 2026

Emulating a process-based global land-ecosystem model (Ecosystem Demography, ED) well
enough to forecast, for 6171 forest sites and 15 seed forest ages under two future
emission scenarios, the **canopy height and aboveground biomass 40 years out**.

Entry for the [IEEE BigData Cup 2026 AI Emulation Challenge](https://www.kaggle.com/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge),
built on the CarbonGlobe benchmark (Wang et al., NeurIPS 2025). Metric: MSE on each
variable divided by its global mean. Author: Francisco Angulo de Lafuente
([@Agnuxo1](https://github.com/Agnuxo1)). Licence: MIT. No competition data is
redistributed here — see [ATTRIBUTION.md](ATTRIBUTION.md).

---

## The result in one table

Everything below is a measured number. `local` is held-out validation on the
present-day training distribution; `leaderboard` is the competition's public score on
the real future-climate test set. **The two disagree violently, and that disagreement
is the main finding of this work.**

| system | local | leaderboard | notes |
|---|---|---|---|
| per-age constant | 1.6455 | **1.542** | null reference |
| persistence, `y40 = y0` | 0.2962 | **0.285** | |
| **age-shifted anchor, nothing fitted** | 0.1799 | **0.094** | **best system measured** |
| LightGBM direct 40-year regression | 0.0425 | **0.257** | 6.0x worse than its own validation |
| Holographic-Reservoir one-step emulator, 40-year rollout | 0.1808 | **0.245** | 1.35x |
| neural one-step emulator, 40-year rollout | 0.0593 ± 0.0091 | pending | |

For scale: the public leaderboard over 218 teams runs from 0.037 (first) through a
median of 0.092 to 0.24 at the 90th percentile.

## What actually works: the age-shifted anchor

The competition hands you, for every test site, the initial ecosystem state at **all
15 seed ages** — the site's own state-versus-forest-age curve. A forest that starts at
seed age *a* and grows for 40 years ends at forest age *a+40*. So the answer can simply
be **read off that curve at age a+40**, with no model and no fitted parameter:

```python
anchor = interp_over_log_age(y0_curve, at=age + 40)      # src/dataset.py
```

That scores **0.094**, around the median of a 218-team field, with zero training.
Five interpolators were compared (`src/anchor_variants.py`); they land between 0.173
and 0.186 locally, so the interpolation is *not* the bottleneck — the residual is the
genuine effect of 40 years of changed climate.

## What does not work, and why it matters

A gradient-boosted direct regression on (seed age, initial state, the whole y0 curve,
256 aggregated climate features) reaches **0.0425** in spatial-block cross-validation
and **0.257** on the leaderboard. It ends up **2.7x worse than one of its own input
features**: given the anchor as an input, it learned to ignore it in favour of
present-day coincidences that do not survive the change of climate.

Three deterministic probes pin the blame on the model rather than on the submission
pipeline — a per-age constant (1.6455 local / 1.542 leaderboard), persistence (0.2962 /
0.285) and the anchor (0.1799 / 0.094) all transfer almost exactly, and the anchor
transfers *better* than it validates. So the format, the scaling and the test variance
are all as expected; what fails is anything fitted against present-day validation, and
it fails in proportion to its capacity.

The structural reason is visible in the data: **25 of the 136 drivers lie almost
entirely outside the training range on the test** (indices 8–31 and 135; more than 90 %
of test rows outside the training 1–99 percentile band, with trends up to 8.8
training-spans outside). Those are exactly the channels that distinguish SSP126 from
SSP585 — and the training set contains a single forcing record per site, so it holds
**no scenario variation at all** from which their effect could be learned. They are fed
as within-record anomalies instead (`src/shift_report.py`, `src/hrc_onestep.py`).

Consequence for anyone working on this benchmark: present-day held-out validation does
not rank models for the future-climate test. Rollout-based validation is far better
behaved (1.35x versus 6.0x) but still not calibrated.

## Hypothesis H1: the Holographic-Reservoir, and its measured contribution

[`holographic-reservoir`](https://pypi.org/project/holographic-reservoir/) is this
author's own reservoir-computing library: a leaky echo-state reservoir with a sparse
recurrent matrix rescaled to a target spectral radius, SHA-256-derived entropy on the
diagonal, and a trained linear readout. H1 was that it would beat a professional
control at emulating the ecosystem dynamics.

It was tested in the role that fits the task best — the **recurrent core of the one-step
emulator** — against a residual MLP trained through the 40-year rollout, on the same
held-out sites, the same metric and a matched budget. Three seeds per arm:

| arm | MSE year 40 (mean ± sd, 3 seeds) | warm sites | min/seed |
|---|---|---|---|
| Holographic-Reservoir + ridge readout (N=801, 3 leak rates, quadratic readout) | 0.1528 ± 0.0031 | 0.2384 | 5.4 |
| **neural control** (residual MLP, trained on the rollout) | **0.0593 ± 0.0091** | 0.0996 | 2.8 |
| hybrid (control + reservoir state as extra input) | 0.0733 ± 0.0059 | 0.1294 | 3.7 |

**H1 is rejected.** The reservoir is 2.58x the control's MSE on its own, and feeding its
state to the control as extra input makes the control 1.24x *worse* while costing 28 %
more time. The error bars do not overlap. Within the reservoir family the useful
ingredients were a quadratic readout (0.1808 → 0.1638) and more units (0.1560 at
N=801); multiple leak rates alone bought almost nothing.

What the reservoir does deliver is training cost: only the readout is fitted, in closed
form from all 32.5 million teacher-forced samples in about 50 seconds. That is a real
property, and it is not enough to make it competitive here.

**`pixelflow` was dropped with a measured reason, not an excuse.** It was the planned
spatial-coupling component, but the 6171 test sites cannot be located on the 0.5°
grid: matching on the 7 static soil channels gives a nearest-neighbour distance (0.091)
essentially equal to the second-nearest (0.104) with not one exact match, and static
channel 135 is itself out of the training range (`src/locate_test_sites.py`). Without
coordinates there is no neighbourhood to couple, so no spatial model was built and none
is claimed.

## The climate-delta construction

Since the anchor supplies a good level and the emulator generalises poorly in absolute
terms, the emulator is used only for the **difference**: roll it out under the SSP
forcing, roll it out again with the 25 shifted drivers' anomalies zeroed (a
stationary-climate counterfactual), and add the difference to the anchor.

```
y40 ≈ anchor + ( rollout(SSP forcing) − rollout(stationary counterfactual) )
```

The emulator's systematic bias largely cancels in the difference. This is the standard
"delta change" construction from climate-impact studies. The learned delta has the
expected sign: mean +0.0058 for SSP585 against −0.0009 for SSP126, with a per-site
standard deviation of 0.276. Its leaderboard score is pending.

## Reproducing

```bash
pip install -r requirements.txt
pip install holographic-reservoir          # the technology under test
DATA=/some/disk/with/35GB bash reproduce.sh
```

`reproduce.sh` downloads both datasets with the official Kaggle API (you need a token
and to have accepted the competition rules), builds the derived arrays, reruns the H1
ablation and writes every submission candidate. A GPU is needed for the emulators; the
anchor and the probes are pure NumPy. `tests/test_hrc_port.py` checks that the batched
GPU reservoir reproduces the library's own `_collect_states` to 8e-8 — the only thing
ported is the loop order.

Hardware used: one RTX 3090, 24 GB host RAM. Every array is memory-mapped and streamed
in chunks, so peak RAM stays under about 2 GB.

## Status and limits

- **Official status:** submitted and scored, competition open until 2026-11-16 05:00 UTC.
  Best public score so far **0.094**, from the untrained anchor. Not a prize position:
  the leader is at 0.037.
- The ablation arms run 8 epochs and are **not converged** — the training loss was still
  falling — and the seed-to-seed spread (±0.0091) is large next to the gap between the
  neural arms. The ablation orders the arms reliably; it does not measure the gap
  precisely.
- No local validation here predicts the leaderboard. Model selection for the final
  submissions is therefore done against the leaderboard itself, with a small number of
  deliberately distinct candidates.
- The one number that matters for honesty: **the best system in this repository has no
  fitted parameters.** Every trained model built here, including the author's own
  technology, scored worse than it on the real test set.
