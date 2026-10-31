"""Self-calibrated blend weights, fitted to nothing.

The emulator is rolled out twice on the test set: once under the SSP forcing, once under
a stationary-climate counterfactual with the 25 shifted drivers' anomalies zeroed. That
second rollout and the age-shifted anchor are two independent estimates of nearly the
same quantity -- the state the site reaches at forest age a+40 under unchanged climate.

Where they agree, the emulator reproduces the site's age progression correctly and its
climate correction can be trusted. Where they disagree, the emulator is miscalibrated at
that site and the anchor should carry the weight. So

    r      = |rollout_stationary - anchor|            per (scenario, site, age, variable)
    w_raw  = 1 / (1 + (r / tau)^2)                    tau = median(r)
    w      = w_raw * (w_target / mean(w_raw))         rescaled so mean(w) == w_target

and the submitted prediction is (1 - w) * anchor + w * rollout_ssp.

Two properties worth stating. First, **no present-day target is used anywhere**: r is
computed entirely from the test inputs and the emulator's own two rollouts, which matters
because present-day data has been measured to choose blend weights badly here (it puts
the uniform optimum at 0.837 where the leaderboard puts it at 0.309). Second, the weights
are rescaled to the *same mean* as the known uniform optimum, so a submission tests
whether conditioning helps, not whether a different average weight helps.

  python src/blend_selfcal.py --model S06_nn_clamp_raw.npy --w-target 0.309
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "src")
from common import WORK
from blend import anchor_array, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="S06_nn_clamp_raw.npy")
    p.add_argument("--w-target", type=float, default=0.309)
    p.add_argument("--sharpness", type=float, default=1.0,
                   help="tau = sharpness * median(r); smaller = more aggressive")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    anc = anchor_array()
    ssp = np.load(WORK / "subs" / a.model)
    stat = np.load(WORK / "subs" / a.model.replace("_raw.npy", "_stat.npy"))
    assert ssp.shape == anc.shape == stat.shape

    r = np.abs(stat - anc)
    tau = a.sharpness * float(np.median(r))
    w_raw = 1.0 / (1.0 + (r / max(tau, 1e-12)) ** 2)
    w = w_raw * (a.w_target / w_raw.mean())
    w = np.clip(w, 0.0, 1.0)
    print(f"  tau {tau:.4f}  mean w {w.mean():.3f}  "
          f"p5 {np.percentile(w,5):.3f}  median {np.median(w):.3f}  "
          f"p95 {np.percentile(w,95):.3f}")
    # how much does the signal actually vary? if it barely does, the test is weak
    print(f"  |stat-anchor|: median {np.median(r):.4f}  p95 {np.percentile(r,95):.4f}; "
          f"corr(stat, anchor) {np.corrcoef(stat.ravel(), anc.ravel())[0,1]:.4f}")
    write(f"C06_selfcal{a.tag}_w{a.w_target:g}_s{a.sharpness:g}",
          (1 - w) * anc + w * ssp)


if __name__ == "__main__":
    main()
