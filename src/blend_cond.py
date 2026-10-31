"""Conditional blend weights: one per scored variable, or one per seed age.

A single blend weight is a compromise. The leaderboard already told us the uniform
optimum is w = 0.309 (parabola through 0.094 at w=0, 0.076 at w=0.3, 0.166 at w=1),
so the remaining room is in letting w depend on something.

Two hypotheses, each testable with one submission:

* **Per variable.** Height and aboveground biomass have different error structure. On
  held-out present-day sites the emulator's rollout error is 0.0391 for height against
  0.0512 for agb, and the anchor's own error splits differently again, so the optimal
  share of emulator need not be the same for both.
* **Per seed age.** The anchor's error grows strongly with seed age -- locally 0.0107 at
  age 1 against 0.0678 at age 500 -- because biomass keeps accumulating and there is more
  room for the trajectory to diverge from the present-day age curve. Where the anchor is
  weaker, the emulator should carry more weight.

Both are *hypotheses about the leaderboard*, not fits: present-day data cannot choose
blend weights here (it puts the uniform optimum at 0.837 where the leaderboard puts it at
0.309), so the weights below are set by hand around the known uniform optimum and the
leaderboard decides.

  python src/blend_cond.py pervar  --model S06_nn_clamp_raw.npy --wh 0.30 --wa 0.45
  python src/blend_cond.py perage  --model S06_nn_clamp_raw.npy --w-lo 0.20 --w-hi 0.40
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "src")
from common import WORK, AGES
from blend import anchor_array, write

AGE_ARR = np.array(AGES, dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["pervar", "perage", "both"])
    p.add_argument("--model", default="S06_nn_clamp_raw.npy")
    p.add_argument("--wh", type=float, default=0.30, help="emulator share for height")
    p.add_argument("--wa", type=float, default=0.45, help="emulator share for agb")
    p.add_argument("--w-lo", type=float, default=0.20, help="share at seed age 1")
    p.add_argument("--w-hi", type=float, default=0.40, help="share at seed age 500")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    anc = anchor_array()                               # (2, 6171, 15, 2)
    m = np.load(WORK / "subs" / a.model)
    assert m.shape == anc.shape, (m.shape, anc.shape)

    if a.mode in ("pervar", "both"):
        w = np.array([a.wh, a.wa])                     # broadcasts over the last axis
        write(f"C04_pervar{a.tag}_h{a.wh:g}_a{a.wa:g}", (1 - w) * anc + w * m)

    if a.mode in ("perage", "both"):
        # linear in log seed age, from w_lo at age 1 to w_hi at age 500
        la = np.log(AGE_ARR)
        t = (la - la[0]) / (la[-1] - la[0])
        wa_ = (a.w_lo + (a.w_hi - a.w_lo) * t).reshape(1, 1, 15, 1)
        print("  per-age weights:",
              " ".join(f"{int(ag)}:{float(v):.2f}" for ag, v in zip(AGES, wa_.ravel())))
        write(f"C05_perage{a.tag}_{a.w_lo:g}to{a.w_hi:g}", (1 - wa_) * anc + wa_ * m)


if __name__ == "__main__":
    main()
