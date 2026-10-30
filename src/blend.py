"""Build submission candidates around the age-shifted anchor.

The anchor -- the site's own initial-state-versus-age curve read at age+40 -- is the
strongest single predictor measured on the leaderboard (0.094, with no fitting at
all), while every model trained against present-day validation scored worse than it.
So the anchor is the baseline to beat, and candidates are built as corrections to it
rather than as replacements.

  python src/blend.py anchor_affine
  python src/blend.py blend --model S05_nn_raw.npy --weights 0.2,0.35,0.5,0.65
"""
import argparse, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from common import WORK, COMP, SCENARIOS, AGES, SUB_SCALE
from dataset import age_shift_anchor

OUT = WORK / "subs"; OUT.mkdir(parents=True, exist_ok=True)
TMPL = pd.read_csv(COMP / "sample_submission.csv")


def write(name, arr):
    """arr: (2, 6171, 15, 2) already divided by SUB_SCALE."""
    rows = []
    for si, scn in enumerate(SCENARIOS):
        n = arr.shape[1]
        rows.append(pd.DataFrame({
            "id": [f"{scn}_{i}_{ag}" for i in range(n) for ag in AGES],
            "height": arr[si, :, :, 0].ravel(), "agb": arr[si, :, :, 1].ravel()}))
    sub = pd.concat(rows, ignore_index=True)
    sub = TMPL[["id"]].merge(sub, on="id", how="left")
    assert sub[["height", "agb"]].notna().all().all(), "unmatched ids"
    sub[["height", "agb"]] = sub[["height", "agb"]].clip(lower=0.0)
    sub.to_csv(OUT / f"{name}.csv", index=False)
    print(f"  {name}: height {sub.height.mean():.4f} agb {sub.agb.mean():.4f}")


def anchor_array():
    out = []
    for scn in SCENARIOS:
        y0 = np.load(WORK / f"{scn}_test_y0.npy").astype(np.float64)
        out.append(age_shift_anchor(y0)[:, :, :2] / SUB_SCALE)
    return np.stack(out)


def affine_coeffs():
    """Per-age, per-variable affine recalibration of the anchor, fitted on the
    present-day training set. 60 parameters in total, so it cannot encode the
    site-specific present-day coincidences that sank the gradient-boosted model."""
    y0 = np.load(WORK / "train_y0.npy").astype(np.float64)
    y40 = np.load(WORK / "train_y40.npy").astype(np.float64)
    anc = age_shift_anchor(y0)[:, :, :2] / SUB_SCALE
    tgt = y40[:, :, :2] / SUB_SCALE
    C = np.zeros((15, 2, 2))
    for ai in range(15):
        for k in range(2):
            A = np.stack([anc[:, ai, k], np.ones(anc.shape[0])], 1)
            C[ai, k], *_ = np.linalg.lstsq(A, tgt[:, ai, k], rcond=None)
    return C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["anchor_affine", "blend", "delta"])
    p.add_argument("--model", default=None, help="raw .npy from submit_nn, (2,6171,15,2)")
    p.add_argument("--weights", default="0.25,0.4,0.55")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    anc = anchor_array()
    if a.mode == "anchor_affine":
        C = affine_coeffs()
        print("per-age affine slope/intercept (height, agb):")
        for ai, ag in enumerate(AGES):
            print(f"  age {ag:3d}  h {C[ai,0,0]:.3f}x{C[ai,0,1]:+.3f}   "
                  f"agb {C[ai,1,0]:.3f}x{C[ai,1,1]:+.3f}")
        out = anc * C[None, None, :, :, 0] + C[None, None, :, :, 1]
        write("C01_anchor_affine", out)
    elif a.mode == "delta":
        m = np.load(OUT / a.model)
        st = np.load(OUT / a.model.replace("_raw.npy", "_stat.npy"))
        d = m - st
        print(f"climate delta: mean {d.mean():+.4f} sd {d.std():.4f} | "
              f"height {d[:,:,:,0].mean():+.4f} agb {d[:,:,:,1].mean():+.4f} | "
              f"ssp126 {d[0].mean():+.4f} ssp585 {d[1].mean():+.4f}")
        write(f"C02_delta{a.tag}", anc + d)
        for g in [0.5, 1.5, 2.0]:
            write(f"C02_delta{a.tag}_g{g:g}", anc + g * d)
    else:
        m = np.load(OUT / a.model)
        assert m.shape == anc.shape, (m.shape, anc.shape)
        print(f"anchor vs model mean abs gap: "
              f"{np.abs(m-anc).mean():.4f}; corr "
              f"{np.corrcoef(m.ravel(), anc.ravel())[0,1]:.4f}")
        for w in [float(v) for v in a.weights.split(",")]:
            write(f"C_blend{a.tag}_w{w:g}", (1 - w) * anc + w * m)


if __name__ == "__main__":
    main()
