"""Pick the forcing columns worth keeping.

Two filters, in this order:

1. Covariate-shift filter. The model is fitted on present-day forcing and applied
   to SSP126/SSP585 forcing. A tree cannot extrapolate: beyond the training range
   it predicts a constant. A column whose test values sit largely outside the
   training range is therefore not just useless, it silently pins many test rows
   to the edge of a split. We measure, per column, the fraction of test rows
   outside the 1-99 percentile band of the training column and drop the worst.
2. Gain importance from a LightGBM fit on a site subsample.

Writes sel_top<N>.npy (column indices into train_xf) and a JSON diagnostic.
"""
import argparse, json, sys, time
import numpy as np
sys.path.insert(0, "src")
from common import WORK, SCENARIOS, forcing_names
import dataset as D


def shift_report(xf, tests):
    lo = np.percentile(xf, 1, axis=0)
    hi = np.percentile(xf, 99, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    out = np.zeros(xf.shape[1])
    for t in tests:
        out = np.maximum(out, ((t < lo) | (t > hi)).mean(axis=0))
    # how far outside, in training-span units (median over test rows)
    far = np.zeros(xf.shape[1])
    for t in tests:
        d = np.maximum(np.maximum(lo - t, t - hi), 0.0) / span
        far = np.maximum(far, np.median(d, axis=0))
    return out, far


def main(a):
    import lightgbm as lgb
    y0 = np.load(WORK / "train_y0.npy")
    y40 = np.load(WORK / "train_y40.npy")
    xf = np.asarray(np.load(WORK / "train_xf.npy", mmap_mode="r"))
    tests = [np.load(WORK / f"test_xf_{s}.npy") for s in SCENARIOS]

    frac_out, far = shift_report(xf, tests)
    keep = np.where((frac_out <= a.max_frac_out) & (far <= a.max_far))[0]
    names = forcing_names()
    print(f"shift filter: {len(keep)}/{xf.shape[1]} columns kept "
          f"(frac_out<={a.max_frac_out}, far<={a.max_far})")
    worst = np.argsort(-frac_out)[:12]
    for j in worst:
        print(f"  drop-candidate {names[j]:>14}  frac_out={frac_out[j]:.3f} far={far[j]:.2f}")

    rng = np.random.default_rng(0)
    sites = np.sort(rng.choice(y0.shape[0], a.sites, replace=False))
    fold, _, _ = D.spatial_folds(5, 10.0, seed=0)
    tr = sites[fold[sites] != 0]
    va = sites[fold[sites] == 0]
    Xtr, Fs = D.build_rows(y0, xf, tr, keep)
    Xva, _ = D.build_rows(y0, xf, va, keep)
    Ytr = D.targets(y40, tr); Yva = D.targets(y40, va)
    print("subsample rows", Xtr.shape, "state feats", Fs)

    params = dict(objective="l2", metric="l2", learning_rate=0.08, num_leaves=127,
                  min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, num_threads=a.threads,
                  verbosity=-1, seed=0, max_bin=127)
    gain = np.zeros(Xtr.shape[1])
    t0 = time.time()
    for k, nm in enumerate(["height", "agb"]):
        dtr = lgb.Dataset(Xtr, label=Ytr[:, k], free_raw_data=False)
        dva = lgb.Dataset(Xva, label=Yva[:, k], reference=dtr, free_raw_data=False)
        bst = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        g = bst.feature_importance("gain").astype(np.float64)
        gain += g / max(g.sum(), 1e-12)
        print(f"  {nm}: iter {bst.best_iteration} val {bst.best_score['valid_0']['l2']:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del dtr, dva, bst

    fgain = gain[Fs:]                      # forcing part only
    order = np.argsort(-fgain)
    sel = np.sort(keep[order[:a.top]])
    np.save(WORK / f"sel_top{a.top}.npy", sel)
    diag = dict(kept_after_shift=int(len(keep)), total=int(xf.shape[1]),
                top=a.top, sites=a.sites,
                best=[names[keep[j]] for j in order[:25]],
                dropped_by_shift=[names[j] for j in np.where(
                    (frac_out > a.max_frac_out) | (far > a.max_far))[0]][:60])
    (WORK / f"sel_top{a.top}.json").write_text(json.dumps(diag, indent=1))
    print("top-25 forcing features:", diag["best"])
    print("wrote", WORK / f"sel_top{a.top}.npy", sel.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sites", type=int, default=10000)
    p.add_argument("--top", type=int, default=256)
    p.add_argument("--rounds", type=int, default=600)
    p.add_argument("--max-frac-out", type=float, default=0.35)
    p.add_argument("--max-far", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=18)
    main(p.parse_args())
