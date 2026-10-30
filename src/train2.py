"""Experiment driver with composable feature blocks, for the H1 ablation.

Blocks:
  state   : age + y0 + age-shifted anchor + full y0 age-curve   (always on, 122 feats)
  forcing : selected columns of the hand-aggregated 40-year forcing
  hrc     : selected columns of the Holographic-Reservoir encoding of the same forcing

Equal-budget comparison: pass the same --n-forcing / --n-hrc so each encoder gets the
same number of columns into the identical LightGBM head.

  python src/train2.py --exp E03 --forcing sel_top256.npy --one-fold
  python src/train2.py --exp E04_hrc_s0 --hrc 0 --n-hrc 256 --one-fold
  python src/train2.py --exp E05_both_s0 --forcing sel_top256.npy --hrc 0 --n-hrc 256 --one-fold
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "src")
from common import WORK, SUB_SCALE, AGES
import dataset as D

RUNS = WORK / "runs"
RUNS.mkdir(parents=True, exist_ok=True)


def hrc_paths(seed, size=256, masked=True):
    tag = "_m" if masked else ""
    return (WORK / f"hrc_train_s{seed}_n{size}{tag}.npy",
            {s: WORK / f"hrc_test_{s}_s{seed}_n{size}{tag}.npy"
             for s in ("ssp126", "ssp585")})


def pick_hrc_cols(hrc, y40, sites, n, seed=0):
    """Rank reservoir dimensions by |corr| with the two scored targets, pick top n.

    A cheap, leakage-free selector (uses only training sites of the fit) so that
    the reservoir gets the same column budget as the hand-crafted forcing block.
    """
    t = (y40[sites][:, :, :2] / SUB_SCALE).mean(axis=1)          # (S,2) age-averaged
    h = np.asarray(hrc[sites], dtype=np.float64)
    h = (h - h.mean(0)) / (h.std(0) + 1e-9)
    t = (t - t.mean(0)) / (t.std(0) + 1e-9)
    c = np.abs(h.T @ t) / len(sites)                              # (F,2)
    score = c.max(axis=1)
    return np.sort(np.argsort(-score)[:n])


def run(a):
    import lightgbm as lgb
    y0 = np.load(WORK / "train_y0.npy")
    y40 = np.load(WORK / "train_y40.npy")
    nsite = y0.shape[0]
    fold, _, _ = D.spatial_folds(a.folds, a.block_deg, seed=0)

    blocks = []
    tags = []
    if a.forcing:
        xf = np.asarray(np.load(WORK / "train_xf.npy", mmap_mode="r"))
        cols = np.load(WORK / a.forcing)
        blocks.append(xf[:, cols]); tags.append(f"forcing{len(cols)}")
        del xf
    hrc_cols = None
    if a.hrc is not None:
        hp, _ = hrc_paths(a.hrc, a.hrc_size, not a.hrc_unmasked)
        hrc = np.asarray(np.load(hp, mmap_mode="r"))
        tr0 = np.arange(nsite)[fold != 0]
        hrc_cols = pick_hrc_cols(hrc, y40, tr0, a.n_hrc)
        blocks.append(hrc[:, hrc_cols]); tags.append(f"hrc_s{a.hrc}_{a.n_hrc}")
        del hrc
    extra = (np.concatenate(blocks, axis=1).astype(np.float32)
             if blocks else np.zeros((nsite, 0), dtype=np.float32))

    params = dict(objective="l2", metric="l2", learning_rate=a.lr,
                  num_leaves=a.leaves, min_data_in_leaf=a.min_leaf,
                  feature_fraction=a.ff, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, num_threads=a.threads, verbosity=-1,
                  seed=a.seed, max_bin=127)

    oof = np.full((nsite, 15, 2), np.nan, dtype=np.float32)
    iters, t0 = {}, time.time()
    for f in range(a.folds):
        tr = np.arange(nsite)[fold != f]
        va = np.arange(nsite)[fold == f]
        if a.sites and a.sites < len(tr):
            tr = np.sort(np.random.default_rng(a.seed).choice(tr, a.sites, replace=False))
        Xtr, Fs = D.build_rows(y0, extra, tr, None)
        Xva, _ = D.build_rows(y0, extra, va, None)
        Ytr = D.targets(y40, tr); Yva = D.targets(y40, va)
        for k, nm in enumerate(["height", "agb"]):
            dtr = lgb.Dataset(Xtr, label=np.ascontiguousarray(Ytr[:, k]), free_raw_data=False)
            dva = lgb.Dataset(Xva, label=np.ascontiguousarray(Yva[:, k]),
                              reference=dtr, free_raw_data=False)
            bst = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(a.patience, verbose=False)])
            oof[va, :, k] = bst.predict(Xva, num_iteration=bst.best_iteration).reshape(len(va), 15)
            iters[f"f{f}_{nm}"] = bst.best_iteration
            print(f"  fold {f} {nm}: iter {bst.best_iteration} "
                  f"val {bst.best_score['valid_0']['l2']:.5f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            del dtr, dva, bst
        nfeat = Xva.shape[1]
        del Xtr, Xva, Ytr, Yva
        if a.one_fold:
            break

    m = ~np.isnan(oof[:, 0, 0])
    true = y40[m][:, :, :2] / SUB_SCALE
    pred = oof[m]
    res = dict(exp=a.exp, blocks=tags, mse=float(np.mean((pred - true) ** 2)),
               mse_height=float(np.mean((pred[:, :, 0] - true[:, :, 0]) ** 2)),
               mse_agb=float(np.mean((pred[:, :, 1] - true[:, :, 1]) ** 2)),
               per_age={str(AGES[i]): float(np.mean((pred[:, i] - true[:, i]) ** 2))
                        for i in range(15)},
               n_sites_scored=int(m.sum()), n_features=int(nfeat), folds=a.folds,
               one_fold=a.one_fold, iters=iters,
               minutes=round((time.time() - t0) / 60, 2), args=vars(a))
    if hrc_cols is not None:
        res["hrc_cols"] = hrc_cols.tolist()
    (RUNS / f"{a.exp}.json").write_text(json.dumps(res, indent=1))
    np.save(RUNS / f"{a.exp}_oof.npy", oof)
    print(json.dumps({k: res[k] for k in
                      ["exp", "blocks", "mse", "mse_height", "mse_agb",
                       "n_features", "n_sites_scored", "minutes"]}, indent=1))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--forcing", default=None)
    p.add_argument("--hrc", type=int, default=None)
    p.add_argument("--hrc-size", type=int, default=256)
    p.add_argument("--hrc-unmasked", action="store_true")
    p.add_argument("--n-hrc", type=int, default=256)
    p.add_argument("--sites", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=10.0)
    p.add_argument("--one-fold", action="store_true")
    p.add_argument("--rounds", type=int, default=2500)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--leaves", type=int, default=127)
    p.add_argument("--min-leaf", type=int, default=100)
    p.add_argument("--ff", type=float, default=0.6)
    p.add_argument("--threads", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())
