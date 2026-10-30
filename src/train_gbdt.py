"""LightGBM control model for the direct year-40 regression.

Usage:
  python src/train_gbdt.py --exp E01 --no-forcing --sites 54152
  python src/train_gbdt.py --exp E02 --fcols sel_top256.npy
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "src")
from common import WORK, SUB_SCALE, AGES
import dataset as D

RUNS = WORK / "runs"
RUNS.mkdir(parents=True, exist_ok=True)


def load_train():
    y0 = np.load(WORK / "train_y0.npy")
    y40 = np.load(WORK / "train_y40.npy")
    p = WORK / "train_xf.npy"
    xf = np.load(p, mmap_mode="r") if p.exists() else None
    return y0, y40, xf


def run(args):
    import lightgbm as lgb
    y0, y40, xf = load_train()
    nsite = y0.shape[0]
    fold, lon, lat = D.spatial_folds(args.folds, args.block_deg, seed=0)

    fcols = None
    if args.no_forcing:
        xf = np.zeros((nsite, 0), dtype=np.float32)
    elif args.fcols:
        fcols = np.load(WORK / args.fcols)

    rng = np.random.default_rng(args.seed)
    all_sites = np.arange(nsite)
    if args.sites and args.sites < nsite:
        all_sites = np.sort(rng.choice(nsite, args.sites, replace=False))

    params = dict(objective="l2", metric="l2", learning_rate=args.lr,
                  num_leaves=args.leaves, min_data_in_leaf=args.min_leaf,
                  feature_fraction=args.ff, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, num_threads=args.threads, verbosity=-1,
                  seed=args.seed, max_bin=127)

    oof = np.full((nsite, 15, 2), np.nan, dtype=np.float32)
    models = {}
    t0 = time.time()
    for f in range(args.folds):
        tr = all_sites[fold[all_sites] != f]
        va = all_sites[fold[all_sites] == f]
        Xtr, Fs = D.build_rows(y0, xf, tr, fcols)
        Xva, _ = D.build_rows(y0, xf, va, fcols)
        Ytr = D.targets(y40, tr); Yva = D.targets(y40, va)
        for k, name in enumerate(["height", "agb"]):
            dtr = lgb.Dataset(Xtr, label=Ytr[:, k], free_raw_data=False)
            dva = lgb.Dataset(Xva, label=Yva[:, k], reference=dtr, free_raw_data=False)
            bst = lgb.train(params, dtr, num_boost_round=args.rounds,
                            valid_sets=[dva],
                            callbacks=[lgb.early_stopping(args.patience, verbose=False)])
            p = bst.predict(Xva, num_iteration=bst.best_iteration)
            oof[va, :, k] = p.reshape(len(va), 15)
            models[(f, name)] = bst.best_iteration
            print(f"  fold {f} {name}: best_iter {bst.best_iteration} "
                  f"val_mse {bst.best_score['valid_0']['l2']:.5f}", flush=True)
            del dtr, dva, bst
        del Xtr, Xva, Ytr, Yva
        if args.one_fold:
            break

    m = ~np.isnan(oof[:, 0, 0])
    true = y40[m][:, :, :2] / SUB_SCALE
    pred = oof[m]
    mse = float(np.mean((pred - true) ** 2))
    per_var = [float(np.mean((pred[:, :, k] - true[:, :, k]) ** 2)) for k in range(2)]
    per_age = [float(np.mean((pred[:, i] - true[:, i]) ** 2)) for i in range(15)]
    res = dict(exp=args.exp, mse=mse, mse_height=per_var[0], mse_agb=per_var[1],
               per_age=dict(zip(map(str, AGES), per_age)),
               n_sites_scored=int(m.sum()), folds=args.folds, seed=args.seed,
               n_features=int(Xva.shape[1]) if 'Xva' in dir() else None,
               rounds_used={f"{a}|{b}": c for (a, b), c in models.items()},
               minutes=round((time.time() - t0) / 60, 2), args=vars(args))
    print(json.dumps({k: v for k, v in res.items() if k != "per_age"}, indent=1))
    (RUNS / f"{args.exp}.json").write_text(json.dumps(res, indent=1))
    np.save(RUNS / f"{args.exp}_oof.npy", oof)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--no-forcing", action="store_true")
    p.add_argument("--fcols", default=None)
    p.add_argument("--sites", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--block-deg", type=float, default=10.0)
    p.add_argument("--one-fold", action="store_true")
    p.add_argument("--rounds", type=int, default=3000)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--leaves", type=int, default=127)
    p.add_argument("--min-leaf", type=int, default=100)
    p.add_argument("--ff", type=float, default=0.6)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())
