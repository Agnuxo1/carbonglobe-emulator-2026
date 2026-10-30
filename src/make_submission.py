"""Fit the chosen configuration on all 54152 sites and write submission.csv.

The number of boosting rounds is taken from the cross-validated run (mean best
iteration over folds, scaled up by the extra training data), never tuned on the
leaderboard.

The submitted value is the December year-40 state divided by its global mean, as
public/README.md requires: an unscaled submission scores as if it were completely
wrong.
"""
import argparse, json, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from common import WORK, COMP, SCENARIOS, AGES, SUB_SCALE
import dataset as D
from train2 import hrc_paths, pick_hrc_cols

OUT = WORK / "subs"
OUT.mkdir(parents=True, exist_ok=True)


def assemble(a):
    """Return (extra_train, extra_test_by_scenario) feature blocks."""
    y40 = np.load(WORK / "train_y40.npy")
    ntr = y40.shape[0]
    tr_blocks, te_blocks = [], {s: [] for s in SCENARIOS}
    tags = []
    if a.forcing:
        cols = np.load(WORK / a.forcing)
        xf = np.asarray(np.load(WORK / "train_xf.npy", mmap_mode="r"))
        tr_blocks.append(xf[:, cols]); del xf
        for s in SCENARIOS:
            te_blocks[s].append(np.load(WORK / f"test_xf_{s}.npy")[:, cols])
        tags.append(f"forcing{len(cols)}")
    if a.hrc is not None:
        hp, tp = hrc_paths(a.hrc, a.hrc_size, True)
        h = np.asarray(np.load(hp, mmap_mode="r"))
        cols = pick_hrc_cols(h, y40, np.arange(ntr), a.n_hrc)
        tr_blocks.append(h[:, cols]); del h
        for s in SCENARIOS:
            te_blocks[s].append(np.load(tp[s])[:, cols])
        tags.append(f"hrc_s{a.hrc}_{a.n_hrc}")
    tr = (np.concatenate(tr_blocks, 1).astype(np.float32) if tr_blocks
          else np.zeros((ntr, 0), np.float32))
    te = {s: (np.concatenate(te_blocks[s], 1).astype(np.float32) if te_blocks[s]
              else np.zeros((6171, 0), np.float32)) for s in SCENARIOS}
    return tr, te, tags


def main(a):
    import lightgbm as lgb
    y0 = np.load(WORK / "train_y0.npy")
    y40 = np.load(WORK / "train_y40.npy")
    extra_tr, extra_te, tags = assemble(a)

    cv = json.loads((WORK / "runs" / f"{a.cv_run}.json").read_text())
    rounds = {}
    for nm in ("height", "agb"):
        vals = [v for k, v in cv["iters" if "iters" in cv else "rounds_used"].items()
                if k.endswith(nm)]
        rounds[nm] = max(50, int(np.mean(vals) * a.round_scale))
    print("rounds from CV", cv["exp"], rounds, "blocks", tags)

    Xtr, Fs = D.build_rows(y0, extra_tr, None, None)
    Ytr = D.targets(y40, None)
    params = dict(objective="l2", learning_rate=a.lr, num_leaves=a.leaves,
                  min_data_in_leaf=a.min_leaf, feature_fraction=a.ff,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=a.threads, verbosity=-1, seed=a.seed, max_bin=127)
    models = {}
    for k, nm in enumerate(["height", "agb"]):
        t0 = time.time()
        d = lgb.Dataset(Xtr, label=np.ascontiguousarray(Ytr[:, k]))
        models[nm] = lgb.train(params, d, num_boost_round=rounds[nm])
        print(f"  fitted {nm} on {Xtr.shape[0]} rows x {Xtr.shape[1]} feats "
              f"({time.time()-t0:.0f}s)", flush=True)
        del d
    del Xtr, Ytr

    rows = []
    for s in SCENARIOS:
        ty0 = np.load(WORK / f"{s}_test_y0.npy")
        Xte, _ = D.build_rows(ty0, extra_te[s], None, None)
        pred = np.stack([models[nm].predict(Xte) for nm in ("height", "agb")], 1)
        n = ty0.shape[0]
        ids = [f"{s}_{i}_{ag}" for i in range(n) for ag in AGES]
        rows.append(pd.DataFrame({"id": ids, "height": pred[:, 0], "agb": pred[:, 1]}))
        print(f"  predicted {s}: {pred.shape}, height {pred[:,0].mean():.3f} "
              f"agb {pred[:,1].mean():.3f}", flush=True)
        del Xte, pred
    sub = pd.concat(rows, ignore_index=True)

    tmpl = pd.read_csv(COMP / "sample_submission.csv")
    assert len(sub) == len(tmpl), (len(sub), len(tmpl))
    sub = tmpl[["id"]].merge(sub, on="id", how="left")
    assert sub[["height", "agb"]].notna().all().all(), "unmatched ids"
    sub["height"] = sub["height"].clip(lower=0.0)
    sub["agb"] = sub["agb"].clip(lower=0.0)
    path = OUT / f"{a.name}.csv"
    sub.to_csv(path, index=False)
    print("wrote", path, sub.shape)
    print(sub.head(3).to_string())
    (OUT / f"{a.name}.meta.json").write_text(json.dumps(
        dict(name=a.name, cv_run=a.cv_run, cv_mse=cv["mse"], blocks=tags,
             rounds=rounds, args=vars(a)), indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--cv-run", required=True)
    p.add_argument("--forcing", default=None)
    p.add_argument("--hrc", type=int, default=None)
    p.add_argument("--hrc-size", type=int, default=256)
    p.add_argument("--n-hrc", type=int, default=256)
    p.add_argument("--round-scale", type=float, default=1.15)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--leaves", type=int, default=127)
    p.add_argument("--min-leaf", type=int, default=100)
    p.add_argument("--ff", type=float, default=0.6)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
