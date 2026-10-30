"""Build per-site forcing features and per-(site,age) targets, streaming over the
14 GB CarbonGlobe arrays so that peak RAM stays a few hundred MB.

Outputs in WORK:
  train_xf.npy      (54152, 1904) float32   aggregated present-day forcing
  train_y0.npy      (54152, 15, 7) float32  December year-0 state per seed age
  train_y40.npy     (54152, 15, 7) float32  December year-40 state per seed age
  test_xf_<scn>.npy (6171, 1904)  float32   aggregated SSP forcing
  test_y0_<scn>.npy (6171, 15, 7) float32
"""
import sys, time
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import CG, COMP, WORK, AGES, N_SITE, DEC, aggregate_forcing, SCENARIOS

WORK.mkdir(parents=True, exist_ok=True)
CHUNK = 400


def build_train_forcing():
    out = WORK / "train_xf.npy"
    if out.exists():
        print("skip", out.name); return
    X = np.load(CG / "glob_X_fea.npy", mmap_mode="r")
    assert X.shape == (N_SITE, 40, 12, 136), X.shape
    probe = aggregate_forcing(np.asarray(X[:2], dtype=np.float32))
    F = probe.shape[1]
    res = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(N_SITE, F))
    t0 = time.time()
    for i in range(0, N_SITE, CHUNK):
        j = min(i + CHUNK, N_SITE)
        res[i:j] = aggregate_forcing(np.asarray(X[i:j], dtype=np.float32))
        if (i // CHUNK) % 20 == 0:
            print(f"  {j}/{N_SITE}  {time.time()-t0:.0f}s", flush=True)
    res.flush()
    print("wrote", out, res.shape, f"{time.time()-t0:.0f}s")


def build_train_targets():
    o0, o40 = WORK / "train_y0.npy", WORK / "train_y40.npy"
    if o0.exists() and o40.exists():
        print("skip targets"); return
    y0 = np.zeros((N_SITE, len(AGES), 7), dtype=np.float32)
    y40 = np.zeros((N_SITE, len(AGES), 7), dtype=np.float32)
    for ai, a in enumerate(AGES):
        Y = np.load(CG / f"glob_Y{a:03d}.npy", mmap_mode="r")
        assert Y.shape == (N_SITE, 41, 12, 7), Y.shape
        for i in range(0, N_SITE, 4000):
            j = min(i + 4000, N_SITE)
            blk = np.asarray(Y[i:j], dtype=np.float32)
            y0[i:j, ai] = blk[:, 0, DEC, :]
            y40[i:j, ai] = blk[:, 40, DEC, :]
        print("  age", a, "done", flush=True)
    np.save(o0, y0); np.save(o40, y40)
    print("wrote targets", y0.shape)


def build_test():
    for scn in SCENARIOS:
        of, oy = WORK / f"test_xf_{scn}.npy", WORK / f"test_y0_{scn}.npy"
        if of.exists() and oy.exists():
            print("skip", scn); continue
        p = COMP / f"test_{scn}.npz"
        if not p.exists():
            print("MISSING", p); continue
        z = np.load(p, mmap_mode="r") if p.suffix == ".npy" else np.load(p)
        tx = z["test_x"]                       # (6171, 40, 12, 136)
        ty = z["test_y0"]                      # (6171, 15, 7)
        S = tx.shape[0]
        feats = []
        for i in range(0, S, CHUNK):
            feats.append(aggregate_forcing(np.asarray(tx[i:i + CHUNK], dtype=np.float32)))
        np.save(of, np.concatenate(feats, 0))
        np.save(oy, np.asarray(ty, dtype=np.float32))
        print("wrote", scn, np.concatenate(feats, 0).shape, ty.shape, flush=True)
        del z, tx, ty, feats


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "targets"):
        build_train_targets()
    if what in ("all", "test"):
        build_test()
    if what in ("all", "forcing"):
        build_train_forcing()
    print("OK")
