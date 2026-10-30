"""Precompute the arrays the one-step autoregressive emulator needs.

  train_xann.npy  (54152, 40, 136) float32  annual-mean forcing per year
  train_ydec.npy  (54152, 15, 41, 7) float32 December state, year 0 = initial
  test_xann_<scn>.npy (6171, 40, 136) float32

Why annual means: that is the feature the organizers' own baseline uses, and the
targets are annual (December) values, so a within-year resolution buys nothing for
the one-step map.
"""
import sys, time
import numpy as np
sys.path.insert(0, "src")
from common import CG, WORK, AGES, N_SITE, DEC, SCENARIOS

CHUNK = 400


def xann(src, out, n):
    if out.exists():
        print("skip", out.name); return
    X = np.load(src, mmap_mode="r")
    res = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(n, 40, 136))
    t0 = time.time()
    for i in range(0, n, CHUNK):
        j = min(i + CHUNK, n)
        res[i:j] = np.asarray(X[i:j], dtype=np.float32).mean(axis=2)
    res.flush()
    print("wrote", out.name, res.shape, f"{time.time()-t0:.0f}s", flush=True)


def ydec():
    out = WORK / "train_ydec.npy"
    if out.exists():
        print("skip", out.name); return
    res = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32,
                                    shape=(N_SITE, len(AGES), 41, 7))
    for ai, a in enumerate(AGES):
        Y = np.load(CG / f"glob_Y{a:03d}.npy", mmap_mode="r")
        for i in range(0, N_SITE, 3000):
            j = min(i + 3000, N_SITE)
            res[i:j, ai] = np.asarray(Y[i:j], dtype=np.float32)[:, :, DEC, :]
        print("  age", a, flush=True)
    res.flush()
    print("wrote", out.name, res.shape)


if __name__ == "__main__":
    ydec()
    xann(CG / "glob_X_fea.npy", WORK / "train_xann.npy", N_SITE)
    for s in SCENARIOS:
        xann(WORK / f"{s}_test_x.npy", WORK / f"test_xann_{s}.npy", 6171)
    print("OK")
