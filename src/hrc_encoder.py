"""Holographic-Reservoir as a sequence encoder for the 480-month forcing record.

This is Francisco Angulo's own technology under test (hypothesis H1). The reservoir
is built by the *original* code path -- `HolographicReservoirModel.__post_init__`
in the holographic_reservoir package -- so `W_in` and the sparse recurrent `W`,
including the SHA-256-derived diagonal entropy from the simulated ASIC backend and
the spectral-radius rescaling, are exactly the ones that library produces.

What is ported, and declared as such: the original `_collect_states` is a Python
loop over one sequence at a time, which is hopeless for 54152 x 480 steps. We run
the identical update rule

    x <- (1 - leak) * x + leak * tanh(W_in u_t + W x)

batched over sites on the GPU in float32. Same dynamics, same weights, different
loop order. Only the readout differs: instead of the library's ridge readout we
export pooled reservoir states as features for the downstream head, which is the
role JEV selected (`hrc_role = encoder`, confidence 0.88).
"""
import os
import argparse, json, sys, time
import numpy as np
import torch

sys.path.insert(0, "src")
# pip install holographic-reservoir; or point HRC_SRC at a local checkout
if os.environ.get("HRC_SRC"):
    sys.path.insert(0, os.environ["HRC_SRC"])
from common import CG, WORK, SCENARIOS, N_SITE, stats

from holographic_reservoir.model import HolographicReservoirModel  # noqa: E402


def build_reservoir(size, input_dim, seed, spectral_radius, leak):
    """Instantiate the original model purely to harvest its weights."""
    m = HolographicReservoirModel(size=size, input_dim=input_dim,
                                  spectral_radius=spectral_radius, leak=leak,
                                  random_seed=seed)
    return m.W_in.astype(np.float32), m.W.astype(np.float32), float(m.leak)


@torch.no_grad()
def encode(xseq, W_in, W, leak, dev):
    """xseq: (B, T, Din) float32 already z-scored. Returns pooled states (B, 4*N)."""
    B, T, _ = xseq.shape
    N = W.shape[0]
    u = torch.as_tensor(xseq, device=dev)
    Wi = torch.as_tensor(W_in.T.copy(), device=dev)          # (Din, N)
    Wr = torch.as_tensor(W.T.copy(), device=dev)             # (N, N)
    x = torch.zeros((B, N), device=dev, dtype=torch.float32)
    acc = torch.zeros((B, N), device=dev, dtype=torch.float32)
    acc2 = torch.zeros((B, N), device=dev, dtype=torch.float32)
    tail = torch.zeros((B, N), device=dev, dtype=torch.float32)
    TAIL = 120                                               # last 10 years of months
    for t in range(T):
        x = (1.0 - leak) * x + leak * torch.tanh(u[:, t] @ Wi + x @ Wr)
        acc += x
        acc2 += x * x
        if t >= T - TAIL:
            tail += x
    mean = acc / T
    sd = torch.sqrt(torch.clamp(acc2 / T - mean * mean, min=0.0))
    out = torch.cat([x, mean, sd, tail / TAIL], dim=1)
    return out.float().cpu().numpy()


def run_split(path, out_path, W_in, W, leak, xm, xs, dev, chunk, mask=None):
    X = np.load(path, mmap_mode="r")
    S = X.shape[0]
    N = W.shape[0]
    res = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(S, 4 * N))
    t0 = time.time()
    for i in range(0, S, chunk):
        j = min(i + chunk, S)
        blk = np.asarray(X[i:j], dtype=np.float32).reshape(j - i, -1, X.shape[-1])
        blk = (blk - xm) / xs
        if mask is not None:
            blk *= mask
        res[i:j] = encode(blk, W_in, W, leak, dev)
        print(f"  {j}/{S} {time.time()-t0:.0f}s", flush=True)
    res.flush()
    print("wrote", out_path, res.shape)


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x_mean, x_std, _, _ = stats()
    xm = x_mean.astype(np.float32)
    xs = (x_std + 1e-10).astype(np.float32)
    mask = None
    tag = ""
    if a.mask_shifted:
        bad = json.loads((WORK / "bad_channels.json").read_text())
        mask = np.ones(136, dtype=np.float32)
        mask[bad] = 0.0
        tag = "_m"
        print(f"  masking {len(bad)} shifted drivers out of the reservoir input")
    W_in, W, leak = build_reservoir(a.size, 136, a.seed, a.spectral_radius, a.leak)
    print(f"reservoir size={a.size} seed={a.seed} rho={a.spectral_radius} leak={leak} dev={dev}")
    print(f"  |W| nnz={int((W != 0).sum())} spectral={np.abs(np.linalg.eigvals(W)).max():.3f}")

    if a.what in ("all", "train"):
        run_split(CG / "glob_X_fea.npy",
                  WORK / f"hrc_train_s{a.seed}_n{a.size}{tag}.npy",
                  W_in, W, leak, xm, xs, dev, a.chunk, mask)
    if a.what in ("all", "test"):
        for scn in SCENARIOS:
            run_split(WORK / f"{scn}_test_x.npy",
                      WORK / f"hrc_test_{scn}_s{a.seed}_n{a.size}{tag}.npy",
                      W_in, W, leak, xm, xs, dev, a.chunk, mask)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spectral-radius", type=float, default=0.9)
    p.add_argument("--leak", type=float, default=0.3)
    p.add_argument("--chunk", type=int, default=1500)
    p.add_argument("--what", default="all")
    p.add_argument("--mask-shifted", action="store_true")
    main(p.parse_args())
