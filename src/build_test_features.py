"""Aggregate the SSP test forcing with exactly the same function used for training."""
import sys
import numpy as np
sys.path.insert(0, "src")
from common import WORK, SCENARIOS, aggregate_forcing

CHUNK = 300
for scn in SCENARIOS:
    out = WORK / f"test_xf_{scn}.npy"
    if out.exists():
        print("skip", out.name); continue
    tx = np.load(WORK / f"{scn}_test_x.npy", mmap_mode="r")
    parts = []
    for i in range(0, tx.shape[0], CHUNK):
        parts.append(aggregate_forcing(np.asarray(tx[i:i + CHUNK], dtype=np.float32)))
    arr = np.concatenate(parts, 0)
    np.save(out, arr)
    print("wrote", out.name, arr.shape, flush=True)
print("OK")
