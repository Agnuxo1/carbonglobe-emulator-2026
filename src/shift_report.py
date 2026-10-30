"""Find the drivers whose test values lie outside the training range, and record them.

Writes bad_channels.json in WORK: the raw driver indices for which more than 90 % of
the test rows fall outside the 1-99 percentile band of the training values, measured
on the record mean, the final-decade mean and the 40-year trend. These are the CO2 and
warming channels that distinguish SSP126 from SSP585; the training set has no scenario
variation, so their absolute level cannot be learned from it and is used as a
within-record anomaly instead.
"""
import json, sys
import numpy as np
sys.path.insert(0, "src")
from common import WORK, SCENARIOS, N_FEA

xf = np.asarray(np.load(WORK / "train_xf.npy", mmap_mode="r"))
te = [np.load(WORK / f"test_xf_{s}.npy") for s in SCENARIOS]
bad, rep = [], {}
for j in range(N_FEA):
    frac = 0.0
    for c in (j, 4 * N_FEA + j, 8 * N_FEA + j):          # mean, last decade, slope
        lo, hi = np.percentile(xf[:, c], [1, 99])
        for t in te:
            frac = max(frac, float(((t[:, c] < lo) | (t[:, c] > hi)).mean()))
    rep[str(j)] = round(frac, 4)
    if frac > 0.9:
        bad.append(j)
(WORK / "bad_channels.json").write_text(json.dumps(bad))
(WORK / "shift_report.json").write_text(json.dumps(rep, indent=1))
print(f"{len(bad)} of {N_FEA} drivers are out of range on the test: {bad}")
