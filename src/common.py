"""Shared constants and helpers for the CarbonGlobe AI Emulation Challenge.

Data layout (official tutorial 01_data_tutorial.ipynb, IEEE BigData Cup 2026):
  glob_X_fea.npy    (54152, 40, 12, 136)  site x year x month x driver
  glob_Y{age}.npy   (54152, 41, 12, 7)    site x year(0=init) x month x target
  target of the Cup  = DECEMBER (month index 11) of YEAR 40, variables height & agb,
                       each divided by its global mean.
"""
import os
import numpy as np
from pathlib import Path

# Where the data lives. Set these before running; see reproduce.sh.
CG = Path(os.environ.get("CG_DIR", "./data/carbonglobe"))    # present-day benchmark
COMP = Path(os.environ.get("COMP_DIR", "./data/competition"))  # competition test data
WORK = Path(os.environ.get("WORK_DIR", "./work"))              # derived arrays (~4 GB)

TREE_BAND = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
N_SITE, N_YEAR, N_MONTH, N_FEA, N_OUT = 54152, 40, 12, 136, 7
DEC = 11            # December, the month the benchmark and the Cup score
SCORED = [0, 1]     # height, agb
# Published submission scalers (public/README.md of the competition data)
SUB_SCALE = np.array([10.90645, 3.39567], dtype=np.float64)
SCENARIOS = ["ssp126", "ssp585"]


def stats():
    s = np.load(CG / "data_stats.npz")
    return s["x_mean"], s["x_std"], s["y_mean"], s["y_std"]


def age_triplet():
    a = np.load(CG / "age_triplet.npz")
    return a["age"], a["age_mean"], a["age_slope"]


def mask2d():
    return np.load(CG / "mask2d.npy")


def site_lonlat():
    """Latitude/longitude of each of the 54152 sites, in site-index order.

    mask2d is (360, 720) on a 0.5 deg grid, origin='upper' (row 0 = +90 lat).
    """
    m = mask2d().astype(bool)
    rows, cols = np.nonzero(m)
    lat = 89.75 - rows * 0.5
    lon = -179.75 + cols * 0.5
    return lon.astype(np.float32), lat.astype(np.float32)


# ---------------------------------------------------------------- feature block
def aggregate_forcing(xb):
    """Aggregate one chunk of raw forcing into fixed-length per-site features.

    xb: (S, 40, 12, 136) float32, physical units.
    Returns (S, F) float32 together with the list of feature names (see names()).

    The aggregation is deliberately shift-tolerant: the model is trained on
    present-day forcing and applied to SSP126/SSP585 future forcing, so we keep
    level, trend, variability and seasonal shape rather than raw year-by-year
    values, which would be out of range.
    """
    S = xb.shape[0]
    ann = xb.mean(axis=2)                                   # (S, 40, 136) annual mean
    blocks = [
        ann.mean(axis=1),                                   # 40-year mean
        ann[:, :10].mean(axis=1),                           # decade 1
        ann[:, 10:20].mean(axis=1),                         # decade 2
        ann[:, 20:30].mean(axis=1),                         # decade 3
        ann[:, 30:].mean(axis=1),                           # decade 4
        ann.std(axis=1),                                    # inter-annual sd
        ann.min(axis=1),
        ann.max(axis=1),
    ]
    # least-squares slope over the 40 years (per driver)
    tc = np.arange(40, dtype=np.float32) - 19.5
    slope = (ann * tc[None, :, None]).sum(axis=1) / (tc * tc).sum()
    blocks.append(slope)
    # seasonal shape: climatological quarterly means over the whole record
    clim = xb.mean(axis=1)                                  # (S, 12, 136)
    for q in range(4):
        blocks.append(clim[:, 3 * q:3 * q + 3].mean(axis=1))
    # seasonal amplitude
    blocks.append(clim.max(axis=1) - clim.min(axis=1))
    return np.concatenate([b.reshape(S, -1) for b in blocks], axis=1).astype(np.float32)


FORCING_BLOCKS = ["mean", "dec1", "dec2", "dec3", "dec4", "sd", "min", "max",
                  "slope", "q1", "q2", "q3", "q4", "amp"]


def forcing_names():
    return [f"x{j:03d}_{b}" for b in FORCING_BLOCKS for j in range(N_FEA)]
