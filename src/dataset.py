"""Row construction for the direct year-40 regression.

One row = one (site, seed age) pair. Target = December year-40 height and agb,
divided by their global means, i.e. exactly the quantity Kaggle scores.
"""
import sys
import numpy as np
sys.path.insert(0, "src")
from common import WORK, AGES, SUB_SCALE, forcing_names, site_lonlat

AGE_ARR = np.array(AGES, dtype=np.float64)
LOG_AGE = np.log(AGE_ARR)
TGT_LOG_AGE = np.log(AGE_ARR + 40.0)
_IDX = np.searchsorted(LOG_AGE, TGT_LOG_AGE).clip(1, 14)
_LO, _HI = _IDX - 1, _IDX
_W = ((TGT_LOG_AGE - LOG_AGE[_LO]) / (LOG_AGE[_HI] - LOG_AGE[_LO])).clip(0, 1)


def age_shift_anchor(y0):
    """Interpolate each site's y0-vs-age curve at age+40. y0: (S,15,7) -> (S,15,7).

    At test time the organisers give y0 for all 15 seed ages of the same site, so
    the state the site would reach at forest age a+40 under *present* conditions
    can be read off that curve. It is the single strongest predictor available.
    """
    return y0[:, _LO, :] * (1 - _W)[None, :, None] + y0[:, _HI, :] * _W[None, :, None]


def state_feature_names():
    n = ["age", "log_age", "log_age40"]
    n += [f"y0_{k}" for k in range(7)]
    n += [f"anchor_{k}" for k in range(7)]
    n += [f"curve_a{a}_{k}" for a in AGES for k in range(7)]
    return n


def build_state_block(y0):
    """(S,15,7) -> (S,15,Fs) state/age features."""
    S = y0.shape[0]
    anchor = age_shift_anchor(y0)
    curve = y0.reshape(S, 1, -1).repeat(15, axis=1)                     # (S,15,105)
    age = np.broadcast_to(AGE_ARR[None, :, None], (S, 15, 1))
    la = np.broadcast_to(LOG_AGE[None, :, None], (S, 15, 1))
    la40 = np.broadcast_to(TGT_LOG_AGE[None, :, None], (S, 15, 1))
    return np.concatenate([age, la, la40, y0, anchor, curve], axis=2).astype(np.float32)


def build_rows(y0, xf, sites=None, fcols=None):
    """Assemble (X, n_state) for the given sites.

    y0: (S,15,7); xf: (S,Fx) aggregated forcing; sites: indices into S.
    fcols: columns of xf to keep (None = all).
    Returns X (len(sites)*15, Fs+len(fcols)) float32, row order site-major.
    """
    if sites is None:
        sites = np.arange(y0.shape[0])
    sites = np.asarray(sites)
    st = build_state_block(y0[sites])                                   # (n,15,Fs)
    n, _, Fs = st.shape
    f = xf[sites] if fcols is None else xf[sites][:, fcols]
    f = np.repeat(f[:, None, :], 15, axis=1)
    X = np.concatenate([st, f], axis=2).reshape(n * 15, -1)
    return np.ascontiguousarray(X), Fs


def feature_names(fcols=None):
    fn = forcing_names()
    if fcols is not None:
        fn = [fn[j] for j in fcols]
    return state_feature_names() + fn


def targets(y40, sites=None):
    if sites is None:
        sites = np.arange(y40.shape[0])
    t = y40[np.asarray(sites)][:, :, :2] / SUB_SCALE
    return t.reshape(-1, 2).astype(np.float32)


def spatial_folds(n_folds=5, block_deg=10.0, seed=0):
    """Assign the 54152 global sites to folds by contiguous lon/lat blocks."""
    lon, lat = site_lonlat()
    bx = np.floor((lon + 180.0) / block_deg).astype(np.int64)
    by = np.floor((lat + 90.0) / block_deg).astype(np.int64)
    bid = by * 10000 + bx
    uniq = np.unique(bid)
    rng = np.random.default_rng(seed)
    assign = rng.permutation(len(uniq)) % n_folds
    m = dict(zip(uniq.tolist(), assign.tolist()))
    return np.array([m[b] for b in bid.tolist()], dtype=np.int8), lon, lat
