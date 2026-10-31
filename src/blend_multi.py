"""Blend the anchor with the AVERAGE of the two emulators, at the leaderboard-optimal
total emulator share.

The leaderboard parabola through the three measured points (anchor 0.094, blend w=0.3
0.076, emulator alone 0.166) puts the optimum at w = 0.309, so the submitted weight was
already optimal and re-tuning it buys nothing. The open question is different: whether
the emulator side of the blend benefits from averaging two emulators with different
error structure (the neural one and the reservoir one) instead of using only the best.
Present-day data cannot answer it -- it picks w = 0.837 where the leaderboard picks
0.309 -- so this costs one submission.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,'src')
from common import WORK, COMP, SCENARIOS, AGES, SUB_SCALE
from blend import anchor_array, write

anc = anchor_array()
nn = np.load(WORK/'subs'/'S06_nn_clamp_raw.npy')                  # (2,6171,15,2)
# the reservoir emulator's test predictions, read back from the submitted file
h = pd.read_csv(WORK/'subs'/'S03_hrc_onestep_n400.csv')
tm = pd.read_csv(COMP/'sample_submission.csv')[['id']]
h = tm.merge(h, on='id', how='left')
assert h[['height','agb']].notna().all().all()
hrc = np.zeros_like(nn)
for si, scn in enumerate(SCENARIOS):
    m = h['id'].str.startswith(scn+'_').values
    blk = h.loc[m, ['height','agb']].values.reshape(6171, 15, 2)
    hrc[si] = blk
print('correlaciones con el ancla: nn %.4f  hrc %.4f  nn-hrc %.4f' % (
    np.corrcoef(nn.ravel(), anc.ravel())[0,1],
    np.corrcoef(hrc.ravel(), anc.ravel())[0,1],
    np.corrcoef(nn.ravel(), hrc.ravel())[0,1]))
mix = 0.5*nn + 0.5*hrc
write('C03_blend_twoemu_w0.3', 0.7*anc + 0.3*mix)
