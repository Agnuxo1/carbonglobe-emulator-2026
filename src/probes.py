"""Deterministic leaderboard probes. No fitting, so they cannot overfit to the
present-day climate; their LB scores isolate where the train->test gap comes from."""
import numpy as np, pandas as pd, sys
sys.path.insert(0,'src')
from common import WORK, COMP, SCENARIOS, AGES, SUB_SCALE
from dataset import age_shift_anchor
OUT=WORK/'subs'; OUT.mkdir(parents=True,exist_ok=True)
tmpl=pd.read_csv(COMP/'sample_submission.csv')

def write(name, fn):
    rows=[]
    for s in SCENARIOS:
        y0=np.load(WORK/f'{s}_test_y0.npy')
        p=fn(y0)                                  # (6171,15,2) already /SUB_SCALE
        n=y0.shape[0]
        rows.append(pd.DataFrame({'id':[f'{s}_{i}_{a}' for i in range(n) for a in AGES],
                                  'height':p[:,:,0].ravel(),'agb':p[:,:,1].ravel()}))
    sub=pd.concat(rows,ignore_index=True)
    sub=tmpl[['id']].merge(sub,on='id',how='left')
    assert sub[['height','agb']].notna().all().all()
    sub[['height','agb']]=sub[['height','agb']].clip(lower=0)
    sub.to_csv(OUT/f'{name}.csv',index=False)
    print(name, sub.shape, 'height mean',round(sub.height.mean(),4),'agb mean',round(sub.agb.mean(),4))

# P1: per-age constant taken from the TRAINING year-40 December means
y40=np.load(WORK/'train_y40.npy')
const=(y40[:,:,:2]/SUB_SCALE).mean(axis=0)        # (15,2)
write('P1_perage_const', lambda y0: np.broadcast_to(const[None],(y0.shape[0],15,2)).copy())

# P2: raw age-shifted anchor read off the test's own y0 age-curve, no fitting
write('P2_anchor_raw', lambda y0: age_shift_anchor(y0.astype(np.float64))[:,:,:2]/SUB_SCALE)

# P3: persistence, y40 = y0
write('P3_persistence', lambda y0: y0[:,:,:2]/SUB_SCALE)
