"""Compare untrained interpolators for the age-shifted anchor.

All variants read the site's own y0-vs-age curve at age+40; none fits anything to
the targets, so comparing them on present-day data is legitimate even though
trained models do not transfer (the anchor's local-to-leaderboard ratio is a
measured 0.52, and it applies to every variant equally).
"""
import numpy as np, sys, json
sys.path.insert(0,'src')
from common import WORK, AGES, SUB_SCALE
from scipy.interpolate import PchipInterpolator, CubicSpline

A=np.array(AGES,float); T=A+40.0
y0=np.load(WORK/'train_y0.npy').astype(np.float64)
y40=np.load(WORK/'train_y40.npy').astype(np.float64)
tgt=y40[:,:,:2]/SUB_SCALE
S=y0.shape[0]
def mse(p): return float(np.mean((p-tgt)**2))

def lin_log():
    la,tl=np.log(A),np.log(T)
    i=np.searchsorted(la,tl).clip(1,14); lo,hi=i-1,i
    w=((tl-la[lo])/(la[hi]-la[lo])).clip(0,1)
    return (y0[:,lo,:2]*(1-w)[None,:,None]+y0[:,hi,:2]*w[None,:,None])/SUB_SCALE

def lin_age():
    i=np.searchsorted(A,T).clip(1,14); lo,hi=i-1,i
    w=((T-A[lo])/(A[hi]-A[lo])).clip(0,1)
    return (y0[:,lo,:2]*(1-w)[None,:,None]+y0[:,hi,:2]*w[None,:,None])/SUB_SCALE

def shaped(kind):
    out=np.zeros((S,15,2))
    la,tl=np.log(A),np.log(T)
    for k in range(2):
        c=y0[:,:,k]
        f=(PchipInterpolator if kind=='pchip' else CubicSpline)(la,c,axis=1,extrapolate=True)
        out[:,:,k]=f(tl)
    return out/SUB_SCALE

res={'linear in log-age (current)':mse(lin_log()),
     'linear in age':mse(lin_age()),
     'monotone PCHIP in log-age':mse(shaped('pchip')),
     'cubic spline in log-age':mse(shaped('spline'))}
# mean of the bracketing grid ages, a crude but very robust variant
la,tl=np.log(A),np.log(T)
i=np.searchsorted(la,tl).clip(1,14)
res['bracket mean (no weighting)']=mse((y0[:,i-1,:2]+y0[:,i,:2])[None][0]/2/SUB_SCALE)
for k,v in sorted(res.items(),key=lambda x:x[1]): print(f'  {k:30s} {v:.5f}')
json.dump(res,open('experiments/anchor_variants.json','w'),indent=1)
