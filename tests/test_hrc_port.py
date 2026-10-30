"""Check the batched GPU port reproduces holographic_reservoir's own _collect_states."""
import os
import numpy as np, sys, torch
sys.path.insert(0, 'src')
if os.environ.get('HRC_SRC'):
    sys.path.insert(0, os.environ['HRC_SRC'])
from holographic_reservoir.model import HolographicReservoirModel
m=HolographicReservoirModel(size=64,input_dim=5,spectral_radius=0.9,leak=0.3,random_seed=0)
rng=np.random.default_rng(1); u=rng.standard_normal((50,5))
ref=m._collect_states(u)                       # original, float64, single sequence
# batched port, float32, batch of 3 identical sequences
Wi=torch.as_tensor(m.W_in.T.copy().astype(np.float32)); Wr=torch.as_tensor(m.W.T.copy().astype(np.float32))
U=torch.as_tensor(np.repeat(u[None],3,0).astype(np.float32))
x=torch.zeros((3,64)); outs=[]
for t in range(50):
    x=(1-0.3)*x+0.3*torch.tanh(U[:,t]@Wi+x@Wr); outs.append(x.clone())
got=torch.stack(outs,1).numpy()
print('max abs diff final state :', float(np.abs(got[0,-1]-ref[-1]).max()))
print('max abs diff whole traj  :', float(np.abs(got[0]-ref).max()))
print('batch rows identical     :', bool(np.allclose(got[0],got[1])and np.allclose(got[0],got[2])))
print('state range              :', float(got.min()), float(got.max()))
