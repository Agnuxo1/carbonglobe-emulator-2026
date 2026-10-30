"""Locate the 6171 competition sites on the global 0.5 deg grid.

The competition ships sites_ssp.csv as a bare 0..6170 row index with no lon/lat.
But 7 of the 136 drivers (channels 129-135, soil properties) are constant in time
and site-specific, so they fingerprint a site. We match each test site to the
nearest of the 54152 CarbonGlobe sites in that 7-d space and report how clean the
match is. A clean match unlocks lat/lon, spatial neighbours and pixelflow.
"""
import numpy as np, sys, json
sys.path.insert(0,'src')
from common import CG, WORK, SCENARIOS, N_SITE, site_lonlat

STAT = np.arange(129,136)
X = np.load(CG/'glob_X_fea.npy', mmap_mode='r')
ref = np.zeros((N_SITE,len(STAT)),dtype=np.float64)
for i in range(0,N_SITE,4000):
    j=min(i+4000,N_SITE)
    ref[i:j]=np.asarray(X[i:j,0,0,:],dtype=np.float64)[:,STAT]
print('ref static',ref.shape)
# also confirm these are static in the TRAIN array
chk=np.asarray(X[:200],dtype=np.float64)[:,:,:,STAT]
print('train temporal sd of static channels (max):',float(chk.std(axis=(1,2)).max()))

sd=ref.std(axis=0); sd[sd==0]=1.0
refn=(ref-ref.mean(axis=0))/sd
from scipy.spatial import cKDTree
tree=cKDTree(refn)
lon,lat=site_lonlat()
out={}
for scn in SCENARIOS:
    T=np.load(WORK/f'{scn}_test_x.npy',mmap_mode='r')
    q=np.asarray(T[:,0,0,:],dtype=np.float64)[:,STAT]
    qn=(q-ref.mean(axis=0))/sd
    d,idx=tree.query(qn,k=2)
    exact=int((d[:,0]<1e-6).sum())
    uniq=len(np.unique(idx[:,0]))
    print(f'{scn}: exact matches {exact}/{len(idx)}  unique targets {uniq}  '
          f'd1 median {np.median(d[:,0]):.2e}  d2 median {np.median(d[:,1]):.3f}')
    out[scn]=dict(exact=exact,unique=uniq,n=len(idx))
    np.save(WORK/f'map_{scn}.npy',idx[:,0].astype(np.int32))
    np.save(WORK/f'map_{scn}_dist.npy',d[:,0].astype(np.float32))
    if scn=='ssp126':
        s=idx[:,0]
        print('  lon/lat span of matched sites:',round(float(lon[s].min()),1),round(float(lon[s].max()),1),
              round(float(lat[s].min()),1),round(float(lat[s].max()),1))
(WORK/'site_match.json').write_text(json.dumps(out,indent=1))
# are the two scenarios the same sites?
a=np.load(WORK/'map_ssp126.npy');b=np.load(WORK/'map_ssp585.npy')
print('ssp126 and ssp585 map to the same sites:',bool((a==b).all()))
