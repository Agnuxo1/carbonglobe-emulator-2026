import zipfile, shutil, sys, numpy as np
sys.path.insert(0,'src')
from common import COMP, WORK, SCENARIOS
WORK.mkdir(parents=True, exist_ok=True)
for scn in SCENARIOS:
    p = COMP/f'test_{scn}.npz'
    with zipfile.ZipFile(p) as z:
        print(scn, [(i.filename, i.file_size, i.compress_type) for i in z.infolist()])
        for name in z.namelist():
            out = WORK/f'{scn}_{name}'
            if out.exists(): print('  skip', out.name); continue
            with z.open(name) as src, open(out,'wb') as dst:
                shutil.copyfileobj(src, dst, 1<<22)
            a = np.load(out, mmap_mode='r')
            print('  ', out.name, a.shape, a.dtype)
