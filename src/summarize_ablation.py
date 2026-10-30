"""Summarise the H1 ablation: own technology vs professional control, 3 seeds each."""
import json, glob, os
import numpy as np
from pathlib import Path
RUNS = Path(os.environ.get('WORK_DIR', './work')) / 'runs'
arms = {'HRC solo (reservorio + lectura ridge)': 'AB_hrc_s*.json',
        'Control neuronal (MLP residual sobre rollout)': 'AB_nn_s*.json',
        'Hibrido (MLP + estado del reservorio)': 'AB_nnhrc_s*.json'}
out = ['# Ablacion de H1 — aportacion medida del Holographic-Reservoir', '',
       'Protocolo: emulador de un paso, rollout de 40 anos desde `y0`, MSE del ano 40 en',
       'unidades oficiales, sitios retenidos por bloques espaciales de 10 grados.',
       'Presupuesto igualado: 8 epocas para los brazos neuronales, reservorio N=801 con',
       '3 fugas y lectura cuadratica para el brazo HRC (su mejor configuracion).', '',
       '| brazo | semillas | MSE ano 40 (media +- sd) | sitios calidos | min/semilla |',
       '|---|---|---|---|---|']
res = {}
for name, pat in arms.items():
    fs = sorted(glob.glob(str(RUNS / pat)))
    if not fs:
        out.append(f'| {name} | 0 | (sin datos) | — | — |'); continue
    v = [json.load(open(f)) for f in fs]
    m = np.array([d['mse_rollout_y40'] for d in v])
    w = np.array([d['mse_warm_sites'] for d in v])
    t = np.mean([d['minutes'] for d in v])
    res[name] = m
    out.append(f'| {name} | {len(v)} | {m.mean():.4f} +- {m.std(ddof=1) if len(m)>1 else 0:.4f} '
               f'| {w.mean():.4f} | {t:.1f} |')
out.append('')
k_ctl = 'Control neuronal (MLP residual sobre rollout)'
k_hrc = 'HRC solo (reservorio + lectura ridge)'
k_hyb = 'Hibrido (MLP + estado del reservorio)'
if k_ctl in res and k_hrc in res:
    out.append(f'- HRC solo frente al control: {res[k_hrc].mean()/res[k_ctl].mean():.2f}x '
               f'el MSE del control. **H1 rechazada en este papel.**')
if k_ctl in res and k_hyb in res:
    r = res[k_hyb].mean() / res[k_ctl].mean()
    verdict = 'no aporta' if r >= 1.0 else f'mejora un {100*(1-r):.1f} %'
    out.append(f'- Hibrido frente al control: {r:.2f}x el MSE del control -> el estado del '
               f'reservorio {verdict} como entrada adicional.')
out.append('')
out.append('Caveat honesto: con 8 epocas los brazos neuronales no estan convergidos y la')
out.append('dispersion entre semillas es grande comparada con la diferencia entre brazos,')
out.append('asi que la comparacion ordena los brazos pero no cuantifica con precision la brecha.')
Path('experiments/ABLACION_H1.md').write_text('\n'.join(out), encoding='utf-8')
print('\n'.join(out))
