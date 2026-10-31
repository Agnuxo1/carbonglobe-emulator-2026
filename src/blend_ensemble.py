"""Mezcla el ancla con el PROMEDIO de varias semillas del emulador.

La capacidad ya esta medida como inutil (768x4 se estanca en 0,128 de perdida de
rollout frente a 0,0451 de 512x3), asi que las semillas eran el eje que quedaba.
Promediar semillas de la misma arquitectura reduce la varianza del lado entrenado
sin cambiar nada mas, y el peso de la mezcla se deja en el 0,309 que midio el
leaderboard, para que el envio pruebe el ensamblado y no otro peso.
"""
import glob, sys
import numpy as np
sys.path.insert(0, 'src')
from common import WORK
from blend import anchor_array, write

G = 'E:/ai-emulation-work/gpu-resultados'
partes = [np.load(WORK/'subs'/'S06_nn_clamp_raw.npy')]          # semilla 1, local
nombres = ['s1(local)']
for f in sorted(glob.glob(G+'/ens_s*_raw.npy')):
    partes.append(np.load(f)); nombres.append(f.split('ens_')[1].split('_raw')[0])
print('semillas:', nombres)
M = np.stack(partes)
print('dispersion entre semillas: sd media %.4f  max %.4f' % (M.std(0).mean(), M.std(0).max()))
print('correlacion media entre pares:', round(float(np.mean([
    np.corrcoef(M[i].ravel(), M[j].ravel())[0,1]
    for i in range(len(M)) for j in range(i+1, len(M))])), 4))
mean = M.mean(0)
anc = anchor_array()
print('ancla vs promedio: gap medio abs %.4f  corr %.4f' % (
    np.abs(mean-anc).mean(), np.corrcoef(mean.ravel(), anc.ravel())[0,1]))
for w in (0.309, 0.40):
    write(f'C07_ens{len(M)}_w{w:g}', (1-w)*anc + w*mean)
