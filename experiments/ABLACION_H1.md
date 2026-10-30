# Ablacion de H1 — aportacion medida del Holographic-Reservoir

Protocolo: emulador de un paso, rollout de 40 anos desde `y0`, MSE del ano 40 en
unidades oficiales, sitios retenidos por bloques espaciales de 10 grados.
Presupuesto igualado: 8 epocas para los brazos neuronales, reservorio N=801 con
3 fugas y lectura cuadratica para el brazo HRC (su mejor configuracion).

| brazo | semillas | MSE ano 40 (media +- sd) | sitios calidos | min/semilla |
|---|---|---|---|---|
| HRC solo (reservorio + lectura ridge) | 3 | 0.1528 +- 0.0031 | 0.2384 | 5.4 |
| Control neuronal (MLP residual sobre rollout) | 3 | 0.0593 +- 0.0091 | 0.0996 | 2.8 |
| Hibrido (MLP + estado del reservorio) | 3 | 0.0733 +- 0.0059 | 0.1294 | 3.7 |

- HRC solo frente al control: 2.58x el MSE del control. **H1 rechazada en este papel.**
- Hibrido frente al control: 1.24x el MSE del control -> el estado del reservorio no aporta como entrada adicional.

Caveat honesto: con 8 epocas los brazos neuronales no estan convergidos y la
dispersion entre semillas es grande comparada con la diferencia entre brazos,
asi que la comparacion ordena los brazos pero no cuantifica con precision la brecha.
## Comprobacion con la receta estabilizada (anadido despues)

La receta de la tabla anterior entrena 8 epocas con el teacher forcing anulado en la 3.
Al intentar alargarla a 24 epocas **divergio a NaN exactamente en la epoca en que el
teacher forcing llega a 0 mientras el OneCycleLR sigue cerca de su pico**: el estado
realimentado sale del rango que el mapa de un paso vio nunca y 40 composiciones del
incremento explotan. Se corrigio acotando el estado realimentado al rango fisicamente
alcanzable (`state_bounds` en `src/nn_onestep.py`, tomado de las trayectorias de
entrenamiento con un margen del 25 %) y retrasando la anulacion del teacher forcing a la
mitad del entrenamiento.

Con esa receta estable, misma semilla (1), mismos sitios retenidos:

| arm | MSE ano 40 | sitios calidos | min |
|---|---|---|---|
| control neuronal, 24 epocas con cota | **0,0451** | 0,0709 | 9,8 |
| hibrido (control + estado del reservorio), 24 epocas con cota | 0,0467 | 0,0750 | 13,1 |
| control neuronal, 8 epocas sin cota (misma semilla) | 0,0515 | 0,0860 | 2,8 |

Dos lecturas honestas:

1. La cota mejora el control por si sola (0,0515 -> 0,0451) y permite entrenar tres veces
   mas sin divergir, asi que **la brecha con el reservorio solo (0,1528) se ensancha**.
2. Para el hibrido, la diferencia se estrecha mucho: de 1,24x con la receta fragil a
   **1,04x** con la estable, que cae dentro de la dispersion entre semillas del control
   (+-0,0091). La afirmacion defendible es por tanto la mas debil: **el estado del
   reservorio como entrada adicional no aporta, y cuesta un 33 % mas de tiempo**; no que
   perjudique de forma clara.

La tabla de 3 semillas de arriba sigue siendo valida como comparacion: ambos brazos
neuronales corrieron la *misma* receta de 8 epocas, asi que la comparacion es justa,
aunque ninguno estuviera convergido.
