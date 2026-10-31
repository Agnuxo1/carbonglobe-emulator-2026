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

## Ablacion CONVERGIDA (31-10, en server-ia, job 9462de440dcc4c17)

Esta es la tabla que sustituye a la de 8 epocas. Misma receta estabilizada para los tres
brazos (cota del estado, 24 epocas, teacher forcing anulado a mitad), 3 semillas por brazo,
mismos sitios retenidos y misma metrica. 1,55 h de GPU, exit 0.

| brazo | MSE ano 40 (media +- sd, 3 semillas) | valores | sitios calidos | min/semilla |
|---|---|---|---|---|
| HRC solo (N=801, 3 fugas, lectura cuadratica) | 0,1528 +- 0,0031 | 0,1560 / 0,1498 / 0,1526 | 0,2384 | 5,3 |
| **control neuronal** (MLP residual sobre el rollout) | **0,0489 +- 0,0026** | 0,0516 / 0,0463 / 0,0488 | 0,0710 | 7,1 |
| hibrido (control + estado del reservorio) | 0,0497 +- 0,0037 | 0,0504 / 0,0457 / 0,0530 | 0,0852 | 9,0 |

Que cambia respecto a la version sin converger, en los dos sentidos:

1. **El rechazo de H1 se refuerza.** El reservorio solo pasa de 2,58x a **3,13x** el MSE del
   control, y la dispersion del control baja de +-0,0091 a +-0,0026: converger estrecho el
   intervalo y separo mas los brazos. Los intervalos siguen sin solaparse, ahora con mucho
   mas margen.
2. **El hibrido mejora su posicion y hay que decirlo.** Pasa de 1,24x a **1,02x**: la
   diferencia con el control es 0,0008, muy por debajo de las desviaciones (0,0026 y 0,0037).
   La afirmacion correcta ya no es «no aporta», sino que **el estado del reservorio como
   entrada adicional es indistinguible del control en la metrica, y cuesta un 27 % mas de
   tiempo** (9,0 frente a 7,1 min por semilla). Sigue sin haber razon para usarlo, pero por
   coste, no por precision.

Este segundo punto es un resultado **a favor** de la tecnologia propia frente a lo que se
habia publicado, y se recoge aqui con la misma prominencia por el compromiso que dejo JEV al
autorizar el trabajo (`jev/20261031-gpu/answer.json`, `publish_if_reverses` 0,81).

Lo que no cambia: el reservorio solo sigue muy por detras; el peso que le da una regresion a
tres bandas sobre sitios retenidos sigue siendo 0,016 frente a 0,806 del emulador neuronal; y
promediar los dos emuladores empeoro la mezcla en el leaderboard (0,082 frente a 0,076).
