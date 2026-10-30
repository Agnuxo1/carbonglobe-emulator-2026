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