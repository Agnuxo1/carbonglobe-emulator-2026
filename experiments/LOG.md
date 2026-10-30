# Registro de experimentos — ai-emulation (IEEE BigData Cup 2026, CarbonGlobe)

Métrica oficial: MSE sobre `height/10,90645` y `agb/3,39567` en diciembre del año 40.
`10,90645` y `3,39567` son exactamente `y_mean[0:2]` de `data_stats.npz` del dataset,
comprobado, así que la métrica se reproduce en local sin ambigüedad.

Dos validaciones locales distintas, y conviene no confundirlas:
- **directa**: MSE del valor del año 40 predicho de un salto, con bloques espaciales de 10°.
- **rollout**: MSE del año 40 tras rodar el emulador 40 pasos desde `y0`, en sitios retenidos
  por los mismos bloques. Es la que imita el protocolo oficial.

## Tabla maestra

| id | hipótesis / sistema | validación local | leaderboard | decisión |
|---|---|---|---|---|
| — | constante global | 1,7179 (directa) | — | referencia |
| — | constante por edad | 1,6455 (directa) | **1,542** | referencia nula |
| — | prior `age_mean` del dataset | 1,6574 (directa) | — | descartado |
| P3 | persistencia `y40 = y0` | 0,2962 (directa) | **0,285** | referencia |
| — | lineal por edad sobre `y0` (1 var) | 0,1648 | — | — |
| — | lineal por edad sobre `y0` (7 vars) | 0,1232 | — | — |
| **P2** | **ancla por desplazamiento de edad, cruda, sin ajustar** | 0,1799 | **0,094** | **mejor sistema medido** |
| — | lineal por edad sobre la curva `y0` completa (105 feats) | 0,0727 | — | no enviado |
| E01 | LightGBM directo, solo estado (122 feats) | 0,0425 (1 pliegue) | — | → S01 |
| S01 | E01 reentrenado en los 54152 sitios | — | **0,257** | **rechazado: 6× peor que su validación** |
| E02 | LightGBM directo + 256 feats de clima, 5 pliegues | 0,0260 | — | validación no fiable |
| E03 | igual que E02, 1 pliegue (comparable con E01) | 0,0244 | — | validación no fiable |
| A | HRC un paso, N=400, 1 fuga, lectura lineal | 0,1808 (rollout) | — | — |
| S03 | A reentrenado en todos los sitios | — | **0,245** | rollout transfiere 1,36×, no 6× |
| B | HRC un paso, N=400 + lectura cuadrática | 0,1638 (rollout) | — | el término cuadrático aporta |
| C | HRC un paso, N=399, 3 fugas (0,1/0,3/0,7) | 0,1762 (rollout) | — | multiescala aporta poco solo |
| D | C + lectura cuadrática | 0,1638 (rollout) | — | sin ganancia extra sobre B |
| E | HRC un paso, N=801, 3 fugas + cuadrática | **0,1560** (rollout) | — | mejor HRC |
| N01 | control neuronal: MLP residual un paso entrenado sobre el rollout | **0,0694** (rollout) | — | **control gana a HRC 2,3×** |

## Ablación de H1 (3 semillas, presupuesto igualado: 8 épocas / ~3-6 min por brazo)

Ver `ABLACION_H1.md`, generado desde `E:/ai-emulation-work/runs/AB_*.json`.

## Hechos establecidos, con su evidencia

1. **El objetivo es diciembre del año 40**, no la media anual: el notebook oficial
   `01_data_tutorial.ipynb` usa `Y[s, year, 11, :]` («use Dec. for Y in all experiments»)
   y el año 40 es el índice 40 de los 41 años (el 0 es el inicial).
2. **`test_y0` está en unidades físicas**, no normalizado. Tiene valores ligeramente
   negativos (altura mínima −0,901), pero el entrenamiento también los tiene en gpp y npp:
   son artefactos del modelo ED cerca de suelo desnudo, no un cambio de escala. La media de
   altura por edad del test (0,80 → 10,33 m) sigue la misma forma que la del entrenamiento
   (2,23 → 11,71 m), con los sitios de test algo menos productivos.
3. **25 de los 136 drivers están fuera de rango en el test** (índices 8–31 y 135): más del
   90 % de las filas de test caen fuera de la banda 1–99 % del entrenamiento, y la tendencia
   de algunos se sale hasta 8,8 veces el rango. Son los que distinguen SSP126 de SSP585.
   El entrenamiento no contiene variación de escenario, así que su efecto no se puede
   aprender de él; se usan como anomalía respecto a la línea base de cada sitio (decisión de
   JEV, `shifted_drivers = anomaly`).
4. **Los 6171 sitios de test no se pueden localizar en la rejilla de 0,5°.** Emparejando por
   los 7 canales estáticos de suelo, la distancia al vecino más cercano (0,091) es
   prácticamente la del segundo (0,104) y no hay ni un emparejamiento exacto. Además el canal
   estático 135 está fuera del rango de entrenamiento. Consecuencia medida: **sin lat/lon no
   hay acoplamiento espacial, así que `pixelflow` queda descartado para este concurso**, y no
   se le atribuye ninguna contribución.
5. **La validación local directa no ordena los modelos para este test.** Calibración con
   cinco puntos enviados:

   | sistema | local | leaderboard | razón LB/local |
   |---|---|---|---|
   | constante por edad | 1,6455 | 1,542 | 0,94 |
   | persistencia | 0,2962 | 0,285 | 0,96 |
   | ancla cruda | 0,1799 | 0,094 | **0,52** |
   | HRC un paso (rollout) | 0,1808 | 0,245 | 1,35 |
   | LightGBM directo | 0,0425 | 0,257 | **6,05** |

   Los predictores sin ajustar transfieren casi exactos; el ancla incluso mejora en el test.
   Lo que entrena contra la validación del presente se degrada, y tanto más cuanto más
   capacidad tiene. El modelo directo acabó siendo **2,7× peor que una de sus propias
   entradas** (el ancla), es decir, aprendió a no usarla.

## Consecuencias para el diseño

- El sistema a batir es el ancla (0,094), y los candidatos se construyen como **correcciones
  al ancla**, no como sustitutos.
- El emulador se usa solo para el **delta climático**: rollout con el forzamiento SSP menos
  rollout con el mismo forzamiento sin la anomalía de los 25 drivers desplazados. El sesgo
  sistemático del emulador se cancela en la diferencia y el ancla aporta el nivel. Es la
  construcción «delta change» habitual en estudios de impacto climático.
- La selección final se hace contra el leaderboard con pocos candidatos claramente distintos,
  porque no hay validación local fiable para ordenarlos.
