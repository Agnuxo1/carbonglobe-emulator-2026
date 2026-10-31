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

## Segunda tanda (30-10, tras medir la calibración local↔leaderboard)

| id | sistema | validación local (rollout) | nota |
|---|---|---|---|
| AB_nn_s0/s1/s2 | control neuronal, 8 épocas sin cota | 0,0593 ± 0,0091 | brazo de control de la ablación |
| AB_nnhrc_s0/s1/s2 | híbrido, 8 épocas sin cota | 0,0733 ± 0,0059 | 1,24× el control |
| AB_hrc_s0/s1/s2 | HRC solo, N=801, 3 fugas, cuadrática | 0,1528 ± 0,0031 | 2,58× el control |
| IM_nn_e24_h512 | control neuronal, 24 épocas **sin** cota | **NaN** | divergió al anularse el teacher forcing en el pico del LR |
| IM_nn_e24_h768d4 | ídem, más ancho, sin cota | **NaN** | misma causa |
| IM_clamp_e24_h512 | control neuronal, 24 épocas **con cota** | **0,0451** | mejor sistema entrenado; cálidos 0,0709 |
| IM_clamp_hyb_e24 | híbrido, 24 épocas con cota | 0,0467 | 1,04× el control, dentro del ruido entre semillas |

Los dos NaN se registran tal cual: son el fallo que motivó la cota física del estado, y la
cota mejora el control por sí sola (0,0515 → 0,0451 con la misma semilla) además de
permitir triplicar las épocas.

### Interpolador del ancla (comparación de fórmulas sin ajustar)

| variante | local |
|---|---|
| media de las dos edades que lo acotan | 0,17332 |
| lineal en la edad | 0,17877 |
| lineal en log-edad (la usada) | 0,17987 |
| PCHIP monótono en log-edad | 0,18181 |
| spline cúbica en log-edad | 0,18578 |

Rango total 0,173–0,186: la interpolación **no** es el cuello de botella del ancla, así que
no se gastó más esfuerzo ahí. El residuo del ancla es el efecto real del cambio de clima.

### Delta climático (construcción «delta change»)

Rollout con el forzamiento SSP menos rollout con la anomalía de los 25 drivers desplazados
puesta a cero, con el emulador mejorado: media +0,0156, sd 0,124, y el orden de escenarios
es el correcto (SSP585 +0,0202 frente a SSP126 +0,0110). Con el emulador anterior la sd era
0,276, es decir mucho más ruido. Correlación ancla–emulador 0,935.

## Cola de envíos del 31-10 (5/5) — el blend bate al ancla

| # | candidato | pregunta | LB | puesto aprox. | veredicto |
|---|---|---|---|---|---|
| 6 | `C01_anchor_affine` | ¿una calibración mínima del presente (60 parámetros) ayuda al ancla? | 0,137 | ~144 | **no**: empeora el ancla (0,094). Se activa la regla de parada: no se vuelve a ajustar nada contra la validación del presente |
| 7 | `S06_nn_clamp` | ¿el emulador neuronal mejorado (0,0451 local) bate al ancla? | 0,166 | ~164 | no. Razón LB/local **3,68×**: el mejor rollout local transfiere peor que el malo |
| 8 | `C02_delta_v2` | ¿el delta climático corrige el ancla? | 0,109 | ~130 | **no**: la construcción «delta change» empeora el ancla. No se escala el delta |
| 9 | **`C_blend_v2_w0.3`** | ¿la mezcla ancla/emulador bate a las dos por separado? | **0,076** | **~71** | **sí, −19 % sobre el ancla.** Mejor sistema del concurso |
| 10 | `C03_blend_twoemu_w0.3` | ¿aporta diversidad usar la media de los dos emuladores? | 0,082 | ~78 | no: el reservorio arrastra la mezcla hacia abajo incluso al 15 % efectivo |

Parábola por los tres puntos de la mezcla (0,094 con w=0; 0,076 con w=0,3; 0,166 con w=1):
óptimo en **w = 0,309**. El peso enviado ya era el óptimo, así que reajustarlo no da nada.

### Los pesos de mezcla tampoco se pueden elegir en local (`experiments/blend_weights.json`)

Sobre los sitios retenidos del presente, el peso óptimo ancla/emulador sale **w = 0,837**
—casi el emulador puro— cuando el leaderboard dice **0,309**. Haber hecho caso a la
validación local habría dado algo cercano a 0,166 en vez de 0,076. En la misma regresión a
tres bandas, los pesos óptimos locales son ancla 0,169 / neuronal 0,806 / **reservorio
0,016**: una confirmación independiente más de que el reservorio no aporta.

Tabla de calibración local↔leaderboard con los ocho puntos ya medidos:

| sistema | local | LB | razón |
|---|---|---|---|
| constante por edad | 1,6455 | 1,542 | 0,94 |
| persistencia | 0,2962 | 0,285 | 0,96 |
| ancla cruda | 0,1799 | 0,094 | 0,52 |
| emulador HRC (rollout) | 0,1808 | 0,245 | 1,35 |
| emulador neuronal (rollout) | 0,0451 | 0,166 | 3,68 |
| LightGBM directo | 0,0425 | 0,257 | 6,05 |

La razón crece monótonamente con lo ajustado que está el sistema. Es el resultado
metodológico del trabajo.

## Más capacidad no ayuda (31-10)

Hipótesis: con la receta estable (cota del estado + anulación del teacher forcing a mitad), un
emulador más ancho y más entrenado debería bajar del 0,0451 de rollout local.

Resultado: **rechazada.** El modelo de 768×4 con 36 épocas colapsa al anular el teacher forcing
—pérdida de rollout 0,0165 en la época 12 (teacher 0,33) → 0,1301 en la 18 (teacher 0,00)— y se
estanca en 0,128 sin recuperarse, muy por encima del que alcanza el de 512×3. El run además se
quedó sin RAM del equipo en la época 27 (0,69 GB libres, con otras sesiones trabajando), así que
paré mi propio proceso de ajuste con todos los sitios en vez de competir por memoria.

Decisión: se conserva la receta de **512×3, 24 épocas, con cota** como lado entrenado de la mezcla,
y no se gasta más cómputo en capacidad. El emulador sigue sin estar en su techo en *épocas*, pero
sí parece estarlo en *anchura* con este objetivo de rollout.

| config | épocas | pérdida al anular el teacher forcing | rollout y40 local |
|---|---|---|---|
| 512×3 con cota | 24 | se mantiene baja | **0,0451** |
| 768×4 con cota | 36 | 0,0165 → 0,1301, estancada en 0,128 | no converge (run sin RAM en la época 27) |
| 512×3 sin cota | 24 | — | NaN |
| 768×4 sin cota | 24 | — | NaN |

## Candidatos construidos y verificados para la cola siguiente

Todos con un control de identidad que comprueba que reproducen exactamente el envío de 0,076
cuando el peso se vuelve uniforme (diferencia 0,0 y 1e-06 respectivamente):

| candidato | idea | qué tiene de distinto |
|---|---|---|
| `C06_selfcal_w0.309_s1` | **pesos autocalibrados**: `w ∝ 1/(1+(r/τ)²)` con `r = |rollout estacionario − ancla|` | la señal sale **solo del test** (inputs y los dos rollouts del propio emulador), sin tocar ningún objetivo del presente, y se reescala para que la media de `w` sea 0,309, el óptimo uniforme ya medido. Así el envío prueba si **condicionar** ayuda, no si ayuda otra media |
| `C06_selfcal_sharp_w0.309_s0.5` | ídem, más agresivo (τ a la mitad) | mide la pendiente del efecto |
| `C04_pervar_h0.3_a0.45` y `C04_pervar_h0.45_a0.3` | un peso por variable | altura y agb tienen errores distintos; las dos direcciones, para no suponer de qué lado cae el óptimo |
| `C05_perage_0.2to0.4` | peso creciente con la edad semilla | el error del ancla crece con la edad (0,0107 a edad 1 y 0,0678 a edad 500 en local) |

## Trabajo GPU declarado en la cola (31-10, 05:49 UTC)

Las dos RTX 3090 estaban paradas y el carril `gpu1` quedó asignado a `ai-emulation` hasta el
1-11 05:49 UTC. Decisión de JEV (`jev/20261031-gpu/answer.json`, provenance=jev,
remote_decision=true): `what_to_queue=both_one_job` (0,41), `order=ensemble_first` (0,46),
`ensemble_size`≈3 (0,72), `ablation_scope=all_three_arms` (0,61), y **`publish_if_reverses`
0,81**, que me compromete a publicar con la misma prominencia que el rechazo actual si la
ablación convergida revirtiera o estrechara el veredicto sobre la tecnología propia.

Trabajo `ai-emulation-ensamblado-y-ablacion` en `gpu-feeder/RESPALDO_GPU.json`, dos piezas:

1. **Ensamblado por semillas** del emulador de la mezcla: semillas 2, 3 y 4 de la receta
   estabilizada 512×3 con cota, 24 épocas, sobre los 54152 sitios, guardando la predicción
   cruda del año 40 y la del contrafactual estacionario. Es el único eje que queda, porque la
   capacidad ya está medida como inútil.
2. **Ablación convergida** de los tres brazos de H1 con esa misma receta y 3 semillas por
   brazo, para sustituir la tabla publicada, que corrió 8 épocas sin converger.

Preparación, toda verificada antes de declarar nada:

- El venv compartido `/home/ai/venvs/training` trae torch 2.14.1+cu130 con CUDA disponible y
  la cola lo activa sola. Le falta `holographic_reservoir` y el README **prohíbe instalar en
  el entorno compartido**, así que la librería va vendorizada en la carpeta del trabajo
  (Apache-2.0, solo stdlib y numpy; mi código ya lo soportaba por `HRC_SRC`).
- Los datos (2,3 GB) no caben en el envío del trabajo (límite de 2 GiB), así que fueron por
  el puente **como .tar**, extraídos en `/home/ai/datos-ai-emulation` —fuera del espejo, para
  que una subida posterior no los pise— y verificados en el servidor: 11 archivos, formas
  `(54152,40,136)` y `(54152,15,41,7)`, muestras finitas. El tar se borró después (2,24 GiB
  liberados en un disco compartido con 31 GiB).
- **Prueba de humo completa en CPU por el puente** con valores mínimos y
  `CUDA_VISIBLE_DEVICES` vacío, para no usar GPU fuera de la cola: los tres brazos y el
  ensamblado corren, escriben sus JSON y el reservorio vendorizado funciona (d=343,
  25 860 600 ejemplos con teacher forcing). Es así como descargué la señal de segunda opinión
  de JEV (0,81): una verificación **ejecutada** en el entorno real vale más que una lectura.
- Cada resultado se escribe en cuanto termina y al reanudar se salta lo ya hecho, porque el
  trabajo corre sin nadie mirando y el carril se puede perder.

Dos fallos de camino que conviene no repetir:
- El puente falla con «Cowardly refusing to create an empty archive» si la carpeta está en la
  **raíz** de una unidad: construye `-C "E:\"` y la barra final escapa la comilla en `cmd`.
  Hay que subir desde un nivel más abajo.
- Los datasets grandes no van por `puente subir` directo, sino empaquetados en .tar.

## Resultados del trabajo GPU (31-10, job 9462de440dcc4c17, exit 0, 1,55 h)

**Ablacion convergida**: ver `ABLACION_H1.md`. Resumen: HRC 0,1528 +- 0,0031, control
0,0489 +- 0,0026, hibrido 0,0497 +- 0,0037. El rechazo de H1 se refuerza (2,58x -> 3,13x) y
el hibrido pasa a ser indistinguible del control (1,24x -> 1,02x), lo segundo a favor de la
tecnologia propia y recogido con la misma prominencia.

**Ensamblado por semillas**: las tres semillas nuevas (2, 3, 4) ajustadas sobre los 54152
sitios, ~9,5 min cada una, con sus predicciones crudas del ano 40 y las del contrafactual
estacionario. Con la semilla 1 ya ajustada en local son cuatro. Dispersion entre semillas
sd media 0,1204 (hay varianza real que promediar), correlacion media entre pares 0,9667. El
promedio correlaciona 0,9389 con el ancla, algo mas que una sola semilla (0,9351).

Candidatos construidos: `C07_ens4_w0.309` (mismo peso que el optimo medido, para aislar el
efecto del ensamblado) y `C07_ens4_w0.4` (porque un emulador con menos varianza podria
merecer mas peso). Pendientes de la cuota, que se renueva a las 00:00 UTC del 1-11.
