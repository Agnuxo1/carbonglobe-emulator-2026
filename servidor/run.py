"""Trabajo GPU de ai-emulation para la cola de server-ia.

Dos piezas, en este orden (decision de JEV 31-10, jev/20261031-gpu, provenance=jev:
`what_to_queue = both_one_job`, `order = ensemble_first`):

  1. ENSAMBLADO POR SEMILLAS del emulador que va en la mezcla enviada. La capacidad ya
     esta medida como inutil (768x4 se estanca en 0,128 frente a 0,0451 de 512x3), asi
     que las semillas son el eje que queda. Ajusta la receta estabilizada de 512x3 con
     cota sobre los 54152 sitios con varias semillas y guarda la prediccion cruda del
     ano 40 de cada una, mas la del contrafactual de clima estacionario. El promedio se
     mezcla con el ancla y se envia; una sola puntuacion decide si aporta.

  2. ABLACION CONVERGIDA de los tres brazos de H1 con la receta estabilizada, 3 semillas
     por brazo (`ablation_scope = all_three_arms`). La tabla publicada hoy corrio 8
     epocas con la perdida aun bajando y se reporta con ese aviso; esto la sustituye por
     una comparacion convergida. JEV deja dicho (0,81) que si el resultado convergido
     revirtiera o estrechara el rechazo de la tecnologia propia, se publica con la misma
     prominencia que el rechazo actual.

Cada resultado se escribe en cuanto termina (resultados/*.json y *.npy), porque el
trabajo corre sin ningun agente mirando y el carril se puede perder.

Entorno: usa el venv compartido /home/ai/venvs/training (torch CUDA), que la cola activa
sola. No instala nada: `holographic_reservoir` va vendorizado en esta misma carpeta, que
es lo que pide el README del servidor ("no instalar paquetes arbitrarios sobre el entorno
compartido").

Datos: se transfieren aparte por el puente (el envio del trabajo tiene limite de 2 GiB y
los arrays pesan 2,3 GB). DATA_DIR apunta al espejo.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("JOB_BASE", str(HERE)))   # solo la prueba de humo lo cambia
DATA = Path(os.environ.get("DATA_DIR", "/home/ai/datos-ai-emulation"))
OUT = BASE / "resultados"
WORK = BASE / "work"

# --- preparar un WORK local (enlaces a los arrays del espejo) antes de importar nada
WORK.mkdir(parents=True, exist_ok=True)
(WORK / "runs").mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
if not DATA.is_dir():
    print(f"FATAL: no existe DATA_DIR {DATA}", flush=True)
    sys.exit(2)
for f in sorted(DATA.iterdir()):
    if not f.is_file():
        continue
    dst = WORK / f.name
    if dst.exists() or dst.is_symlink():
        continue
    try:
        dst.symlink_to(f)
    except OSError:
        import shutil
        shutil.copy2(f, dst)
os.environ["WORK_DIR"] = str(WORK)
os.environ["CG_DIR"] = str(WORK)
os.environ["COMP_DIR"] = str(WORK)
os.environ["HRC_SRC"] = str(HERE)          # holographic_reservoir vendorizado
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import torch.nn as nn                                            # noqa: E402

from common import SUB_SCALE, N_SITE, SCENARIOS, stats           # noqa: E402
import dataset as D                                              # noqa: E402
from hrc_onestep import forcing_norm, AGE_COL, DEV               # noqa: E402
import hrc_onestep                                               # noqa: E402
from nn_onestep import Step, Batcher, unroll, year_weights, state_bounds  # noqa: E402

def _env_int(name, default):
    try: return int(os.environ.get(name, default))
    except ValueError: return default


def _env_seeds(name, default):
    v = os.environ.get(name)
    return [int(x) for x in v.split(",")] if v else default


# Valores de produccion. Se pueden bajar por entorno SOLO para la prueba de humo
# (SMOKE=1 en CPU por el puente), nunca para el trabajo real de la cola.
CHUNK = _env_int("JOB_CHUNK", 96)
EPOCHS = _env_int("JOB_EPOCHS", 24)
TF_EPOCHS = _env_int("JOB_TF_EPOCHS", 12)
HIDDEN = _env_int("JOB_HIDDEN", 512)
DEPTH = _env_int("JOB_DEPTH", 3)
VAL_SITES = _env_int("JOB_VAL_SITES", 2500)
MAX_SITES = _env_int("JOB_MAX_SITES", 0)          # 0 = los 54152
MAX_TEST = _env_int("JOB_MAX_TEST", 0)            # 0 = los 6171 sitios de test
SEEDS_ENSEMBLE = _env_seeds("JOB_SEEDS_ENS", [2, 3, 4])   # la 1 ya esta ajustada
SEEDS_ABLATION = _env_seeds("JOB_SEEDS_ABL", [0, 1, 2])


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def save(name, obj):
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    log("escrito", p.name)


# ---------------------------------------------------------------- 1. ensamblado
def fit_full_and_predict(seed):
    """Ajusta sobre los 54152 sitios y devuelve la prediccion del ano 40 del test.

    Devuelve (ssp, estacionario), cada uno (2, 6171, 15, 2) ya dividido por SUB_SCALE.
    """
    torch.manual_seed(seed)
    prep, _ = forcing_norm(True)
    bad = np.array(json.loads((WORK / "bad_channels.json").read_text()), dtype=np.int64)
    _, _, y_mean, y_std = stats()
    ymf = y_mean.astype(np.float32); ysf = (y_std + 1e-10).astype(np.float32)
    ym = torch.as_tensor(ymf, device=DEV); ys = torch.as_tensor(ysf, device=DEV)
    ages_col = torch.as_tensor(AGE_COL, device=DEV)

    model = Step(144, HIDDEN, DEPTH).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bat = Batcher(prep, ymf, ysf, CHUNK)
    wy = year_weights(0, DEV)
    bounds = state_bounds()
    sites = np.arange(N_SITE)
    if MAX_SITES:
        sites = sites[:MAX_SITES]
    spe = max(1, len(sites) // CHUNK)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * spe,
                                                pct_start=0.15)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = rng.permutation(sites)
        teacher = max(0.0, 1.0 - ep / TF_EPOCHS)
        model.train(); run = 0.0
        for i in range(spe):
            s = np.sort(perm[i * CHUNK:(i + 1) * CHUNK])
            xz, yz = bat.get(s)
            pred = unroll(model, xz, yz, ages_col, teacher, None, bounds)
            loss = ((pred - yz.reshape(-1, 41, 7)[:, 1:]) ** 2).mean(dim=(0, 2)) @ wy / 40.0
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run += float(loss.detach())
            del xz, yz, pred
        log(f"  semilla {seed} epoca {ep} teacher {teacher:.2f} loss {run/spe:.4f} "
            f"{time.time()-t0:.0f}s")
        if not np.isfinite(run / spe):
            log(f"  semilla {seed}: perdida no finita, se aborta esta semilla")
            return None, None

    model.eval()
    outs, outs_stat = {}, {}
    with torch.no_grad():
        for scn in SCENARIOS:
            xt = np.load(WORK / f"test_xann_{scn}.npy", mmap_mode="r")
            y0 = np.load(WORK / f"{scn}_test_y0.npy")
            if MAX_TEST:
                y0 = y0[:MAX_TEST]
            a_, b_ = [], []
            for i in range(0, y0.shape[0], CHUNK):
                sl = slice(i, min(i + CHUNK, y0.shape[0]))
                xraw = np.asarray(xt[sl], dtype=np.float32)
                xzn = prep(xraw)
                y0b = y0[sl]
                yz = torch.as_tensor(
                    (np.concatenate([y0b[:, :, None, :],
                                     np.zeros((y0b.shape[0], 15, 40, 7), np.float32)],
                                    axis=2) - ymf) / ysf, device=DEV)
                for arr, zz in ((a_, xzn), (b_, np.where(
                        np.isin(np.arange(136), bad)[None, None, :], 0.0, xzn))):
                    xz = torch.as_tensor(np.ascontiguousarray(zz, dtype=np.float32),
                                         device=DEV)
                    pr = unroll(model, xz, yz, ages_col, 0.0, None, bounds)
                    p40 = (pr.view(y0b.shape[0], 15, 40, 7)[:, :, -1, :2] * ys[:2]
                           + ym[:2]).cpu().numpy()
                    arr.append(p40)
                    del xz, pr
                del yz
            outs[scn] = np.concatenate(a_, 0) / SUB_SCALE
            outs_stat[scn] = np.concatenate(b_, 0) / SUB_SCALE
            log(f"  semilla {seed} {scn}: height {outs[scn][:,:,0].mean():.3f} "
                f"agb {outs[scn][:,:,1].mean():.3f}")
    return (np.stack([outs[s] for s in SCENARIOS]),
            np.stack([outs_stat[s] for s in SCENARIOS]))


def ensemble():
    hechas = []
    for seed in SEEDS_ENSEMBLE:
        tag = f"ens_s{seed}"
        if (OUT / f"{tag}_raw.npy").exists():
            log("ya existe, se salta", tag); hechas.append(seed); continue
        log(f"=== ensamblado: semilla {seed} ===")
        t0 = time.time()
        ssp, stat = fit_full_and_predict(seed)
        if ssp is None:
            save(tag, {"seed": seed, "estado": "perdida_no_finita"}); continue
        np.save(OUT / f"{tag}_raw.npy", ssp)
        np.save(OUT / f"{tag}_stat.npy", stat)
        save(tag, {"seed": seed, "estado": "ok",
                   "minutos": round((time.time() - t0) / 60, 2),
                   "media_height": float(ssp[:, :, :, 0].mean()),
                   "media_agb": float(ssp[:, :, :, 1].mean())})
        hechas.append(seed)
    save("ensamblado_resumen", {"semillas_hechas": hechas,
                                "nota": "El promedio de estas semillas mas la de 1 ya "
                                        "ajustada en local se mezcla con el ancla al "
                                        "0,309 medido y se envia."})


# ---------------------------------------------------------------- 2. ablacion
def ablacion():
    import argparse
    for seed in SEEDS_ABLATION:
        for arm, kwargs in (("nn", {}), ("nnhrc", {"hrc_dim": 300})):
            tag = f"abl_{arm}_s{seed}"
            if (OUT / f"{tag}.json").exists():
                log("ya existe, se salta", tag); continue
            log(f"=== ablacion {arm} semilla {seed} ===")
            import nn_onestep
            a = argparse.Namespace(exp=tag, hidden=HIDDEN, depth=DEPTH, epochs=EPOCHS,
                                   tf_epochs=TF_EPOCHS, lr=2e-3, chunk=CHUNK,
                                   val_sites=VAL_SITES, seed=seed, no_anomaly=False,
                                   hrc_dim=kwargs.get("hrc_dim", 0),
                                   leaks="0.1,0.3,0.7", spectral_radius=0.9,
                                   no_clamp=False)
            try:
                nn_onestep.main(a)
                src = WORK / "runs" / f"{tag}.json"
                if src.exists():
                    (OUT / f"{tag}.json").write_text(src.read_text(encoding="utf-8"),
                                                     encoding="utf-8")
                    log("copiado", tag)
            except Exception as e:
                save(tag, {"estado": "fallo", "error": repr(e)[:400]})
                log("FALLO", tag, repr(e)[:200])
    for seed in SEEDS_ABLATION:
        tag = f"abl_hrc_s{seed}"
        if (OUT / f"{tag}.json").exists():
            log("ya existe, se salta", tag); continue
        log(f"=== ablacion hrc semilla {seed} ===")
        a = argparse.Namespace(exp=tag, size=_env_int("JOB_HRC_SIZE", 801),
                               seed=seed, spectral_radius=0.9,
                               leak=0.3, ridge=10.0, chunk=100,
                               fit_sites=MAX_SITES,
                               val_sites=VAL_SITES, year_lo=1, year_hi=40,
                               no_anomaly=False, warm_channel=0,
                               leaks="0.1,0.3,0.7", square=True)
        # fit_and_score escribe en WORK/runs
        try:
            hrc_onestep.fit_and_score(a)
            src = WORK / "runs" / f"{tag}.json"
            if src.exists():
                (OUT / f"{tag}.json").write_text(src.read_text(encoding="utf-8"),
                                                 encoding="utf-8")
                log("copiado", tag)
        except Exception as e:
            save(tag, {"estado": "fallo", "error": repr(e)[:400]})
            log("FALLO", tag, repr(e)[:200])


if __name__ == "__main__":
    log("dispositivo:", DEV, "| torch", torch.__version__,
        "| cuda", torch.version.cuda, "| disponible", torch.cuda.is_available())
    log("DATA_DIR:", DATA, "| archivos:", len(list(DATA.iterdir())))
    if os.environ.get("SMOKE"):
        log("PRUEBA DE HUMO: valores reducidos, no es el trabajo real")
    log(f"config: chunk={CHUNK} epocas={EPOCHS} tf={TF_EPOCHS} {HIDDEN}x{DEPTH} "
        f"val={VAL_SITES} max_sites={MAX_SITES or N_SITE} "
        f"ens={SEEDS_ENSEMBLE} abl={SEEDS_ABLATION}")
    solo = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if solo in ("todo", "ensamblado"):
        ensemble()
    if solo in ("todo", "ablacion"):
        ablacion()
    log("TRABAJO TERMINADO")
