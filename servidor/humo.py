"""Prueba de humo del trabajo, en CPU y con valores minimos.

Sirve para validar en el ENTORNO REAL del servidor (venv compartido, numpy 2.5,
pandas 3.1, la libreria del reservorio vendorizada y las rutas de los datos) antes
de que el trabajo corra horas en la cola sin nadie mirando.

Fuerza CUDA_VISIBLE_DEVICES vacio: el puente NO debe usar GPU; la GPU va por la cola.
"""
import os, runpy, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SMOKE"] = "1"
os.environ["JOB_BASE"] = "/tmp/humo-ai-emulation"
os.environ["DATA_DIR"] = "/home/ai/datos-ai-emulation"
os.environ.update({
    "JOB_CHUNK": "24", "JOB_EPOCHS": "2", "JOB_TF_EPOCHS": "1",
    "JOB_HIDDEN": "32", "JOB_DEPTH": "2", "JOB_MAX_SITES": "240",
    "JOB_VAL_SITES": "48", "JOB_MAX_TEST": "48", "JOB_HRC_SIZE": "99",
    "JOB_SEEDS_ENS": "99", "JOB_SEEDS_ABL": "99",
    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
})
sys.argv = ["run.py", "todo"]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"),
               run_name="__main__")
