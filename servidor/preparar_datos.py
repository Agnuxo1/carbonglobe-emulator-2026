"""Extrae el tar de datos de ai-emulation en el servidor y comprueba lo extraido.

Se extrae FUERA del espejo del puente (/home/ai/datos-ai-emulation) para que una
subida posterior del puente no lo pise ni genere copias .~1~.
"""
import hashlib
import json
import os
import tarfile
from pathlib import Path

TAR = Path("/home/ai/pc/e/ai-emulation-servidor/datos/datos.tar")
DEST = Path("/home/ai/datos-ai-emulation")

ESPERADO = {
    "train_xann.npy": 1178347648,
    "train_ydec.npy": 932497568,
    "train_y0.npy": 22743968,
    "test_xann_ssp126.npy": 134281088,
    "test_xann_ssp585.npy": 134281088,
    "ssp126_test_y0.npy": 2591948,
    "ssp585_test_y0.npy": 2591948,
    "mask2d.npy": 2073728,
    "data_stats.npz": 3282,
    "age_triplet.npz": 2558,
    "bad_channels.json": 99,
}


def main():
    print("tar:", TAR, "existe:", TAR.exists(),
          TAR.stat().st_size if TAR.exists() else "-")
    if not TAR.exists():
        raise SystemExit(2)
    DEST.mkdir(parents=True, exist_ok=True)
    faltan = [n for n in ESPERADO
              if not (DEST / n).exists() or (DEST / n).stat().st_size != ESPERADO[n]]
    if faltan:
        print("extrayendo", len(faltan), "archivos que faltan o no cuadran")
        with tarfile.open(TAR) as t:
            t.extractall(DEST, filter="data")
    else:
        print("ya estaba todo extraido")

    ok = True
    for n, size in sorted(ESPERADO.items()):
        p = DEST / n
        real = p.stat().st_size if p.exists() else -1
        marca = "ok" if real == size else "MAL"
        if real != size:
            ok = False
        print(f"  {marca:3s} {n:24s} {real}")
    # comprobacion de contenido, no solo de tamano
    import numpy as np
    x = np.load(DEST / "train_xann.npy", mmap_mode="r")
    y = np.load(DEST / "train_ydec.npy", mmap_mode="r")
    print("  train_xann", x.shape, x.dtype, "| train_ydec", y.shape, y.dtype)
    assert x.shape == (54152, 40, 136) and y.shape == (54152, 15, 41, 7), "forma inesperada"
    print("  muestra finita:", bool(np.isfinite(np.asarray(x[:3])).all()),
          bool(np.isfinite(np.asarray(y[:3])).all()))
    print("espacio:", os.statvfs(DEST).f_bavail * os.statvfs(DEST).f_frsize // 2**30, "GiB libres")
    print("RESULTADO:", "OK" if ok else "INCOMPLETO")


if __name__ == "__main__":
    main()
