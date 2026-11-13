"""Compare frozen and refit forcing representations without training imports."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "original": (ROOT / "src/age_specialists_20261113.py",
                 "186f462a8f36eed4d644098613fddb6faaeeddf7a98672c85c8fb3ab0f075fe7"),
    "refit": (ROOT / "src/full_data_refit_20261113.py",
              "73a96a2d93fca61833e419e4efac6a09563ca8c5f75910b05937b2405ff35fcb"),
}


def load_source(name: str):
    """Import a verified local module without invoking its command line."""
    path, expected = SOURCES[name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Source changed during the review: {name}")
    spec = importlib.util.spec_from_file_location(f"parity_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Require identical schema, dtype, dimensions and exact output bytes."""
    assert "torch" not in sys.modules and "lightgbm" not in sys.modules
    original, refit = load_source("original"), load_source("refit")
    names = original.feature_names("state_shape_climate_delta")
    assert refit.feature_names() == names
    assert len(names) == len(set(names)) == 635
    assert original.BAD == refit.BAD and original.GOOD == refit.GOOD
    rng = np.random.default_rng(20261113)
    random_values = rng.normal(size=(257, 40, 136)).astype(np.float32)
    scales = np.power(10.0, rng.uniform(-5, 5, size=(1, 1, 136)))
    cases = {
        "single_random_float32": random_values[:1],
        "wide_dynamic_range_float32": (random_values * scales).astype(np.float32),
        "random_float64": rng.normal(size=(7, 40, 136)),
        "noncontiguous_time_reversal": random_values[:7, ::-1, :],
        "fortran_layout": np.asfortranarray(random_values[:7]),
        "constant_profiles": np.broadcast_to(np.arange(136, dtype=np.float32), (3, 40, 136)),
        "linear_profiles": np.broadcast_to(
            np.arange(40, dtype=np.float32)[None, :, None] * .125
            + np.arange(136, dtype=np.float32)[None, None, :] * .25,
            (3, 40, 136)),
    }
    records = []
    for name, values in cases.items():
        before = values.tobytes(order="C")
        old, new = original.climate_features(values), refit.climate_features(values)
        assert old.shape == new.shape == (len(values), 519)
        assert old.dtype == new.dtype == np.float32
        assert np.isfinite(old).all() and np.isfinite(new).all()
        assert old.tobytes(order="C") == new.tobytes(order="C")
        assert values.tobytes(order="C") == before
        records.append({"case": name, "input_shape": list(values.shape),
                        "input_dtype": str(values.dtype), "output_shape": list(new.shape),
                        "output_dtype": str(new.dtype), "exact_bytes_equal": True,
                        "maximum_absolute_difference": float(np.max(np.abs(old - new))),
                        "output_sha256": hashlib.sha256(new.tobytes(order="C")).hexdigest()})
    assert "torch" not in sys.modules and "lightgbm" not in sys.modules
    result = {
        "passed": True,
        "scope": "NumPy-only representation parity; no models, training, targets, or GPU access",
        "source_sha256": {name: item[1] for name, item in SOURCES.items()},
        "python": platform.python_version(), "numpy": np.__version__,
        "feature_names_exactly_equal": True, "feature_count": 635,
        "feature_names_sha256": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest(),
        "bad_and_good_indices_exactly_equal": True, "climate_feature_count": 519,
        "cases": records, "torch_imported": False, "lightgbm_imported": False,
    }
    output = Path(__file__).with_name("feature_parity_receipt.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
