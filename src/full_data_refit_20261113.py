#!/usr/bin/env python3
"""Frozen full-data CarbonGlobe refit and independent saved-weight inference.

Production commands are deliberately separate: prepare, train-cpu, train-gpu,
finalize and predict. The CPU training phase never imports torch or initializes
CUDA. Predict needs only the new model bundle and public test inputs; it has no
competition-date cutoff. No command submits predictions or selects parameters
from an audit, a holdout or a leaderboard. Synthetic smoke tests are explicit.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator

sys.dont_write_bytecode = True

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np

STUDY = "full_data_refit_20261113"
CONTRACT = "carbonglobe-full-data-refit-v1"
PRIMITIVES_SHA = "8d03ba50841828ca27e20661b9a02aa6aaeb034409defd6db3245a03f4b6fe5f"
FIXED_LENGTHS_SHA = "06d4beef87f9880e8afe493dff1029d5cbf7a28ad7dd5ab0661eca2972cd5358"
REUSED_SOURCES = {
    "transfer": "230680807f1ec9a3ccd89c8739ac149284cb5c1b5eaa5b11707fcbd213fa2905",
    "auxiliary_shape": "5684f74e7254d045249b68bcec4d0fa889caf4ad50867c7b48e690f16483350c",
    "age_specialists": "186f462a8f36eed4d644098613fddb6faaeeddf7a98672c85c8fb3ab0f075fe7",
    "portable_primitives": PRIMITIVES_SHA,
}
AGES = (1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500)
TARGETS = ("height", "agb")
SCENARIOS = ("ssp126", "ssp585")
SEEDS = (20261113, 20261114)
EPOCHS = {20261113: 4, 20261114: 16}
ROUNDS = ((1191, 1200), (1194, 1199), (1158, 1176), (978, 1196), (1160, 1194),
          (1192, 1200), (897, 1199), (619, 1200), (415, 1199), (1198, 1190),
          (1101, 1200), (743, 1199), (494, 1189), (851, 1186), (1029, 1181))
CONFIGURATION = {"batch_size": 2048, "depth": 3, "dropout": 0.2, "epochs": 30,
                 "learning_rate": 0.0003, "weight_decay": 0.04, "width": 512}
BAD = tuple(range(8, 32)) + (135,)
GOOD = tuple(index for index in range(136) if index not in BAD)
CPU_THREADS = 4
CPU_SECONDS = 45 * 60
GPU_SECONDS = 5 * 60
HOST_MEMORY_BYTES = 6 * 1024 ** 3
MISSION_ROOT = Path("/srv/ai-data-250-a/ieee-carbon")
TRAINING_CUTOFF = dt.datetime(2026, 11, 16, 4, 35, tzinfo=dt.timezone.utc)
PUBLIC_NAMES = ("ssp126_test_y0.npy", "ssp585_test_y0.npy", "test_xann_ssp126.npy",
                "test_xann_ssp585.npy", "sample_submission.csv")
TRAIN_NAMES = ("train_y0.npy", "train_ydec.npy", "train_xann.npy")
AUX_REFERENCE = Path("research/curve2_aux_shape_20261113_v1/run")
AGE_REFERENCE = Path("research/age_specialists_20261113_v1/run")
FAMILIES = ("aux_shape", "state_shape_climate_delta", "blend")


def utc_now() -> str:
    """Return a timezone-qualified timestamp for an evidence record."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    """Hash one artifact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Fingerprint array dtype, dimensions and exact ordered values."""
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(values.dtype).encode() + repr(values.shape).encode())
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a metadata document without interpreting paths as executable code."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    """Commit finite JSON atomically inside the new experiment directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def contained(base: Path, relative: str) -> Path:
    """Resolve a manifest path while refusing an escape from its bundle."""
    candidate = (base / relative).resolve()
    if candidate == base.resolve() or base.resolve() not in candidate.parents:
        raise ValueError(f"Artifact path leaves its bundle: {relative}")
    return candidate


def load_primitives(path: Path | None = None) -> Any:
    """Load only the verified inference-only mathematical helper."""
    path = (path or Path(__file__).with_name("carbon_inference.py")).resolve()
    if sha256(path) != PRIMITIVES_SHA:
        raise RuntimeError("Portable inference primitives differ from the frozen source")
    spec = importlib.util.spec_from_file_location("full_refit_frozen_primitives", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_versions() -> dict[str, str]:
    """Record installed versions without importing torch or querying CUDA."""
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "torch", "lightgbm", "threadpoolctl"):
        result[name] = importlib.metadata.version(name)
    return result


def memory_usage() -> dict[str, int]:
    """Read resident and virtual memory counters on Linux when available."""
    result = {}
    path = Path("/proc/self/status")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                key, value, _ = line.split()
                result[key.rstrip(":")] = int(value) * 1024
    return result


def enforce_cpu_address_space() -> None:
    """Apply the canonical 6 GiB virtual-memory cap to the torch-free CPU phase."""
    if sys.platform != "linux":
        raise RuntimeError("Production CPU training requires the Linux resource contract")
    import resource
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = HOST_MEMORY_BYTES if hard == resource.RLIM_INFINITY else min(HOST_MEMORY_BYTES, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


class Budget:
    """Track cumulative training time or portable elapsed inference time."""

    def __init__(self, seconds: float, *, ledger: Path | None = None,
                 plan_hash: str | None = None, training: bool = False) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Runtime budget must be finite and positive")
        self.maximum, self.ledger, self.plan_hash = float(seconds), ledger, plan_hash
        self.training = training
        self.previous = 0.0
        self.cancelled = None
        if ledger is not None and ledger.exists():
            old = read_json(ledger)
            if old["plan_sha256"] != plan_hash or old["maximum_seconds"] != self.maximum:
                raise RuntimeError("Resource ledger belongs to a different plan")
            if old["session_active"]:
                raise RuntimeError("Prior phase ended uncleanly; its budget must not be silently reset")
            self.previous = float(old["consumed_seconds"])
        self.started = time.monotonic()
        self.last_persist = self.started
        self.check()
        self.persist(True)

    def elapsed(self) -> float:
        """Return this phase's cumulative elapsed time, including clean resumes."""
        return self.previous + time.monotonic() - self.started

    def check(self) -> None:
        """Enforce finite resource boundaries without evaluating model quality."""
        if self.cancelled is not None:
            raise InterruptedError(f"Stop requested by signal {self.cancelled}")
        if self.elapsed() >= self.maximum:
            raise TimeoutError(f"Phase exceeded its {self.maximum:g}-second budget")
        if self.training and dt.datetime.now(dt.timezone.utc) >= TRAINING_CUTOFF:
            raise TimeoutError("The mission training cutoff has been reached")
        if memory_usage().get("VmRSS", 0) > HOST_MEMORY_BYTES:
            raise MemoryError("Phase exceeded its 6 GiB resident-memory limit")
        if self.ledger is not None and time.monotonic() - self.last_persist >= 15:
            self.persist(True)

    def request_stop(self, signum: int, _frame: Any) -> None:
        """Defer cancellation to the next numerical boundary."""
        self.cancelled = signum

    def persist(self, active: bool) -> None:
        """Save cumulative time without forgetting an interrupted process."""
        if self.ledger is not None:
            atomic_json(self.ledger, {"plan_sha256": self.plan_hash,
                        "maximum_seconds": self.maximum, "consumed_seconds": self.elapsed(),
                        "session_active": active, "memory": memory_usage(), "updated_utc": utc_now()})
        self.last_persist = time.monotonic()

    @contextlib.contextmanager
    def signals(self) -> Iterator[None]:
        """Install and restore process stop handlers for one phase."""
        previous = {s: signal.signal(s, self.request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            yield
        finally:
            self.persist(False)
            for signum, handler in previous.items():
                signal.signal(signum, handler)


@contextlib.contextmanager
def phase_lock(output: Path, name: str) -> Iterator[None]:
    """Prevent duplicate work within one phase without acquiring the GPU lock twice."""
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / name).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {name}") from exc
        yield


def tree_parameters(age_index: int, target_index: int) -> dict[str, Any]:
    """Return exactly the original requested LightGBM parameters."""
    seed = 20261113 + 2 * age_index + target_index
    return {"objective": "regression", "metric": "None", "boosting_type": "gbdt",
            "device_type": "cpu", "num_threads": 4, "deterministic": True,
            "force_col_wise": True, "force_row_wise": False, "num_leaves": 31,
            "max_depth": -1, "min_data_in_leaf": 128, "min_sum_hessian_in_leaf": 1e-3,
            "learning_rate": 0.04, "feature_fraction": 0.9, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l1": 0.0, "lambda_l2": 1.0,
            "max_bin": 255, "min_data_in_bin": 3, "bin_construct_sample_cnt": 200000,
            "histogram_pool_size": 256.0, "feature_pre_filter": True,
            "use_missing": True, "zero_as_missing": False, "extra_trees": False,
            "seed": seed, "data_random_seed": seed, "feature_fraction_seed": seed,
            "bagging_seed": seed, "extra_seed": seed, "verbosity": -1}


def feature_names() -> list[str]:
    """Preserve the original 635 names and operation order."""
    names = [f"frozen_state_shape_{i:03d}" for i in range(116)]
    names += [f"annual_mean_x{i:03d}" for i in GOOD]
    for operation in ("population_std", "ols_slope", "last10_minus_first10"):
        names += [f"annual_{operation}_x{i:03d}" for i in range(136)]
    return names


def climate_features(annual: np.ndarray) -> np.ndarray:
    """Compute the unchanged 519 forcing summaries in float64, then float32."""
    values = np.asarray(annual, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (40, 136) or not np.isfinite(values).all():
        raise ValueError("Annual forcing must be finite and have shape (sites,40,136)")
    centred_time = np.arange(40, dtype=np.float64) - 19.5
    mean = values[:, :, GOOD].mean(axis=1)
    deviation = values.std(axis=1, ddof=0)
    slope = np.einsum("stc,t->sc", values, centred_time, optimize=False) / np.dot(centred_time, centred_time)
    difference = values[:, 30:40].mean(axis=1) - values[:, :10].mean(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.concatenate([mean, deviation, slope, difference], axis=1).astype(np.float32)
    if result.shape != (len(values), 519) or not np.isfinite(result).all():
        raise ValueError("Annual summaries overflowed or violated the 519-feature contract")
    return result


def features_at_age(state: np.ndarray, age_index: int, helper: Any, budget: Budget) -> np.ndarray:
    """Produce raw 116-column state features without the MLP scaler or clamp."""
    if not 0 <= age_index < 15 or state.shape[1:] != (15, 7):
        raise ValueError("Invalid state shape or age index")
    result = np.empty((len(state), 116), dtype=np.float32)
    for start in range(0, len(state), 256):
        budget.check()
        values = state[start:start + 256]
        result[start:start + len(values)] = helper.feature_matrix(values, helper.PRIMITIVES).reshape(-1, 15, 116)[:, age_index]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite state features")
    return result


def load_fixed_lengths(path: Path) -> dict[str, Any]:
    """Require the exact reviewed manifest and all 32 fixed durations."""
    if sha256(path) != FIXED_LENGTHS_SHA:
        raise RuntimeError("The fixed-length proposal manifest differs from the reviewed document")
    value = read_json(path)
    if value["source_training_sites"] != 32490 or value["proposed_training_sites"] != 54152:
        raise ValueError("Training population contract changed")
    aux = sorted(value["auxiliary_models"], key=lambda r: r["seed"])
    if [r["seed"] for r in aux] != list(SEEDS):
        raise ValueError("Auxiliary seed set changed")
    for record in aux:
        if record["fixed_training_epochs"] != EPOCHS[record["seed"]] or record["cosine_T_max"] != 30:
            raise ValueError("Fixed auxiliary epochs or scheduler changed")
    trees = sorted(value["age_specialist_models"], key=lambda r: (r["age_index"], r["target_index"]))
    if [(r["age_index"], r["target_index"]) for r in trees] != [(a, t) for a in range(15) for t in range(2)]:
        raise ValueError("Expected one specialist per age and target")
    for record in trees:
        age, target = record["age_index"], record["target_index"]
        if (record["best_iteration"] != ROUNDS[age][target]
                or record["parameters_requested"] != tree_parameters(age, target)):
            raise ValueError("A fixed tree duration or requested parameter changed")
    return value


def verify_data(data: Path, expected: dict[str, Any], names: tuple[str, ...], budget: Budget) -> None:
    """Hash only the explicitly requested training or public inference inputs."""
    for name in names:
        budget.check()
        path = data / name
        if not path.is_file() or sha256(path) != expected[name]["sha256"]:
            raise RuntimeError(f"Input provenance mismatch: {name}")


def open_array(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    """Open a fixed-shape numeric cache read-only, rejecting pickled objects."""
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != shape or value.dtype != np.float32:
        raise ValueError(f"Unexpected shape/dtype for {path.name}: {value.shape}/{value.dtype}")
    return value


def source_scaler(path: Path, expected_sha: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract the two original float32 scaler arrays on CPU without CUDA initialization."""
    if sha256(path) != expected_sha:
        raise RuntimeError("Original auxiliary checkpoint hash mismatch")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Preparation must start without initialized CUDA")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload["seed"] != seed or payload["code_sha256"] != REUSED_SOURCES["auxiliary_shape"]
            or payload["engine_sha256"] != REUSED_SOURCES["transfer"]
            or payload["configuration"] != {**CONFIGURATION, "patience": 5}
            or payload["best_epoch"] + 1 != EPOCHS[seed]):
        raise RuntimeError("Original auxiliary checkpoint contract changed")
    mean, std = payload["feature_mean"], payload["feature_std"]
    for value in (mean, std):
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (116,) or not np.isfinite(value).all():
            raise ValueError("Original feature scaler is not finite float32[116]")
    if np.any(std <= 0) or torch.cuda.is_initialized():
        raise RuntimeError("Invalid scaler or unexpected CUDA initialization")
    return mean.copy(), std.copy()


def prepare(root: Path, output: Path, manifest_path: Path, helper: Any, budget: Budget) -> dict[str, Any]:
    """Freeze new provenance and original scalers; never fit or evaluate a model."""
    if output.exists():
        raise FileExistsError("Prepare requires a new refit output directory")
    for protected in (root / "data", root / AUX_REFERENCE, root / AGE_REFERENCE):
        protected = protected.resolve()
        if output.resolve() == protected or protected in output.resolve().parents:
            raise ValueError("Refit outputs must be separate from data and original study artifacts")
    fixed = load_fixed_lengths(manifest_path)
    aux_plan = read_json(root / AUX_REFERENCE / "preregistered_plan.json")
    age_plan = read_json(root / AGE_REFERENCE / "preregistered_plan.json")
    for name, relative in (("aux_shape", AUX_REFERENCE), ("age_specialists", AGE_REFERENCE)):
        if sha256(root / relative / "preregistered_plan.json") != fixed["source_plan_sha256"][name]:
            raise RuntimeError(f"Original plan changed: {name}")
    sources = {}
    for name in TRAIN_NAMES + PUBLIC_NAMES:
        expected = age_plan["sources"].get(name, aux_plan["sources"].get(name))
        if not expected or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]):
            raise ValueError(f"Missing original input provenance: {name}")
        sources[name] = {"sha256": expected["sha256"], "bytes": (root / "data" / name).stat().st_size}
    verify_data(root / "data", sources, TRAIN_NAMES + PUBLIC_NAMES, budget)
    for record in fixed["age_specialist_models"]:
        for field, hash_field in (("source_record", "source_record_sha256"), ("source_model", "source_model_sha256")):
            if sha256(contained(root, record[field])) != record[hash_field]:
                raise RuntimeError("An original specialist artifact changed")
    output.mkdir(parents=True)
    (output / "scalers").mkdir()
    scalers = {}
    for record in fixed["auxiliary_models"]:
        seed = record["seed"]
        if sha256(contained(root, record["source_record"])) != record["source_record_sha256"]:
            raise RuntimeError("Original auxiliary training record changed")
        mean, std = source_scaler(contained(root, record["source_checkpoint"]), record["source_checkpoint_sha256"], seed)
        destination = output / "scalers" / f"seed{seed}.npz"
        np.savez(destination, feature_mean=mean, feature_std=std)
        with np.load(destination, allow_pickle=False) as saved:
            if not np.array_equal(saved["feature_mean"], mean) or not np.array_equal(saved["feature_std"], std):
                raise RuntimeError("Copied scaler failed its exact-array check")
        scalers[str(seed)] = {"path": str(destination.relative_to(output)), "sha256": sha256(destination),
                              "mean_sha256": array_sha256(mean), "std_sha256": array_sha256(std)}
    shutil.copyfile(manifest_path, output / "fixed_lengths.json")
    plan = {"contract": CONTRACT, "study": STUDY, "created_utc": utc_now(), "synthetic_fixture": False,
            "refit_code_sha256": sha256(Path(__file__)), "reused_sources_sha256": REUSED_SOURCES,
            "fixed_lengths_sha256": FIXED_LENGTHS_SHA, "training_sites": 54152,
            "training_rows_mlp": 812280, "training_selection": "none; fixed original tune-selected lengths",
            "configuration": CONFIGURATION, "cosine_T_max": 30,
            "epochs": {str(k): v for k, v in EPOCHS.items()}, "tree_rounds": [list(r) for r in ROUNDS],
            "tree_parameters": [tree_parameters(a, t) for a in range(15) for t in range(2)],
            "feature_dimensions": {"aux_shape": 116, "state_shape_climate_delta": 635},
            "feature_names": feature_names(), "decoder": "community_affine",
            "scales": helper.SCALES.tolist(), "scalers": scalers, "sources": sources,
            "environment": package_versions(), "blend": {"aux_shape": 0.5, "state_shape_climate_delta": 0.5},
            "cpu_budget_seconds": CPU_SECONDS, "gpu_budget_seconds": GPU_SECONDS,
            "host_memory_bytes": HOST_MEMORY_BYTES,
            "reference_metrics": fixed["original_evaluation_references"],
            "validation_statement": "All historical sites train; no refit holdout or OOF score is computed"}
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "prepared_receipt.json", {"prepared_utc": utc_now(),
                "plan_sha256": sha256(output / "plan.json"), "training_started": False,
                "cuda_initialized": False, "elapsed_seconds": budget.elapsed()})
    return {"prepared": True, "output": str(output), "plan_sha256": sha256(output / "plan.json")}


def verify_plan(output: Path, *, training: bool = False) -> dict[str, Any]:
    """Validate this new checkpoint contract without invoking an old study verifier."""
    plan = read_json(output / "plan.json")
    if (plan.get("contract") != CONTRACT or plan.get("study") != STUDY
            or plan.get("refit_code_sha256") != sha256(Path(__file__))
            or plan.get("reused_sources_sha256") != REUSED_SOURCES
            or plan.get("synthetic_fixture") is not False):
        raise RuntimeError("Refit plan/source contract mismatch")
    if (plan["configuration"] != CONFIGURATION or plan["cosine_T_max"] != 30
            or plan["epochs"] != {str(k): v for k, v in EPOCHS.items()}
            or plan["tree_rounds"] != [list(r) for r in ROUNDS]
            or plan["training_sites"] != 54152 or plan["feature_names"] != feature_names()):
        raise RuntimeError("A frozen refit decision changed")
    if training:
        if package_versions() != plan["environment"]:
            raise RuntimeError("Refit training environment changed after preparation")
        load_fixed_lengths(output / "fixed_lengths.json")
    return plan


def scaler_arrays(output: Path, plan: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the preserved scaler without fitting or changing its dtype."""
    entry = plan["scalers"][str(seed)]
    path = contained(output, entry["path"])
    if sha256(path) != entry["sha256"]:
        raise RuntimeError("Frozen scaler file changed")
    with np.load(path, allow_pickle=False) as saved:
        mean, std = saved["feature_mean"].copy(), saved["feature_std"].copy()
    if (mean.dtype != np.float32 or std.dtype != np.float32 or mean.shape != (116,) or std.shape != (116,)
            or array_sha256(mean) != entry["mean_sha256"] or array_sha256(std) != entry["std_sha256"]):
        raise RuntimeError("Scaler arrays changed precision or values")
    return mean, std


def cache_features(root: Path, output: Path, plan: dict[str, Any], helper: Any,
                   kind: str, budget: Budget) -> np.ndarray:
    """Build one bounded immutable cache, independently for the CPU and GPU phases."""
    if kind not in ("auxiliary", "climate"):
        raise ValueError(kind)
    folder = output / "cache"
    folder.mkdir(exist_ok=True)
    path, receipt = folder / f"{kind}.npy", folder / f"{kind}.json"
    shape = (812280, 116) if kind == "auxiliary" else (54152, 519)
    if receipt.exists():
        record = read_json(receipt)
        if record["plan_sha256"] != sha256(output / "plan.json") or sha256(path) != record["sha256"]:
            raise RuntimeError("Feature cache provenance mismatch")
        return open_array(path, shape)
    if path.exists():
        raise RuntimeError("Feature cache exists without its complete receipt")
    source_name = "train_y0.npy" if kind == "auxiliary" else "train_xann.npy"
    source_shape = (54152, 15, 7) if kind == "auxiliary" else (54152, 40, 136)
    source = open_array(root / "data" / source_name, source_shape)
    temporary = path.with_suffix(".partial.npy")
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, 54152, 256):
        budget.check()
        stop = min(start + 256, 54152)
        if kind == "auxiliary":
            matrix[start * 15:stop * 15] = helper.feature_matrix(source[start:stop], helper.PRIMITIVES)
        else:
            matrix[start:stop] = climate_features(source[start:stop])
    matrix.flush()
    del matrix, source
    os.replace(temporary, path)
    atomic_json(receipt, {"plan_sha256": sha256(output / "plan.json"), "sha256": sha256(path), "shape": list(shape)})
    return open_array(path, shape)


def model_specs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe exactly two neural models and thirty age/target specialists."""
    records = []
    for seed in SEEDS:
        records.append({"id": f"aux_seed{seed}", "family": "aux_shape", "seed": seed,
                        "fixed_length": plan["epochs"][str(seed)],
                        "model_path": f"models/aux_shape/seed{seed}.pt",
                        "record_path": f"models/aux_shape/seed{seed}.json"})
    for age_index, age in enumerate(AGES):
        for target_index, target in enumerate(TARGETS):
            prefix = f"models/state_shape_climate_delta/age_{age:03d}/{target}"
            records.append({"id": f"age{age:03d}_{target}", "family": "state_shape_climate_delta",
                            "age": age, "age_index": age_index, "target": target,
                            "target_index": target_index, "seed": 20261113 + 2 * age_index + target_index,
                            "fixed_length": plan["tree_rounds"][age_index][target_index],
                            "model_path": prefix + ".txt", "record_path": prefix + ".json"})
    return records


def record_base(plan: dict[str, Any], plan_hash: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Identify new training honestly rather than claiming the old runner produced it."""
    return {**spec, "contract": CONTRACT, "study": STUDY,
            "refit_code_sha256": plan["refit_code_sha256"], "plan_sha256": plan_hash,
            "reused_sources_sha256": REUSED_SOURCES, "training_sites": plan["training_sites"],
            "synthetic_fixture": plan.get("synthetic_fixture", False),
            "new_parameter_selection": False, "heldout_metric_computed": False}


def completed_model(output: Path, plan: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any] | None:
    """Resume only an intact completed model; never overwrite an orphaned weight file."""
    record_path = contained(output, spec["record_path"])
    model_path = contained(output, spec["model_path"])
    if not record_path.exists():
        if model_path.exists():
            raise RuntimeError(f"Weight file has no completion receipt: {model_path}")
        return None
    record = read_json(record_path)
    expected = record_base(plan, sha256(output / "plan.json"), spec)
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Model metadata changed: {record_path}")
    if (record.get("complete") is not True or record.get("executed_length") != spec["fixed_length"]
            or not record.get("reload_check", {}).get("passed")
            or sha256(model_path) != record["model_sha256"]):
        raise RuntimeError(f"Completed model failed its integrity contract: {model_path}")
    return record


def fixed_tree_fit(matrix: np.ndarray, residual: np.ndarray, params: dict[str, Any],
                   rounds: int, budget: Budget) -> Any:
    """Train fixed rounds with no validation data, evaluation callback or early stopping."""
    import lightgbm as lgb
    if matrix.shape[1] != 635 or not np.isfinite(matrix).all() or not np.isfinite(residual).all():
        raise ValueError("Invalid full-data specialist inputs")
    if any(key in params for key in ("num_iterations", "num_boost_round", "early_stopping_round", "early_stopping_rounds")):
        raise ValueError("Requested parameters must not override the fixed duration")
    training = lgb.Dataset(matrix, label=residual, feature_name=feature_names(), free_raw_data=True)
    calls = []

    def guard(environment: Any) -> None:
        budget.check()
        calls.append(int(environment.iteration))

    guard.order = 5
    guard.before_iteration = True
    booster = lgb.train(params, training, num_boost_round=rounds, callbacks=[guard])
    if (booster.current_iteration() != rounds or booster.num_trees() != rounds
            or calls != list(range(rounds))):
        raise RuntimeError("LightGBM stopped before the fixed number of real trees; model is not complete")
    return booster


def save_tree_fit(output: Path, plan: dict[str, Any], spec: dict[str, Any], matrix: np.ndarray,
                  residual: np.ndarray, budget: Budget) -> dict[str, Any]:
    """Save one actual booster and verify its numerical round-trip from disk."""
    import lightgbm as lgb
    prior = completed_model(output, plan, spec)
    if prior is not None:
        return prior
    started = time.monotonic()
    params = tree_parameters(spec["age_index"], spec["target_index"])
    booster = fixed_tree_fit(matrix, residual, params, spec["fixed_length"], budget)
    probe = booster.predict(matrix[:512], num_iteration=spec["fixed_length"], num_threads=CPU_THREADS)
    path = contained(output, spec["model_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.txt")
    booster.save_model(str(temporary), num_iteration=spec["fixed_length"])
    restored = lgb.Booster(model_file=str(temporary))
    if (restored.num_trees() != spec["fixed_length"] or restored.num_feature() != 635
            or restored.feature_name() != feature_names()):
        raise RuntimeError("Saved booster changed its fixed length or feature contract")
    reloaded = restored.predict(matrix[:512], num_iteration=spec["fixed_length"], num_threads=CPU_THREADS)
    maximum = float(np.max(np.abs(probe - reloaded)))
    if not np.allclose(probe, reloaded, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Saved booster failed the fixed reload tolerance")
    os.replace(temporary, path)
    record = {**record_base(plan, sha256(output / "plan.json"), spec), "complete": True,
              "executed_length": spec["fixed_length"], "saved_trees": restored.num_trees(),
              "feature_dimension": 635, "parameters_requested": params,
              "parameters_effective": booster.params, "model_sha256": sha256(path),
              "training_seconds": time.monotonic() - started, "completed_utc": utc_now(),
              "reload_check": {"passed": True, "maximum_absolute_difference": maximum,
                               "atol": 1e-12, "rtol": 1e-12}}
    atomic_json(contained(output, spec["record_path"]), record)
    print(json.dumps({"completed": spec["id"], "rounds": spec["fixed_length"],
                      "seconds": record["training_seconds"]}), flush=True)
    return record


def train_cpu(root: Path, output: Path, helper: Any) -> dict[str, Any]:
    """Fit the thirty CPU specialists on all source sites without importing torch."""
    if "torch" in sys.modules:
        raise RuntimeError("The production CPU phase must start in a fresh torch-free process")
    enforce_cpu_address_space()
    plan = verify_plan(output, training=True)
    with phase_lock(output, "study_cpu.lock"):
        budget = Budget(CPU_SECONDS, ledger=output / "cpu_resource_budget.json",
                        plan_hash=sha256(output / "plan.json"), training=True)
        with budget.signals():
            verify_data(root / "data", plan["sources"], TRAIN_NAMES, budget)
            climate = cache_features(root, output, plan, helper, "climate", budget)
            state = open_array(root / "data/train_y0.npy", (54152, 15, 7))
            specs = model_specs(plan)[2:]
            records = []
            for age_index in range(15):
                selected = specs[age_index * 2:age_index * 2 + 2]
                if all(completed_model(output, plan, spec) is not None for spec in selected):
                    records.extend(completed_model(output, plan, spec) for spec in selected)
                    continue
                budget.check()
                base = features_at_age(state, age_index, helper, budget)
                matrix = np.ascontiguousarray(np.concatenate([base, climate], axis=1), dtype=np.float32)
                del base
                source = open_array(root / "data/train_ydec.npy", (54152, 15, 41, 7))
                target = np.asarray(source[:, age_index, 40, :2], dtype=np.float32) / helper.SCALES
                del source
                anchor = helper.anchor_at_40(state)[:, age_index, :2] / helper.SCALES
                if not np.isfinite(target).all():
                    raise ValueError("Every historical target must be finite; no site may be dropped")
                for spec in selected:
                    column = spec["target_index"]
                    residual = target[:, column].astype(np.float64) - anchor[:, column]
                    records.append(save_tree_fit(output, plan, spec, matrix, residual, budget))
                del matrix, target, anchor
                gc.collect()
            if "torch" in sys.modules:
                raise RuntimeError("A CPU dependency unexpectedly imported torch")
            result = {"phase": "cpu", "complete": True, "models": len(records),
                      "training_sites": 54152, "torch_imported": False, "cuda_initialized": False,
                      "heldout_metric_computed": False, "elapsed_seconds": budget.elapsed(),
                      "memory": memory_usage(), "completed_utc": utc_now()}
            atomic_json(output / "cpu_phase_receipt.json", result)
            return result


def fixed_mlp_fit(matrix: np.ndarray, labels: np.ndarray, anchors: np.ndarray, seed: int,
                  epochs: int, device: str, helper: Any, budget: Budget) -> tuple[Any, dict[str, Any]]:
    """Fit the original optimization sequence for a fixed number of epochs."""
    import torch
    if epochs != EPOCHS[seed] or matrix.dtype != np.float32 or matrix.shape[1] != 116:
        raise ValueError("The fixed MLP duration or input dtype/dimension changed")
    if not all(np.isfinite(value).all() for value in (matrix, labels, anchors)):
        raise ValueError("Nonfinite MLP fitting arrays")
    torch.set_num_threads(CPU_THREADS)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda = device.startswith("cuda:")
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    x, y, anchor = torch.from_numpy(matrix), torch.from_numpy(labels), torch.from_numpy(anchors)
    model = helper.make_model(torch, 116, CONFIGURATION).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIGURATION["learning_rate"],
                                 weight_decay=CONFIGURATION["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    scaler = torch.amp.GradScaler("cuda") if cuda else None
    history, steps = [], 0
    for epoch in range(epochs):
        budget.check()
        model.train()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        expected_lr = CONFIGURATION["learning_rate"] * (1 + math.cos(math.pi * epoch / 30)) / 2
        if not math.isclose(learning_rate, expected_lr, rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError("Cosine learning-rate sequence changed")
        loss_sum, count = 0.0, 0
        for rows in torch.randperm(len(x)).split(CONFIGURATION["batch_size"]):
            budget.check()
            xb, yb, ab = x[rows].to(device), y[rows].to(device), anchor[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if cuda else contextlib.nullcontext()
            with context:
                predicted = model(xb).float() + ab
                loss = (predicted - yb).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MLP training loss")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            steps += 1
            loss_sum += float(loss.item()) * len(rows)
            count += len(rows)
        schedule.step()
        history.append({"epoch_zero_based": epoch, "learning_rate": learning_rate,
                        "training_loss_only": loss_sum / count})
        print(json.dumps({"seed": seed, "fixed_epoch": epoch + 1, "epochs": epochs}), flush=True)
    if steps != epochs * math.ceil(len(matrix) / CONFIGURATION["batch_size"]):
        raise RuntimeError("The fixed optimizer-step count was not completed")
    model.eval()
    return model, {"epochs_executed": epochs, "optimizer_steps": steps,
                   "cosine_T_max": schedule.T_max, "history": history,
                   "training_precision": "float16_autocast" if cuda else "float32_synthetic_smoke"}


def save_mlp_fit(output: Path, plan: dict[str, Any], spec: dict[str, Any], matrix: np.ndarray,
                 labels: np.ndarray, anchors: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 device: str, helper: Any, budget: Budget) -> dict[str, Any]:
    """Save new neural weights, original scalers and this runner's true provenance."""
    import torch
    prior = completed_model(output, plan, spec)
    if prior is not None:
        return prior
    started = time.monotonic()
    model, training = fixed_mlp_fit(matrix, labels, anchors, spec["seed"], spec["fixed_length"], device, helper, budget)
    probe = helper.predict_tensor(model, torch.from_numpy(matrix[:128]), anchors[:128], torch, device)
    payload = {**record_base(plan, sha256(output / "plan.json"), spec),
               "configuration": CONFIGURATION, "feature_dimension": 116,
               "feature_mean": mean.copy(), "feature_std": std.copy(),
               "scaler_mean_sha256": array_sha256(mean), "scaler_std_sha256": array_sha256(std),
               "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
               "fixed_epochs": spec["fixed_length"], "cosine_T_max": 30, "saved_utc": utc_now()}
    path = contained(output, spec["model_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    restored_payload = torch.load(temporary, map_location="cpu", weights_only=False)
    restored = helper.make_model(torch, 116, CONFIGURATION).to(device)
    restored.load_state_dict(restored_payload["model_state"], strict=True)
    reloaded = helper.predict_tensor(restored, torch.from_numpy(matrix[:128]), anchors[:128], torch, device)
    maximum = float(np.max(np.abs(probe - reloaded)))
    if not np.allclose(probe, reloaded, atol=1e-6, rtol=1e-6):
        raise RuntimeError("New MLP weights failed the fixed reload tolerance")
    if not np.array_equal(restored_payload["feature_mean"], mean) or not np.array_equal(restored_payload["feature_std"], std):
        raise RuntimeError("Scaler values changed during checkpoint storage")
    os.replace(temporary, path)
    record = {**record_base(plan, sha256(output / "plan.json"), spec), "complete": True,
              "executed_length": training["epochs_executed"], "training": training,
              "model_sha256": sha256(path), "feature_dimension": 116,
              "scaler_mean_sha256": array_sha256(mean), "scaler_std_sha256": array_sha256(std),
              "completed_utc": utc_now(), "training_seconds": time.monotonic() - started,
              "reload_check": {"passed": True, "maximum_absolute_difference": maximum,
                               "atol": 1e-6, "rtol": 1e-6}}
    atomic_json(contained(output, spec["record_path"]), record)
    del model, restored, restored_payload, payload
    gc.collect()
    if device.startswith("cuda:"):
        torch.cuda.empty_cache()
    return record


def train_gpu(root: Path, output: Path, helper: Any) -> dict[str, Any]:
    """Fit only the two fixed MLP seeds under the canonical reserved-GPU lease."""
    plan = verify_plan(output, training=True)
    with phase_lock(output, "refit_gpu_phase.lock"):
        budget = Budget(GPU_SECONDS, ledger=output / "gpu_resource_budget.json",
                        plan_hash=sha256(output / "plan.json"), training=True)
        with helper.execution_context(MISSION_ROOT, "mission", "cuda:0", budget):
            verify_data(root / "data", plan["sources"], ("train_y0.npy", "train_ydec.npy"), budget)
            cache = cache_features(root, output, plan, helper, "auxiliary", budget)
            source = open_array(root / "data/train_ydec.npy", (54152, 15, 41, 7))
            labels = (np.asarray(source[:, :, 40, :2], dtype=np.float32) / helper.SCALES).reshape(-1, 2)
            del source
            state = open_array(root / "data/train_y0.npy", (54152, 15, 7))
            anchors = (helper.anchor_at_40(state)[:, :, :2] / helper.SCALES).reshape(-1, 2)
            del state
            records = []
            for spec in model_specs(plan)[:2]:
                prior = completed_model(output, plan, spec)
                if prior is not None:
                    records.append(prior)
                    continue
                mean, std = scaler_arrays(output, plan, spec["seed"])
                matrix = np.empty(cache.shape, dtype=np.float32)
                for start in range(0, len(cache), 8192):
                    budget.check()
                    part = cache[start:start + 8192]
                    matrix[start:start + len(part)] = np.clip((part - mean) / std, -8, 8)
                records.append(save_mlp_fit(output, plan, spec, matrix, labels, anchors,
                                            mean, std, "cuda:0", helper, budget))
                del matrix
            result = {"phase": "gpu", "complete": True, "models": len(records),
                      "training_sites": 54152, "heldout_metric_computed": False,
                      "elapsed_seconds": budget.elapsed(), "memory": memory_usage(), "completed_utc": utc_now()}
            atomic_json(output / "gpu_phase_receipt.json", result)
            return result


def finalize(output: Path, *, allow_synthetic: bool = False) -> dict[str, Any]:
    """Create a portable inference manifest only after all 32 weight checks pass."""
    plan = read_json(output / "plan.json") if allow_synthetic else verify_plan(output)
    if plan.get("synthetic_fixture", False) and not allow_synthetic:
        raise RuntimeError("Synthetic artifacts cannot be finalized as production models")
    records = []
    for spec in model_specs(plan):
        record = completed_model(output, plan, spec)
        if record is None:
            raise RuntimeError(f"Missing completed model: {spec['id']}")
        records.append({**spec, "model_sha256": record["model_sha256"],
                        "record_sha256": sha256(contained(output, spec["record_path"]))})
    if len(records) != 32:
        raise RuntimeError("The full refit must contain exactly 32 actual models")
    bundle = output / "inference_bundle"
    bundle.mkdir(exist_ok=True)
    artifact_names = ["plan.json", "fixed_lengths.json"]
    artifact_names += [record[key] for record in records for key in ("model_path", "record_path")]
    artifact_names += [value["path"] for value in plan["scalers"].values()]
    for name in artifact_names:
        source, destination = contained(output, name), contained(bundle, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(source):
            raise RuntimeError(f"An existing portable artifact differs: {name}")
        if not destination.exists():
            shutil.copyfile(source, destination)
    code_dir = bundle / "code"
    code_dir.mkdir(exist_ok=True)
    for path in (Path(__file__), Path(__file__).with_name("carbon_inference.py")):
        destination = code_dir / path.name
        if destination.exists() and sha256(destination) != sha256(path):
            raise RuntimeError("Bundled code already exists with different contents")
        if not destination.exists():
            shutil.copyfile(path, destination)
    allowed = set(artifact_names) | {"code/full_data_refit_20261113.py", "code/carbon_inference.py", "bundle_manifest.json"}
    unexpected = [str(path.relative_to(bundle)) for path in bundle.rglob("*")
                  if path.is_file() and str(path.relative_to(bundle)).replace(os.sep, "/") not in allowed]
    if unexpected:
        raise RuntimeError(f"Portable bundle contains unexpected artifacts: {unexpected}")
    manifest = {"contract": CONTRACT, "study": STUDY, "created_utc": utc_now(),
                "synthetic_fixture": plan.get("synthetic_fixture", False),
                "refit_code_sha256": plan["refit_code_sha256"], "reused_sources_sha256": REUSED_SOURCES,
                "plan_path": "plan.json", "plan_sha256": sha256(output / "plan.json"),
                "models": records, "public_inputs": {n: plan["sources"][n] for n in PUBLIC_NAMES},
                "decoder": "community_affine", "blend": plan["blend"],
                "scalers": plan["scalers"], "training_sites": plan["training_sites"],
                "training_environment": plan["environment"],
                "inference_requires_training_arrays": False,
                "inference_entrypoint": "code/full_data_refit_20261113.py predict",
                "numerical_conventions": {"mlp_site_chunks": 128, "mlp_inner_rows": 8192,
                    "gbdt_site_chunks": 4096, "mlp_seed_mean": "float64 accumulation then float32",
                    "gbdt_prediction": "float64", "blend": "0.5*A.astype(float64)+0.5*G", "csv_format": ".9g"}}
    path = bundle / "bundle_manifest.json"
    if path.exists():
        previous = read_json(path)
        manifest["created_utc"] = previous["created_utc"]
        if previous != manifest:
            raise RuntimeError("Existing final manifest differs; no artifact was replaced")
    else:
        atomic_json(path, manifest)
    result = {"complete": True, "models": 32, "bundle_manifest": str(path), "sha256": sha256(path),
              "training_caches_included": False, "public_test_inputs_included": False}
    atomic_json(output / "finalized_receipt.json", result)
    return result


def read_bundle(folder: Path, family: str, *, allow_synthetic: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify new model metadata and weights without accessing any training source."""
    manifest = read_json(folder / "bundle_manifest.json")
    if (manifest.get("contract") != CONTRACT or manifest.get("study") != STUDY
            or manifest.get("refit_code_sha256") != sha256(Path(__file__))
            or manifest.get("reused_sources_sha256") != REUSED_SOURCES):
        raise RuntimeError("Bundle does not identify this full-refit inference contract")
    if manifest.get("synthetic_fixture", False) and not allow_synthetic:
        raise RuntimeError("Synthetic smoke weights are not production checkpoints")
    plan_path = contained(folder, manifest["plan_path"])
    if sha256(plan_path) != manifest["plan_sha256"]:
        raise RuntimeError("Bundled plan hash mismatch")
    plan = read_json(plan_path)
    if (plan["refit_code_sha256"] != manifest["refit_code_sha256"]
            or plan["reused_sources_sha256"] != REUSED_SOURCES
            or plan.get("synthetic_fixture", False) != manifest.get("synthetic_fixture", False)
            or plan["configuration"] != CONFIGURATION or plan["cosine_T_max"] != 30
            or plan["feature_names"] != feature_names()
            or plan["blend"] != {"aux_shape": .5, "state_shape_climate_delta": .5}
            or manifest["blend"] != plan["blend"] or manifest["decoder"] != "community_affine"):
        raise RuntimeError("Bundled arithmetic or provenance contract changed")
    if not plan.get("synthetic_fixture", False):
        if (plan["training_sites"] != 54152 or plan["epochs"] != {str(k): v for k, v in EPOCHS.items()}
                or plan["tree_rounds"] != [list(r) for r in ROUNDS]):
            raise RuntimeError("Production bundle changed the fixed refit durations")
    expected = model_specs(plan)
    if len(manifest["models"]) != 32 or [r["id"] for r in manifest["models"]] != [r["id"] for r in expected]:
        raise RuntimeError("Bundle must declare exactly the ordered 32 models")
    needed = {family} if family in FAMILIES[:2] else set(FAMILIES[:2])
    for entry, spec in zip(manifest["models"], expected):
        if any(entry.get(key) != value for key, value in spec.items()):
            raise RuntimeError("A model declaration differs from its fixed plan")
        if entry["family"] not in needed:
            continue
        model = contained(folder, entry["model_path"])
        record_path = contained(folder, entry["record_path"])
        if sha256(model) != entry["model_sha256"] or sha256(record_path) != entry["record_sha256"]:
            raise RuntimeError(f"Model integrity failure: {entry['id']}")
        record = read_json(record_path)
        base = record_base(plan, manifest["plan_sha256"], spec)
        if (any(record.get(key) != value for key, value in base.items())
                or record.get("complete") is not True or record.get("executed_length") != spec["fixed_length"]
                or record.get("model_sha256") != entry["model_sha256"]
                or not record.get("reload_check", {}).get("passed")):
            raise RuntimeError("Saved model receipt changed")
    return manifest, plan


def load_new_mlp(folder: Path, manifest: dict[str, Any], plan: dict[str, Any],
                 entry: dict[str, Any], device: str, helper: Any) -> tuple[Any, dict[str, Any]]:
    """Restore an actual refit checkpoint; old engine identifiers are not accepted."""
    import torch
    path = contained(folder, entry["model_path"])
    if sha256(path) != entry["model_sha256"]:
        raise RuntimeError("Neural weight file changed before loading")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec = next(s for s in model_specs(plan) if s["id"] == entry["id"])
    base = record_base(plan, manifest["plan_sha256"], spec)
    if (any(payload.get(key) != value for key, value in base.items())
            or payload["configuration"] != CONFIGURATION or payload["feature_dimension"] != 116
            or payload["fixed_epochs"] != spec["fixed_length"] or payload["cosine_T_max"] != 30):
        raise RuntimeError("Neural checkpoint is not the declared new refit")
    scaler = plan["scalers"][str(entry["seed"])]
    for key, expected_key in (("feature_mean", "mean_sha256"), ("feature_std", "std_sha256")):
        values = payload[key]
        if (not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.shape != (116,)
                or not np.isfinite(values).all() or array_sha256(values) != scaler[expected_key]):
            raise RuntimeError("Checkpoint no longer contains the exact frozen float32 scaler")
    if np.any(payload["feature_std"] <= 0):
        raise ValueError("Invalid neural feature standard deviation")
    model = helper.make_model(torch, 116, CONFIGURATION).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def predict_auxiliary(folder: Path, manifest: dict[str, Any], plan: dict[str, Any], inputs: Path,
                      device: str, helper: Any, budget: Budget) -> np.ndarray:
    """Predict each saved seed with the original scenario/128-site batching."""
    import torch
    torch.set_num_threads(CPU_THREADS)
    if device.startswith("cuda:"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.cuda.is_initialized():
        raise RuntimeError("CPU inference must not initialize CUDA")
    predictions = []
    for entry in manifest["models"][:2]:
        model, payload = load_new_mlp(folder, manifest, plan, entry, device, helper)
        pieces = []
        for scenario in SCENARIOS:
            budget.check()
            state = open_array(inputs / f"{scenario}_test_y0.npy", (6171, 15, 7))
            decoded = helper.decode_initial_state(state, "community_affine", helper.PRIMITIVES)
            pieces.append(helper.infer_auxiliary(model, payload, decoded, device, budget))
            del decoded, state
        predictions.append(np.concatenate(pieces).astype(np.float32))
        del model, payload
        gc.collect()
    result = np.mean(np.stack(predictions), axis=0, dtype=np.float64).astype(np.float32)
    if device == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("CPU neural inference unexpectedly initialized CUDA")
    return result


def predict_specialists(folder: Path, manifest: dict[str, Any], plan: dict[str, Any], inputs: Path,
                        helper: Any, budget: Budget) -> np.ndarray:
    """Reproduce the thirty saved boosters in float64 without neural scalers."""
    import lightgbm as lgb
    outputs = []
    for scenario in SCENARIOS:
        budget.check()
        original = open_array(inputs / f"{scenario}_test_y0.npy", (6171, 15, 7))
        state = helper.decode_initial_state(original, "community_affine", helper.PRIMITIVES)
        annual = open_array(inputs / f"test_xann_{scenario}.npy", (6171, 40, 136))
        climate = np.empty((6171, 519), dtype=np.float32)
        for start in range(0, 6171, 256):
            budget.check()
            climate[start:start + 256] = climate_features(annual[start:start + 256])
        del annual, original
        anchors = helper.anchor_at_40(state)[:, :, :2] / helper.SCALES
        prediction = np.empty((6171, 15, 2), dtype=np.float64)
        for age_index in range(15):
            base = features_at_age(state, age_index, helper, budget)
            matrix = np.concatenate([base, climate], axis=1)
            for entry in manifest["models"][2 + age_index * 2:4 + age_index * 2]:
                budget.check()
                booster = lgb.Booster(model_file=str(contained(folder, entry["model_path"])))
                if (booster.num_feature() != 635 or booster.feature_name() != feature_names()
                        or booster.num_trees() != entry["fixed_length"]):
                    raise RuntimeError("Portable booster schema or fixed tree count changed")
                target_index = entry["target_index"]
                for start in range(0, 6171, 4096):
                    budget.check()
                    stop = min(start + 4096, 6171)
                    residual = booster.predict(matrix[start:stop], num_iteration=entry["fixed_length"], num_threads=CPU_THREADS)
                    prediction[start:stop, age_index, target_index] = np.maximum(
                        residual + anchors[start:stop, age_index, target_index], 0)
                del booster
            del base, matrix
        outputs.append(prediction.reshape(-1, 2))
        del state, climate, anchors
    return np.concatenate(outputs)


def mix_families(auxiliary: np.ndarray, specialists: np.ndarray) -> np.ndarray:
    """Apply the predeclared half-and-half mixture without extra rounding."""
    if auxiliary.dtype != np.float32 or specialists.dtype != np.float64 or auxiliary.shape != specialists.shape:
        raise ValueError("Family dtype/shape contract changed before blending")
    if not np.isfinite(auxiliary).all() or not np.isfinite(specialists).all() or min(auxiliary.min(), specialists.min()) < 0:
        raise ValueError("Family predictions must be finite and nonnegative")
    return .5 * auxiliary.astype(np.float64) + .5 * specialists


def write_csv(inputs: Path, destination: Path, values: np.ndarray, helper: Any) -> dict[str, Any]:
    """Write exactly the official ordered normalized CSV using the frozen .9g format."""
    with (inputs / "sample_submission.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "height", "agb"]:
            raise ValueError("Official template columns changed")
        identifiers = [row["id"] for row in reader]
    if identifiers != helper.expected_ids() or len(set(identifiers)) != 185130:
        raise ValueError("Official ID ordering or uniqueness changed")
    if values.shape != (185130, 2) or not np.isfinite(values).all() or values.min() < 0:
        raise ValueError("Invalid prediction shape, finiteness or nonnegative contract")
    if destination.exists():
        raise FileExistsError("Prediction output must be new")
    temporary = destination.with_suffix(".partial.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "height", "agb"])
        for identifier, row in zip(identifiers, values):
            writer.writerow([identifier, format(float(row[0]), ".9g"), format(float(row[1]), ".9g")])
    os.replace(temporary, destination)
    return {"path": destination.name, "sha256": sha256(destination), "rows": 185130,
            "columns": ["id", "height", "agb"], "finite": True, "nonnegative": True,
            "normalization": "height/10.90645, agb/3.39567"}


def export_predictions(inputs: Path, destination: Path, arrays: dict[str, np.ndarray], helper: Any,
                       expected: dict[str, str] | None = None, reference: Path | None = None) -> dict[str, Any]:
    """Export and gate predictions; mismatches retain CSVs and a difference report."""
    if destination.exists():
        raise FileExistsError("Use a fresh prediction directory")
    if expected is not None and set(expected) != set(arrays):
        raise ValueError("Expected SHA map must name exactly the exported families")
    if expected is not None:
        expected = {name: helper.normalize_sha256(value) for name, value in expected.items()}
    destination.mkdir(parents=True)
    records = {}
    passed = True
    for family, values in arrays.items():
        path = destination / f"{family}.csv"
        record = write_csv(inputs, path, values, helper)
        np.save(destination / f"{family}.npy", values, allow_pickle=False)
        if expected is not None:
            digest = helper.normalize_sha256(expected[family])
            record["expected_sha256"] = digest
            record["sha256_matches"] = record["sha256"] == digest
            passed = passed and record["sha256_matches"]
        if reference is not None:
            reference_file = reference / f"{family}.csv"
            record["reference_difference"] = helper.csv_difference(path, reference_file)
        records[family] = record
    result = {"gate": "PASS" if expected is not None and passed else "FAIL" if not passed else "NOT_REQUESTED",
              "passed": passed, "outputs": records,
              "interpretation": "Prediction-to-prediction reproduction; no target or leaderboard evaluation"}
    atomic_json(destination / "prediction_difference_report.json", result)
    return result


def predict(folder: Path, inputs: Path, destination: Path, family: str, device: str, mode: str,
            helper: Any, budget: Budget, expected: dict[str, str] | None = None,
            reference: Path | None = None, *, allow_synthetic: bool = False) -> dict[str, Any]:
    """Predict one or both families from weights and public inputs only."""
    if family not in (*FAMILIES, "all"):
        raise ValueError("Unknown prediction family")
    if family == "state_shape_climate_delta" and device != "cpu":
        raise ValueError("The specialist-only path is CPU-only")
    for protected in (folder.resolve(), inputs.resolve()):
        if destination.resolve() == protected or protected in destination.resolve().parents:
            raise ValueError("Prediction outputs must be separate from the model bundle and public inputs")
    if destination.exists():
        raise FileExistsError("Prediction output directory must be new")
    manifest, plan = read_bundle(folder, family, allow_synthetic=allow_synthetic)
    names = PUBLIC_NAMES if family != "aux_shape" else (PUBLIC_NAMES[0], PUBLIC_NAMES[1], PUBLIC_NAMES[-1])
    for name in names:
        if manifest["public_inputs"][name] != plan["sources"][name]:
            raise RuntimeError("Public-input provenance changed between plan and bundle")
    verify_data(inputs, manifest["public_inputs"], names, budget)
    arrays = {}
    with helper.execution_context(MISSION_ROOT, mode, device, budget):
        if family in ("aux_shape", "blend", "all"):
            arrays["aux_shape"] = predict_auxiliary(folder, manifest, plan, inputs, device, helper, budget)
        if family in ("state_shape_climate_delta", "blend", "all"):
            arrays["state_shape_climate_delta"] = predict_specialists(folder, manifest, plan, inputs, helper, budget)
        if family in ("blend", "all"):
            arrays["blend"] = mix_families(arrays["aux_shape"], arrays["state_shape_climate_delta"])
        selected = arrays if family == "all" else {family: arrays[family]}
        result = export_predictions(inputs, destination, selected, helper, expected, reference)
        result.update({"contract": CONTRACT, "study": STUDY, "created_utc": utc_now(),
                       "bundle_manifest_sha256": sha256(folder / "bundle_manifest.json"),
                       "refit_code_sha256": sha256(Path(__file__)), "device": device, "mode": mode,
                       "training_arrays_read": False, "reference_checkpoints_required": False,
                       "elapsed_seconds": budget.elapsed(), "memory": memory_usage(),
                       "environment": package_versions()})
        atomic_json(destination / "reproduction_receipt.json", result)
        return result


def smoke_math(helper: Any) -> dict[str, Any]:
    """Check fixed decisions and feature/mixture arithmetic without torch or LightGBM."""
    rng = np.random.default_rng(20261113)
    state = rng.uniform(.25, 4, (259, 15, 7)).astype(np.float32)
    state[:, :, :2] += np.arange(15, dtype=np.float32)[None, :, None] * .5
    state[0, 0, :2] = 0
    encoded = state.copy()
    encoded[:, :, :2] = encoded[:, :, :2] * helper.AFFINE_SLOPE + helper.AFFINE_OFFSET
    decoded = helper.decode_initial_state(encoded, "community_affine", helper.PRIMITIVES)
    np.testing.assert_array_equal(decoded[:, :, 2:], state[:, :, 2:])
    np.testing.assert_allclose(decoded[:, :, :2], state[:, :, :2], atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(encoded[:, :, 2:], state[:, :, 2:])
    features = helper.feature_matrix(state, helper.PRIMITIVES)
    assert features.shape == (259 * 15, 116) and features.dtype == np.float32
    budget = Budget(60)
    for age in (0, 7, 14):
        np.testing.assert_array_equal(features_at_age(state, age, helper, budget), features.reshape(259, 15, 116)[:, age])
    annual = (np.arange(40, dtype=np.float64)[None, :, None] * .125
              + np.arange(136, dtype=np.float64)[None, None, :] * .25)
    climate = climate_features(annual)
    assert climate.shape == (1, 519) and climate.dtype == np.float32
    np.testing.assert_array_equal(climate[:, 247:383], np.full((1, 136), .125, dtype=np.float32))
    np.testing.assert_array_equal(climate[:, 383:], np.full((1, 136), 3.75, dtype=np.float32))
    assert len(feature_names()) == len(set(feature_names())) == 635
    assert EPOCHS == {20261113: 4, 20261114: 16} and sum(sum(x) for x in ROUNDS) == 32128
    for age in range(15):
        for target in range(2):
            params = tree_parameters(age, target)
            assert params["seed"] == 20261113 + 2 * age + target
            assert "num_iterations" not in params and params["device_type"] == "cpu"
    auxiliary = np.array([[16777216, 1]], dtype=np.float32)
    specialists = np.array([[16777217, 1 + 2 ** -40]], dtype=np.float64)
    mixture = mix_families(auxiliary, specialists)
    assert mixture.dtype == np.float64 and mixture[0, 0] == 16777216.5 and mixture[0, 1] == 1 + 2 ** -41
    try:
        mix_families(auxiliary.astype(np.float64), specialists)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong family dtype was accepted")
    try:
        Budget(1e-12).check()
    except TimeoutError:
        pass
    else:
        raise AssertionError("Elapsed-time guard did not fail")
    return {"passed": True, "scope": "pure mathematical and policy checks; no training smoke",
            "checks": ["seven-state decoder round-trip", "auxiliary bytes preserved",
                       "116/635 feature dimensions and age ordering", "519 forcing summary arithmetic",
                       "4/16 epochs and 32128 fixed tree rounds", "original seed/parameter contract",
                       "float64 family mixture without rounding", "elapsed-time guard"],
            "torch_imported": "torch" in sys.modules, "lightgbm_imported": "lightgbm" in sys.modules}


def synthetic_public_inputs(folder: Path, helper: Any) -> dict[str, Any]:
    """Create explicitly synthetic, full-format public fixtures without source targets."""
    folder.mkdir()
    base = np.arange(15, dtype=np.float32)[None, :, None] * .25
    site = (np.arange(6171, dtype=np.float32) % 101)[:, None, None] * .005
    channel = np.arange(7, dtype=np.float32)[None, None, :] * .125
    state = (base + site + channel).astype(np.float32)
    state[0, 0, :2] = 0
    for scenario_index, scenario in enumerate(SCENARIOS):
        encoded = state.copy()
        encoded[:, :, :2] = encoded[:, :, :2] * helper.AFFINE_SLOPE + helper.AFFINE_OFFSET
        np.save(folder / f"{scenario}_test_y0.npy", encoded, allow_pickle=False)
        annual_frame = (np.arange(40, dtype=np.float32)[None, :, None] * .01
                        + np.arange(136, dtype=np.float32)[None, None, :] * .001
                        + scenario_index * .02)
        np.save(folder / f"test_xann_{scenario}.npy", np.broadcast_to(annual_frame, (6171, 40, 136)), allow_pickle=False)
    with (folder / "sample_submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "height", "agb"])
        writer.writerows((identifier, 0, 0) for identifier in helper.expected_ids())
    return {name: {"sha256": sha256(folder / name), "bytes": (folder / name).stat().st_size} for name in PUBLIC_NAMES}


def smoke(output: Path, manifest_path: Path, helper: Any) -> dict[str, Any]:
    """Exercise all 32 synthetic fits, fixed lengths, portable reloads and CSV gates on CPU."""
    if output.exists():
        raise FileExistsError("Smoke output directory must be new")
    load_fixed_lengths(manifest_path)
    pure = smoke_math(helper)
    if "torch" in sys.modules:
        raise RuntimeError("Run the smoke in a fresh process so CPU isolation is testable")
    output.mkdir(parents=True)
    atomic_json(output / "MATHEMATICAL_SMOKE.json", pure)
    budget = Budget(600)
    with budget.signals(), tempfile.TemporaryDirectory(prefix="full_refit_smoke_", dir=output) as temporary:
        folder = Path(temporary)
        run = folder / "synthetic_run"
        run.mkdir()
        inputs = folder / "public_inputs"
        sources = synthetic_public_inputs(inputs, helper)
        rng = np.random.default_rng(20261113)
        scalers = {}
        (run / "scalers").mkdir()
        for seed in SEEDS:
            mean = np.linspace(-.1, .1, 116, dtype=np.float32)
            std = np.linspace(.8, 1.2, 116, dtype=np.float32)
            path = run / "scalers" / f"seed{seed}.npz"
            np.savez(path, feature_mean=mean, feature_std=std)
            scalers[str(seed)] = {"path": str(path.relative_to(run)), "sha256": sha256(path),
                                  "mean_sha256": array_sha256(mean), "std_sha256": array_sha256(std)}
        plan = {"contract": CONTRACT, "study": STUDY, "synthetic_fixture": True,
                "refit_code_sha256": sha256(Path(__file__)), "reused_sources_sha256": REUSED_SOURCES,
                "training_sites": 640, "configuration": CONFIGURATION, "cosine_T_max": 30,
                "epochs": {str(k): v for k, v in EPOCHS.items()}, "tree_rounds": [list(r) for r in ROUNDS],
                "feature_names": feature_names(), "scalers": scalers, "sources": sources,
                "environment": package_versions(), "blend": {"aux_shape": .5, "state_shape_climate_delta": .5}}
        atomic_json(run / "plan.json", plan)
        shutil.copyfile(manifest_path, run / "fixed_lengths.json")
        matrix = rng.uniform(-1, 1, (640, 635)).astype(np.float32)
        base_residual = .3 * matrix[:, 0].astype(np.float64) ** 2 + .2 * np.sin(6 * matrix[:, 1])
        tree_records = []
        for spec in model_specs(plan)[2:]:
            residual = base_residual + rng.normal(0, .1, len(matrix))
            tree_records.append(save_tree_fit(run, plan, spec, matrix, residual, budget))
        if "torch" in sys.modules:
            raise AssertionError("CPU tree fitting imported torch")
        cpu_torch_free = True
        import torch
        if torch.cuda.is_initialized():
            raise AssertionError("CPU smoke unexpectedly initialized CUDA")
        neural = rng.normal(size=(256, 116)).astype(np.float32)
        anchors = rng.uniform(.5, 1, (256, 2)).astype(np.float32)
        labels = (anchors + rng.normal(0, .1, anchors.shape)).astype(np.float32)
        neural_records = []
        for spec in model_specs(plan)[:2]:
            mean, std = scaler_arrays(run, plan, spec["seed"])
            scaled = np.clip((neural - mean) / std, -8, 8)
            record = save_mlp_fit(run, plan, spec, scaled, labels, anchors, mean, std, "cpu", helper, budget)
            assert record["training"]["epochs_executed"] == EPOCHS[spec["seed"]]
            assert record["training"]["cosine_T_max"] == 30
            assert record["training"]["optimizer_steps"] == EPOCHS[spec["seed"]]
            neural_records.append(record)
        final = finalize(run, allow_synthetic=True)
        bundle = Path(final["bundle_manifest"]).parent
        assert not list(bundle.rglob("train_*.npy")) and not (bundle / "cache").exists()
        isolated = folder / "isolated_bundle"
        shutil.copytree(bundle, isolated)
        shutil.rmtree(run)
        # Import the copied entrypoint itself: no original source directory or
        # reference checkpoints exist in this isolated inference directory.
        source = isolated / "code" / Path(__file__).name
        spec = importlib.util.spec_from_file_location("isolated_full_refit_smoke", source)
        if spec is None or spec.loader is None:
            raise ImportError(str(source))
        independent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(independent)
        independent_helper = independent.load_primitives()
        first = independent.predict(isolated, inputs, folder / "first", "all", "cpu", "portable",
                                    independent_helper, budget, allow_synthetic=True)
        expected = {name: record["sha256"] for name, record in first["outputs"].items()}
        second = independent.predict(isolated, inputs, folder / "second", "all", "cpu", "portable",
                                     independent_helper, budget, expected, folder / "first", allow_synthetic=True)
        assert second["gate"] == "PASS"
        arrays = {name: np.load(folder / "first" / f"{name}.npy", allow_pickle=False) for name in FAMILIES}
        np.testing.assert_array_equal(arrays["blend"], mix_families(arrays["aux_shape"], arrays["state_shape_climate_delta"]))
        failed = export_predictions(inputs, folder / "failed_hash_gate", arrays, helper,
                                    {name: "0" * 64 for name in FAMILIES}, folder / "first")
        assert failed["gate"] == "FAIL" and all((folder / "failed_hash_gate" / f"{name}.csv").is_file() for name in FAMILIES)
        try:
            independent.read_bundle(isolated, "all")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Production reader accepted synthetic weights")
        original_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "INVALID_SYNTHETIC_UUID"
        try:
            try:
                with helper.execution_context(MISSION_ROOT, "mission", "cuda:0", Budget(5)):
                    raise AssertionError("Invalid mission GPU was accepted")
            except RuntimeError:
                pass
        finally:
            if original_visibility is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visibility
        assert not torch.cuda.is_initialized()
        result = {"passed": True, "completed_utc": utc_now(), "refit_code_sha256": sha256(Path(__file__)),
                  "primitives_sha256": PRIMITIVES_SHA, "fixed_lengths_sha256": FIXED_LENGTHS_SHA,
                  "synthetic_models": 32, "tree_iterations_executed": sum(r["executed_length"] for r in tree_records),
                  "neural_epochs_executed": [r["executed_length"] for r in neural_records],
                  "cosine_T_max": 30, "cpu_tree_phase_torch_free": cpu_torch_free,
                  "cuda_initialized": False, "full_csv_rows_per_family": 185130,
                  "portable_replay_sha256": expected, "byte_exact_cpu_replay": True,
                  "hash_mismatch_preserves_csv_and_fails": True,
                  "isolated_bundle_requires_no_training_arrays": True,
                  "synthetic_weights_rejected_by_production_reader": True,
                  "training_reference_scalers_preserved_as_float32": True,
                  "invalid_mission_uuid_rejected_before_cuda_initialization": True,
                  "elapsed_seconds": budget.elapsed(), "memory": memory_usage(),
                  "environment": package_versions(),
                  "limitation": "Synthetic CPU checks do not certify production GPU training or production CSV replay"}
        atomic_json(output / "CPU_SMOKE.json", result)
        return result


def main() -> int:
    """Expose separate reviewed phases; no command invokes another training phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare", help="Freeze provenance and original scalers without training")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--fixed-lengths", type=Path, required=True)
    for action in ("train-cpu", "train-gpu"):
        child = sub.add_parser(action)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    finalize_parser = sub.add_parser("finalize", help="Package all 32 weights without training or inference")
    finalize_parser.add_argument("--output", type=Path, required=True)
    prediction = sub.add_parser("predict", help="Portable independent saved-weight inference")
    prediction.add_argument("--bundle", type=Path, required=True)
    prediction.add_argument("--inputs", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    prediction.add_argument("--device", default="cpu")
    prediction.add_argument("--mode", choices=("portable", "mission"), default="portable")
    prediction.add_argument("--max-runtime-seconds", type=float, default=2700)
    prediction.add_argument("--expected-sha256-json", type=Path)
    prediction.add_argument("--reference-dir", type=Path)
    smoke_parser = sub.add_parser("smoke", help="Run all fixed-length synthetic CPU training/inference checks")
    smoke_parser.add_argument("--output", type=Path, required=True)
    smoke_parser.add_argument("--fixed-lengths", type=Path, required=True)
    mathematical = sub.add_parser("smoke-math", help="Run NumPy-only checks; this is not a training smoke")
    mathematical.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    helper = load_primitives()
    owned_output = None
    try:
        if args.action in ("prepare", "predict", "smoke"):
            if args.output.exists():
                raise FileExistsError("Use a fresh output directory; an existing directory was not modified")
            owned_output = args.output.resolve()
        elif args.action in ("train-cpu", "train-gpu", "finalize"):
            verify_plan(args.output.resolve())
            owned_output = args.output.resolve()
        if args.action == "smoke-math":
            result = smoke_math(helper)
            result["refit_code_sha256"] = sha256(Path(__file__))
            if args.receipt is not None:
                if args.receipt.exists():
                    raise FileExistsError("Use a new mathematical smoke receipt path")
                atomic_json(args.receipt, result)
        elif args.action == "smoke":
            result = smoke(args.output.resolve(), args.fixed_lengths.resolve(), helper)
        elif args.action == "prepare":
            budget = Budget(600, training=True)
            with budget.signals():
                result = prepare(args.root.resolve(), args.output.resolve(), args.fixed_lengths.resolve(), helper, budget)
        elif args.action == "train-cpu":
            result = train_cpu(args.root.resolve(), args.output.resolve(), helper)
        elif args.action == "train-gpu":
            result = train_gpu(args.root.resolve(), args.output.resolve(), helper)
        elif args.action == "finalize":
            result = finalize(args.output.resolve())
        else:
            expected = read_json(args.expected_sha256_json) if args.expected_sha256_json is not None else None
            budget = Budget(args.max_runtime_seconds)
            result = predict(args.bundle.resolve(), args.inputs.resolve(), args.output.resolve(),
                             args.family, args.device, args.mode, helper, budget, expected,
                             args.reference_dir.resolve() if args.reference_dir else None)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return 2 if result.get("gate") == "FAIL" else 0
    except Exception as exc:
        failure = {"failed_utc": utc_now(), "action": args.action, "error_type": type(exc).__name__,
                   "error": str(exc), "refit_code_sha256": sha256(Path(__file__)), "memory": memory_usage()}
        if owned_output is not None and owned_output.is_dir():
            atomic_json(owned_output / "last_failure.json", failure)
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
