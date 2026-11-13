#!/usr/bin/env python3
"""Fit a bounded, preregistered family of age-specific LightGBM residual models.

The two arms use the same frozen 116 state/shape inputs, with or without 519
generic annual-forcing summaries. Each arm has one booster for each of 15 ages
and two targets. Training uses geographic folds 2--4 and iteration selection uses
fold 0. The already-used fold 1 is descriptive only and is read after all 60
models are complete. This program never uploads a submission or fits a blend.

The age-specialist design is motivated by Robina Mirbahar's public EcoPhysNet
notebook (Kaggle, version 14, Apache 2.0). The feature definitions and experiment
below are independently implemented. No unverified physical channel names,
derived physical quantities, or hidden test labels are used.

Actions: smoke (synthetic CPU only), prepare (freeze provenance), run (bounded
CPU training and reporting), finalize (complete reporting), and predict
(saved-booster inference using public test inputs only).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import tempfile
import time
from typing import Any, Iterator

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "4"

import numpy as np


STUDY = "age_specialists_20261113_v1"
TRANSFER_SHA256 = "230680807f1ec9a3ccd89c8739ac149284cb5c1b5eaa5b11707fcbd213fa2905"
AUXILIARY_SHA256 = "5684f74e7254d045249b68bcec4d0fa889caf4ad50867c7b48e690f16483350c"
ARMS = ("state_shape", "state_shape_climate_delta")
DIMENSIONS = {"state_shape": 116, "state_shape_climate_delta": 635}
TARGETS = ("height", "agb")
SCENARIOS = ("ssp126", "ssp585")
AGES = (1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500)
BAD = tuple(range(8, 32)) + (135,)
GOOD = tuple(index for index in range(136) if index not in BAD)
SEED_BASE = 20261113
MAXIMUM_SECONDS = 45 * 60
MEMORY_BYTES = 6 * 1024**3
THREADS = 4
ROUNDS = 1200
PATIENCE = 80
STOP = False
LOG = logging.getLogger(STUDY)
MOTIVATION = {
    "author": "Robina Mirbahar",
    "title": "EcoPhysNet: Physics-Guided Carbon Cycle Emulator",
    "url": "https://www.kaggle.com/code/robinamirbahar/ecophysnet-physics-guided-carbon-cycle-emulator",
    "version_observed": 14,
    "license_observed": "Apache-2.0",
    "reviewed_date": "2026-11-13",
    "scope": "Motivation for age-specific predictors using the full initial age profile; no source code copied",
    "limitation": "Notebook validation numbers are not treated as verified leaderboard scores",
}


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically write a finite, human-readable provenance record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    """Read a required JSON artifact without silently recovering corruption."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_array(path: Path, array: np.ndarray) -> None:
    """Commit a NumPy array atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def utc_now() -> str:
    """Return an explicit UTC timestamp."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def environment() -> dict[str, Any]:
    """Capture the CPU numerical environment without importing torch or CUDA."""
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "lightgbm")}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "num_threads": THREADS, "device_type": "cpu"}


def enforce_memory_limit() -> dict[str, Any]:
    """Apply a Linux address-space cap that also bounds resident process memory.

    The virtual-memory cap is deliberately at least as strict as a 6 GiB RAM
    cap. The implementation opens large input maps in short-lived stages, rather
    than requiring all training tensors and feature copies simultaneously.
    """
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = MEMORY_BYTES if hard == resource.RLIM_INFINITY else min(MEMORY_BYTES, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    return {"kind": "RLIMIT_AS", "bytes": int(limit), "requested_bytes": MEMORY_BYTES,
            "interpretation": "A hard virtual-memory cap, which is stricter than the same resident-memory cap"}


def memory_usage() -> dict[str, int]:
    """Read Linux process memory counters without a third-party dependency."""
    result: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
            key, value, _unit = line.split()
            result[key.rstrip(":")] = int(value) * 1024
    return result


def request_stop(_signal: int, _frame: Any) -> None:
    """Stop at the next numerical boundary while retaining completed models."""
    global STOP
    STOP = True


class BudgetExceeded(RuntimeError):
    """Indicate an explicit time, memory, or stop boundary."""


class Budget:
    """Maintain one cumulative training/reporting time budget across clean resumes."""

    def __init__(self, receipt: Path | None = None, plan_hash: str | None = None,
                 seconds: float = MAXIMUM_SECONDS, respect_deadline: bool = True) -> None:
        self.receipt, self.plan_hash, self.maximum = receipt, plan_hash, float(seconds)
        self.respect_deadline = respect_deadline
        self.previous = 0.0
        if receipt is not None and receipt.exists():
            previous = read_json(receipt)
            if previous["plan_sha256"] != plan_hash or previous["maximum_seconds"] != seconds:
                raise RuntimeError("The stored resource budget belongs to a different plan")
            if previous["session_active"]:
                raise RuntimeError("The previous session did not close cleanly; do not silently reset its budget")
            self.previous = float(previous["consumed_seconds"])
        self.started = time.monotonic()
        self.last_heartbeat = self.started
        self.check()
        self.persist(True)

    def elapsed(self) -> float:
        """Return time already consumed plus the current session's elapsed time."""
        return self.previous + time.monotonic() - self.started

    def persist(self, active: bool) -> None:
        """Record budget consumption without discarding an interrupted session."""
        if self.receipt is not None:
            write_json(self.receipt, {"plan_sha256": self.plan_hash, "maximum_seconds": self.maximum,
                       "consumed_seconds": self.elapsed(), "session_active": active,
                       "updated_utc": utc_now(), "memory": memory_usage()})

    def check(self) -> None:
        """Enforce time, deadline, stop, and resident-memory limits."""
        if STOP:
            raise BudgetExceeded("Stop requested; the current unfinished booster is not accepted")
        if self.elapsed() >= self.maximum:
            raise BudgetExceeded("The cumulative 45-minute CPU experiment budget is exhausted")
        if (self.respect_deadline and
                dt.datetime.now(dt.timezone.utc) >= dt.datetime(2026, 11, 16, 4, 35, tzinfo=dt.timezone.utc)):
            raise BudgetExceeded("The frozen competition deadline margin has been reached")
        if memory_usage().get("VmRSS", 0) > MEMORY_BYTES:
            raise BudgetExceeded("Resident memory exceeds the 6 GiB budget")
        if time.monotonic() - self.last_heartbeat >= 30:
            self.persist(True)
            self.last_heartbeat = time.monotonic()

    def close(self) -> None:
        """Close a cleanly handled session while preserving cumulative consumption."""
        self.persist(False)


@contextlib.contextmanager
def exclusive_run(output: Path) -> Iterator[None]:
    """Prevent duplicate writers to this CPU study without taking any GPU lock."""
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "study_cpu.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This specialist study already has an active writer") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_module(path: Path, expected: str, name: str) -> Any:
    """Import a helper only after verifying its complete frozen source hash."""
    if sha256(path) != expected:
        raise RuntimeError(f"Frozen helper hash mismatch: {path}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_engines(parent_path: Path, auxiliary_path: Path) -> tuple[Any, Any]:
    """Reuse exact state features, anchor, decoder, and output formatting."""
    parent = load_module(parent_path, TRANSFER_SHA256, "specialist_frozen_transfer")
    auxiliary = load_module(auxiliary_path, AUXILIARY_SHA256, "specialist_frozen_auxiliary")
    if tuple(parent.AGES.astype(int)) != AGES or auxiliary.FEATURE_DIMENSION != 116:
        raise RuntimeError("Frozen helper dimensions or age ordering changed")
    return parent, auxiliary


def parameters(age_index: int, target_index: int) -> dict[str, Any]:
    """Return the one predeclared parameter set, with matched seeds across arms."""
    seed = SEED_BASE + 2 * age_index + target_index
    return {"objective": "regression", "metric": "None", "boosting_type": "gbdt",
            "device_type": "cpu", "num_threads": THREADS, "deterministic": True,
            "force_col_wise": True, "force_row_wise": False, "num_leaves": 31,
            "max_depth": -1, "min_data_in_leaf": 128, "min_sum_hessian_in_leaf": 1e-3,
            "learning_rate": 0.04, "feature_fraction": 0.9, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l1": 0.0, "lambda_l2": 1.0,
            "max_bin": 255, "min_data_in_bin": 3, "bin_construct_sample_cnt": 200000,
            "histogram_pool_size": 256.0, "feature_pre_filter": True,
            "use_missing": True, "zero_as_missing": False, "extra_trees": False,
            "seed": seed, "data_random_seed": seed, "feature_fraction_seed": seed,
            "bagging_seed": seed, "extra_seed": seed, "verbosity": -1}


def feature_names(arm: str) -> list[str]:
    """Name dimensions by index and operation without assigning physical semantics."""
    names = [f"frozen_state_shape_{index:03d}" for index in range(116)]
    if arm == ARMS[1]:
        names += [f"annual_mean_x{index:03d}" for index in GOOD]
        for operation in ("population_std", "ols_slope", "last10_minus_first10"):
            names += [f"annual_{operation}_x{index:03d}" for index in range(136)]
    elif arm != ARMS[0]:
        raise ValueError(f"Unknown arm: {arm}")
    assert len(names) == DIMENSIONS[arm]
    return names


def preprocessing_specification() -> dict[str, Any]:
    """Describe every transformation needed for independent saved-model inference."""
    return {"version": STUDY, "state_helper_sha256": AUXILIARY_SHA256,
            "parent_helper_sha256": TRANSFER_SHA256,
            "state_features": "Exact auxiliary.feature_matrix prefix and shape features before MLP scaling",
            "state_dimension": 116, "fitted_feature_scaler": None,
            "feature_clipping": "No additional clipping or z-scoring for LightGBM",
            "auxiliary_normalization": "Inherited frozen float64 min/range with the 8-ULP degeneracy contract",
            "climate_bad_indices": list(BAD), "climate_mean_indices": list(GOOD),
            "climate_order": ["111 ascending-index means", "136 population standard deviations",
                              "136 OLS slopes", "136 last-decade minus first-decade differences"],
            "climate_input": "Annual means already present in xann; no monthly reconstruction",
            "climate_arithmetic": "float64 reductions, one final float32 cast; reject nonfinite values or overflow",
            "climate_standard_deviation_ddof": 0,
            "climate_slope": "sum((t-19.5)*x[t])/sum((t-19.5)^2), t=0,...,39",
            "climate_decades": "mean(x[30:40]) - mean(x[0:10])",
            "bad_channel_provenance": "Frozen empirical index set; not official physical labels",
            "unknown_forcing_encoding": "No official resolution of SSP forcing normalization is assumed",
            "target": "December year 40, channels 0 and 1, divided by the frozen parent scales",
            "residual": "normalized target minus the unchanged age+40 curve anchor",
            "prediction": "max(booster residual prediction + normalized anchor, 0)",
            "decoder": "Unchanged raw/community_affine target-channel variants; auxiliary channels preserved",
            "feature_names": {arm: feature_names(arm) for arm in ARMS}}


def climate_features(annual: np.ndarray) -> np.ndarray:
    """Compute 519 generic summaries from finite site x 40 x 136 annual inputs."""
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


def state_features_at_age(state: np.ndarray, age_index: int, parent: Any, auxiliary: Any,
                          budget: Budget | None = None) -> np.ndarray:
    """Materialize only one age's 116-column matrix, using bounded site chunks."""
    if not 0 <= age_index < 15 or state.shape[1:] != (15, 7):
        raise ValueError("Invalid age index or initial-state shape")
    result = np.empty((len(state), 116), dtype=np.float32)
    for start in range(0, len(state), 256):
        if budget is not None:
            budget.check()
        values = state[start:start + 256]
        result[start:start + len(values)] = auxiliary.feature_matrix(values, parent).reshape(-1, 15, 116)[:, age_index]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite state/shape features")
    return result


def combine_features(state_features: np.ndarray, climate: np.ndarray | None, arm: str) -> np.ndarray:
    """Assemble one age's feature matrix without replicating forcing over 15 ages."""
    if state_features.ndim != 2 or state_features.shape[1] != 116:
        raise ValueError("Invalid state/shape matrix")
    if arm == ARMS[0]:
        return state_features
    if arm != ARMS[1] or climate is None or climate.shape != (len(state_features), 519):
        raise ValueError("The climate arm requires the aligned 519-column forcing matrix")
    return np.concatenate([state_features, climate], axis=1)


def verify_partition(path: Path, plan: dict[str, Any], parent: Any) -> dict[str, np.ndarray]:
    """Check exact site membership and disjoint geographic blocks."""
    with np.load(path, allow_pickle=False) as archive:
        partition = {name: archive[name] for name in ("folds", "blocks", "train", "tune", "audit")}
    if len(partition["folds"]) != 54152 or partition["blocks"].shape != (54152,):
        raise ValueError("Unexpected site population in the frozen partition")
    if parent.partition_hash(partition["folds"], partition["blocks"]) != plan["site_block_fold_sha256"]:
        raise RuntimeError("Site-to-block-to-fold manifest mismatch")
    expected = {"train": np.flatnonzero(np.isin(partition["folds"], [2, 3, 4])),
                "tune": np.flatnonzero(partition["folds"] == 0),
                "audit": np.flatnonzero(partition["folds"] == 1)}
    for name, sites in expected.items():
        if not np.array_equal(sites, partition[name]) or parent.array_hash(sites) != plan["site_hashes"][name]:
            raise RuntimeError(f"Invalid or reordered partition membership: {name}")
    if len(np.unique(np.concatenate(list(expected.values())))) != 54152:
        raise RuntimeError("The three partitions do not cover every site exactly once")
    for left, right in (("train", "tune"), ("train", "audit"), ("tune", "audit")):
        if np.intersect1d(partition["blocks"][expected[left]], partition["blocks"][expected[right]]).size:
            raise RuntimeError("Geographic blocks overlap across partitions")
    return partition


def copy_verified(source: Path, destination: Path) -> str:
    """Preserve the exact bytes of a frozen parent artifact."""
    expected = sha256(source)
    if destination.exists():
        if sha256(destination) != expected:
            raise RuntimeError(f"Existing copied artifact differs: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    if sha256(destination) != expected:
        raise RuntimeError(f"Copied artifact failed its hash check: {destination}")
    return expected


def verify_inputs(root: Path, plan: dict[str, Any], budget: Budget,
                  inference_arm: str | None = None) -> None:
    """Hash all fitting inputs, or only public test inputs for standalone prediction."""
    names = list(plan["sources"])
    if inference_arm is not None:
        names = ["ssp126_test_y0.npy", "ssp585_test_y0.npy", "sample_submission.csv"]
        if inference_arm == ARMS[1]:
            names += ["test_xann_ssp126.npy", "test_xann_ssp585.npy"]
    for name in names:
        budget.check()
        path = root / "data" / name
        expected = plan["sources"][name]
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Input differs from the frozen provenance: {name}")


def prepare(root: Path, output: Path, controls: Path, parent: Any, budget: Budget) -> dict[str, Any]:
    """Freeze the declared study using the completed parent's partition and evidence."""
    if (output / "preregistered_plan.json").exists():
        return verify_study(root, output, parent, budget)
    original = read_json(controls / "preregistered_plan.json")
    if original["code_sha256"] != TRANSFER_SHA256 or original["train_folds"] != [2, 3, 4]:
        raise RuntimeError("The parent plan is not the frozen transfer experiment")
    if original["tune_fold"] != 0 or original["audit_fold"] != 1:
        raise RuntimeError("Parent partition roles changed")
    summary = read_json(controls / "ensemble_summary.json")
    if not summary.get("completed_utc"):
        raise RuntimeError("The parent experiment must be complete")
    audit = original["target_temporal_audit"]
    if (not audit["passed"] or audit["target_year"] != 40 or audit["month_index"] != 11
            or not audit["raw_source_comparison"]
            or audit["cache_sha256"] != original["sources"]["train_ydec.npy"]["sha256"]):
        raise RuntimeError("The December year-40 cache does not have the required source audit")
    verify_inputs(root, original, budget)
    verify_partition(controls / "site_partition.npz", original, parent)
    output.mkdir(parents=True, exist_ok=True)
    copied = {}
    for source_name, target_name in (("preregistered_plan.json", "parent_plan.json"),
                                     ("site_partition.npz", "site_partition.npz"),
                                     ("climate_regimes.npz", "climate_regimes.npz"),
                                     ("ensemble_summary.json", "parent_summary.json"),
                                     ("ensemble_curve2_audit.npy", "frozen_curve2_audit.npy")):
        copied[target_name] = copy_verified(controls / source_name, output / target_name)
    write_json(output / "preprocessing.json", preprocessing_specification())
    plan = {"study": STUDY, "created_utc": utc_now(), "code_sha256": sha256(Path(__file__).resolve()),
            "transfer_sha256": TRANSFER_SHA256, "auxiliary_sha256": AUXILIARY_SHA256,
            "environment": environment(), "sources": original["sources"], "copied_artifacts": copied,
            "preprocessing_sha256": sha256(output / "preprocessing.json"),
            "site_block_fold_sha256": original["site_block_fold_sha256"],
            "site_hashes": original["site_hashes"], "site_counts": original["site_counts"],
            "target_temporal_audit": audit, "decoder": original["decoder"],
            "scales": parent.SCALES.tolist(), "arms": list(ARMS), "feature_dimensions": DIMENSIONS,
            "ages": list(AGES), "targets": list(TARGETS), "models_per_arm": 30, "model_count": 60,
            "parameters_example": parameters(0, 0), "seed_formula": "20261113 + 2*age_index + target_index",
            "num_boost_round": ROUNDS, "early_stopping_rounds": PATIENCE,
            "early_stopping_metric": "Tune-fold MSE after residual + anchor and nonnegative clamp",
            "early_stopping_min_delta": 0.0,
            "execution_order": "age ascending, then declared arm order, then height and agb",
            "train_folds": [2, 3, 4], "tune_fold": 0, "audit_fold": 1,
            "audit_policy": "Reused descriptive audit only, accessed after all 60 models are complete",
            "selection_policy": "No model selection, blending, decoder fitting, or leaderboard fitting on the audit",
            "bootstrap": {"replicates": 2000, "seed": SEED_BASE, "unit": "10-degree geographic block",
                          "weighting": "Sum squared errors and row counts within each sampled block"},
            "test_transforms": ["raw", "community_affine"], "motivation": MOTIVATION,
            "resource_limits": {"maximum_seconds": MAXIMUM_SECONDS, "memory_bytes": MEMORY_BYTES,
                                "num_threads": THREADS, "device": "cpu",
                                "time_scope": "Cumulative elapsed run/finalize time across clean resumes",
                                "incomplete_policy": "Keep completed boosters; do not audit or export an incomplete arm"}}
    write_json(output / "preregistered_plan.json", plan)
    write_json(output / "prepared_receipt.json", {"prepared_utc": utc_now(), "plan_sha256": sha256(output / "preregistered_plan.json"),
               "training_performed": False, "gpu_used": False, "audit_labels_accessed": False})
    return plan


def verify_study(root: Path, output: Path, parent: Any, budget: Budget,
                 inference_arm: str | None = None) -> dict[str, Any]:
    """Reject changed code, software, inputs, feature definitions, or partitions."""
    plan = read_json(output / "preregistered_plan.json")
    if (plan["study"] != STUDY or plan["code_sha256"] != sha256(Path(__file__).resolve())
            or plan["transfer_sha256"] != TRANSFER_SHA256 or plan["auxiliary_sha256"] != AUXILIARY_SHA256):
        raise RuntimeError("The specialist implementation differs from its frozen plan")
    if plan["environment"] != environment() or plan["arms"] != list(ARMS) or plan["feature_dimensions"] != DIMENSIONS:
        raise RuntimeError("The numerical environment or declared arms changed")
    if sha256(output / "preprocessing.json") != plan["preprocessing_sha256"]:
        raise RuntimeError("Preprocessing provenance mismatch")
    if read_json(output / "preprocessing.json") != preprocessing_specification():
        raise RuntimeError("Preprocessing semantics changed")
    verify_inputs(root, plan, budget, inference_arm)
    if inference_arm is None:
        for name, expected in plan["copied_artifacts"].items():
            if sha256(output / name) != expected:
                raise RuntimeError(f"A frozen parent artifact changed: {name}")
        verify_partition(output / "site_partition.npz", plan, parent)
    return plan


def climate_cache(root: Path, output: Path, split: str, plan: dict[str, Any], budget: Budget) -> np.ndarray:
    """Store one 519-feature row per site, with a separate integrity receipt."""
    filename = "train_xann.npy" if split == "train" else f"test_xann_{split}.npy"
    if split not in ("train", *SCENARIOS):
        raise ValueError(split)
    path = output / "features" / f"climate_{split}.npy"
    receipt_path = path.with_suffix(".json")
    expected_rows = 54152 if split == "train" else 6171
    contract = {"source_sha256": plan["sources"][filename]["sha256"],
                "preprocessing_sha256": plan["preprocessing_sha256"], "shape": [expected_rows, 519],
                "dtype": "float32"}
    if path.exists():
        receipt = read_json(receipt_path)
        if any(receipt.get(key) != value for key, value in contract.items()) or sha256(path) != receipt["sha256"]:
            raise RuntimeError(f"Climate feature cache integrity failure: {path}")
        return np.load(path, mmap_mode="r", allow_pickle=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npy")
    result = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(expected_rows, 519))
    annual = np.load(root / "data" / filename, mmap_mode="r", allow_pickle=False)
    if annual.shape != (expected_rows, 40, 136):
        raise ValueError(f"Annual input shape mismatch: {filename}")
    for start in range(0, expected_rows, 256):
        budget.check()
        result[start:start + 256] = climate_features(annual[start:start + 256])
    result.flush()
    del result, annual
    gc.collect()
    os.replace(temporary, path)
    write_json(receipt_path, {**contract, "sha256": sha256(path), "created_utc": utc_now()})
    return np.load(path, mmap_mode="r", allow_pickle=False)


def targets_for_age(root: Path, sites: np.ndarray, age_index: int, parent: Any) -> np.ndarray:
    """Read only the requested partition's two December year-40 targets."""
    source = np.load(root / "data" / "train_ydec.npy", mmap_mode="r", allow_pickle=False)
    if source.shape != (54152, 15, 41, 7):
        raise ValueError("Unexpected cached December target dimensions")
    result = np.asarray(source[sites, age_index, 40, :2], dtype=np.float32) / parent.SCALES
    del source
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite labels cannot be removed from the frozen population")
    return result


def model_paths(output: Path, arm: str, age_index: int, target_index: int) -> tuple[Path, Path]:
    """Use one stable directory and two explicit files for every specialist."""
    directory = output / "models" / arm / f"age_{AGES[age_index]:03d}"
    return directory / f"{TARGETS[target_index]}.txt", directory / f"{TARGETS[target_index]}.json"


def verify_model(output: Path, arm: str, age_index: int, target_index: int,
                 plan: dict[str, Any]) -> dict[str, Any]:
    """Verify every saved booster before resuming, reporting, or predicting."""
    model_path, record_path = model_paths(output, arm, age_index, target_index)
    record = read_json(record_path)
    expected = {"complete": True, "study": STUDY, "arm": arm, "age_index": age_index,
                "target_index": target_index, "code_sha256": plan["code_sha256"],
                "plan_sha256": sha256(output / "preregistered_plan.json"),
                "preprocessing_sha256": plan["preprocessing_sha256"],
                "parameters_requested": parameters(age_index, target_index),
                "feature_dimension": DIMENSIONS[arm], "environment": plan["environment"]}
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Booster metadata does not match the frozen experiment: {model_path}")
    if sha256(model_path) != record["model_sha256"] or not record["reload_check"]["passed"]:
        raise RuntimeError(f"Saved booster integrity failure: {model_path}")
    return record


def train_booster(x_train: np.ndarray, y_train: np.ndarray, x_tune: np.ndarray,
                  target_tune: np.ndarray, anchor_tune: np.ndarray, params: dict[str, Any],
                  names: list[str], budget: Budget, rounds: int = ROUNDS,
                  patience: int = PATIENCE) -> tuple[Any, dict[str, Any]]:
    """Fit one residual model and choose its iteration on the clamped tune metric."""
    import lightgbm as lgb
    if not all(np.isfinite(array).all() for array in (x_train, y_train, x_tune, target_tune, anchor_tune)):
        raise ValueError("A booster received nonfinite features or labels")
    training = lgb.Dataset(x_train, label=y_train, feature_name=names, free_raw_data=True)
    validation = lgb.Dataset(x_tune, label=target_tune - anchor_tune, reference=training,
                             feature_name=names, free_raw_data=True)
    history: dict[str, Any] = {}

    def evaluate(prediction: np.ndarray, _dataset: Any) -> tuple[str, float, bool]:
        scored = np.maximum(prediction + anchor_tune, 0)
        return "clamped_normalized_mse", float(np.mean(np.square(scored - target_tune))), False

    def enforce_budget(_environment: Any) -> None:
        budget.check()

    enforce_budget.order = 5
    enforce_budget.before_iteration = True
    booster = lgb.train(params, training, num_boost_round=rounds, valid_sets=[validation],
                        valid_names=["tune"], feval=evaluate,
                        callbacks=[enforce_budget, lgb.record_evaluation(history),
                                   lgb.early_stopping(patience, first_metric_only=True, verbose=False, min_delta=0.0)])
    budget.check()
    return booster, history


def fit_one(output: Path, arm: str, age_index: int, target_index: int, matrix: np.ndarray,
            training_target: np.ndarray, tuning_target: np.ndarray, anchors: np.ndarray,
            partition: dict[str, np.ndarray], plan: dict[str, Any], budget: Budget) -> dict[str, Any]:
    """Save a complete booster, verify reload inference, and commit its receipt."""
    import lightgbm as lgb
    model_path, record_path = model_paths(output, arm, age_index, target_index)
    if record_path.exists():
        return verify_model(output, arm, age_index, target_index, plan)
    if model_path.exists():
        raise RuntimeError(f"A model without its completion receipt must not be silently overwritten: {model_path}")
    train_sites, tune_sites = partition["train"], partition["tune"]
    x_train = np.ascontiguousarray(matrix[train_sites], dtype=np.float32)
    x_tune = np.ascontiguousarray(matrix[tune_sites], dtype=np.float32)
    anchor_tune = anchors[tune_sites, target_index].astype(np.float64)
    target_tune = tuning_target[:, target_index].astype(np.float64)
    residual = training_target[:, target_index].astype(np.float64) - anchors[train_sites, target_index]
    params = parameters(age_index, target_index)
    started = time.monotonic()
    booster, history = train_booster(x_train, residual, x_tune, target_tune, anchor_tune,
                                     params, feature_names(arm), budget)
    best = int(booster.best_iteration or booster.current_iteration())
    probe = booster.predict(x_tune[:512], num_iteration=best, num_threads=THREADS)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".partial.txt")
    booster.save_model(str(temporary), num_iteration=best)
    restored = lgb.Booster(model_file=str(temporary))
    if restored.num_feature() != DIMENSIONS[arm] or restored.feature_name() != feature_names(arm):
        raise RuntimeError("Restored booster feature schema differs")
    reloaded = restored.predict(x_tune[:512], num_iteration=best, num_threads=THREADS)
    if not np.allclose(probe, reloaded, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Booster reload inference exceeded the fixed numerical tolerance")
    os.replace(temporary, model_path)
    record = {"complete": True, "study": STUDY, "arm": arm, "age_index": age_index,
              "age": AGES[age_index], "target_index": target_index, "target": TARGETS[target_index],
              "seed": params["seed"], "code_sha256": plan["code_sha256"],
              "plan_sha256": sha256(output / "preregistered_plan.json"),
              "preprocessing_sha256": plan["preprocessing_sha256"],
              "parameters_requested": params, "parameters_effective": booster.params,
              "feature_dimension": DIMENSIONS[arm], "environment": plan["environment"],
              "best_iteration": best, "saved_trees": restored.num_trees(),
              "best_tune_mse": float(booster.best_score["tune"]["clamped_normalized_mse"]),
              "history": history, "model_sha256": sha256(model_path),
              "training_rows": len(train_sites), "tuning_rows": len(tune_sites),
              "audit_labels_accessed": False, "fit_seconds": time.monotonic() - started,
              "completed_utc": utc_now(), "memory": memory_usage(),
              "reload_check": {"passed": True, "atol": 1e-12, "rtol": 1e-12,
                               "maximum_absolute_difference": float(np.max(np.abs(probe - reloaded)))},
              "feature_importance_gain": restored.feature_importance(importance_type="gain").tolist()}
    write_json(record_path, record)
    LOG.info("arm=%s age=%d target=%s best=%d tune=%.8f seconds=%.1f", arm, AGES[age_index],
             TARGETS[target_index], best, record["best_tune_mse"], record["fit_seconds"])
    del booster, restored, x_train, x_tune
    gc.collect()
    return record


def train_all(root: Path, output: Path, plan: dict[str, Any], parent: Any,
              auxiliary: Any, budget: Budget) -> None:
    """Fit the 60 declared specialists sequentially, materializing one age at a time."""
    partition = verify_partition(output / "site_partition.npz", plan, parent)
    state = np.load(root / "data" / "train_y0.npy", mmap_mode="r", allow_pickle=False)
    if state.shape != (54152, 15, 7):
        raise ValueError("Unexpected training initial-state dimensions")
    climate = climate_cache(root, output, "train", plan, budget)
    for age_index in range(15):
        budget.check()
        base = state_features_at_age(state, age_index, parent, auxiliary, budget)
        anchors = np.empty((54152, 2), dtype=np.float32)
        for start in range(0, len(state), 256):
            budget.check()
            anchors[start:start + 256] = parent.anchor_at_40(state[start:start + 256])[:, age_index, :2] / parent.SCALES
        training = targets_for_age(root, partition["train"], age_index, parent)
        tuning = targets_for_age(root, partition["tune"], age_index, parent)
        for arm in ARMS:
            matrix = combine_features(base, climate if arm == ARMS[1] else None, arm)
            for target_index in range(2):
                budget.check()
                fit_one(output, arm, age_index, target_index, matrix, training, tuning,
                        anchors, partition, plan, budget)
            del matrix
            gc.collect()
        del base, anchors, training, tuning
    del state, climate
    gc.collect()


def predict_state(output: Path, arm: str, state: np.ndarray, climate: np.ndarray | None,
                  plan: dict[str, Any], parent: Any, auxiliary: Any, budget: Budget) -> np.ndarray:
    """Infer all ages from saved boosters, retaining float64 prediction arithmetic."""
    import lightgbm as lgb
    prediction = np.empty((len(state), 15, 2), dtype=np.float64)
    anchors = parent.anchor_at_40(state)[:, :, :2] / parent.SCALES
    for age_index in range(15):
        budget.check()
        base = state_features_at_age(state, age_index, parent, auxiliary, budget)
        matrix = combine_features(base, climate, arm)
        for target_index in range(2):
            record = verify_model(output, arm, age_index, target_index, plan)
            model_path, _ = model_paths(output, arm, age_index, target_index)
            booster = lgb.Booster(model_file=str(model_path))
            if booster.num_feature() != DIMENSIONS[arm] or booster.feature_name() != feature_names(arm):
                raise RuntimeError("Inference booster schema mismatch")
            for start in range(0, len(state), 4096):
                budget.check()
                stop = min(start + 4096, len(state))
                residual = booster.predict(matrix[start:stop], num_iteration=record["best_iteration"], num_threads=THREADS)
                prediction[start:stop, age_index, target_index] = np.maximum(
                    residual + anchors[start:stop, age_index, target_index], 0)
            del booster
        del base, matrix
        gc.collect()
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite specialist predictions")
    return prediction.reshape(-1, 2)


def test_prediction(root: Path, output: Path, arm: str, transform: str, plan: dict[str, Any],
                    parent: Any, auxiliary: Any, budget: Budget, use_cache: bool) -> np.ndarray:
    """Use only public inputs and saved boosters; optionally cache generic summaries."""
    if transform not in ("raw", "community_affine"):
        raise ValueError(transform)
    predictions = []
    for scenario in SCENARIOS:
        budget.check()
        original = np.load(root / "data" / f"{scenario}_test_y0.npy", mmap_mode="r", allow_pickle=False)
        if original.shape != (6171, 15, 7):
            raise ValueError("Unexpected test initial-state dimensions")
        state = auxiliary.decode_initial_state(original, transform, parent)
        climate = None
        if arm == ARMS[1]:
            if use_cache:
                climate = climate_cache(root, output, scenario, plan, budget)
            else:
                annual = np.load(root / "data" / f"test_xann_{scenario}.npy", mmap_mode="r", allow_pickle=False)
                if annual.shape != (6171, 40, 136):
                    raise ValueError("Unexpected test annual-forcing dimensions")
                climate = np.empty((6171, 519), dtype=np.float32)
                for start in range(0, 6171, 256):
                    budget.check()
                    climate[start:start + 256] = climate_features(annual[start:start + 256])
                del annual
        predictions.append(predict_state(output, arm, state, climate, plan, parent, auxiliary, budget))
        del state, original, climate
        gc.collect()
    return np.concatenate(predictions)


def verify_complete_models(output: Path, plan: dict[str, Any], budget: Budget) -> list[dict[str, Any]]:
    """Require all 60 receipts before accessing descriptive audit labels."""
    records = []
    for age_index in range(15):
        for arm in ARMS:
            for target_index in range(2):
                budget.check()
                records.append(verify_model(output, arm, age_index, target_index, plan))
    return records


def finalize(root: Path, output: Path, plan: dict[str, Any], parent: Any,
             auxiliary: Any, budget: Budget) -> dict[str, Any]:
    """Report every declared model and variant without fitting or selecting blends."""
    records = verify_complete_models(output, plan, budget)
    report_path = output / "descriptive_report.json"
    if report_path.exists():
        report = read_json(report_path)
        for name, expected in report["artifacts_sha256"].items():
            if sha256(output / name) != expected:
                raise RuntimeError(f"Completed report artifact changed: {name}")
        return report
    partition = verify_partition(output / "site_partition.npz", plan, parent)
    sites = partition["audit"]
    write_json(output / "audit_access_receipt.json", {"created_utc": utc_now(), "models_complete": len(records),
               "site_hash": parent.array_hash(sites), "role": "Reused descriptive audit; no selection"})
    source = np.load(root / "data" / "train_y0.npy", mmap_mode="r", allow_pickle=False)
    state = np.asarray(source[sites])
    del source
    labels = np.empty((len(sites), 15, 2), dtype=np.float32)
    for age_index in range(15):
        budget.check()
        labels[:, age_index] = targets_for_age(root, sites, age_index, parent)
    labels = labels.reshape(-1, 2)
    with np.load(output / "climate_regimes.npz", allow_pickle=False) as archive:
        regimes = archive["regime"][sites]
    climate_all = climate_cache(root, output, "train", plan, budget)
    climate = np.asarray(climate_all[sites])
    del climate_all
    anchor = np.maximum(parent.anchor_at_40(state)[:, :, :2] / parent.SCALES, 0).reshape(-1, 2)
    control = np.load(output / "frozen_curve2_audit.npy", allow_pickle=False)
    if control.shape != labels.shape or not np.isfinite(control).all():
        raise ValueError("Frozen curve2 control predictions do not match the audit population")
    report: dict[str, Any] = {
        "study": STUDY, "created_utc": utc_now(), "plan_sha256": sha256(output / "preregistered_plan.json"),
        "status": "COMPLETE", "models_complete": len(records), "gpu_used": False,
        "audit_role": "Reused descriptive audit only; not an independent estimate of SSP generalization",
        "selection_performed": False, "blending_performed": False, "kaggle_upload_performed": False,
        "anchor_metric": parent.metric(anchor, labels), "frozen_curve2_metric": parent.metric(control, labels),
        "model_summaries": [{key: record[key] for key in ("arm", "age", "target", "seed", "best_iteration", "best_tune_mse", "fit_seconds")}
                            for record in records], "arms": {}, "artifacts_sha256": {},
    }
    predictions = {}
    for arm in ARMS:
        budget.check()
        values = predict_state(output, arm, state, climate if arm == ARMS[1] else None,
                               plan, parent, auxiliary, budget)
        predictions[arm] = values
        destination = output / f"audit_{arm}.npy"
        save_array(destination, values)
        report["artifacts_sha256"][destination.name] = sha256(destination)
        comparison = parent.paired_bootstrap(values, control, labels, partition["blocks"][sites], plan)
        comparison["comparison"] = f"{arm} minus frozen curve2; negative favors the specialist arm"
        comparison["interpretation"] = "Descriptive reused-fold comparison; no model selection or hidden-target claim"
        report["arms"][arm] = {"audit_metric": parent.metric(values, labels),
                               "audit_strata": parent.stratified_metrics(values, labels, regimes),
                               "paired_bootstrap_vs_curve2": comparison, "test_candidates": {}}
    comparison = parent.paired_bootstrap(predictions[ARMS[1]], predictions[ARMS[0]], labels,
                                        partition["blocks"][sites], plan)
    comparison["comparison"] = "state_shape_climate_delta minus state_shape; negative favors added forcing summaries"
    comparison["interpretation"] = "Descriptive reused-fold comparison; no automatic arm selection"
    report["paired_arm_bootstrap"] = comparison
    del state, climate, predictions, labels, anchor, control
    gc.collect()
    for arm in ARMS:
        for transform in plan["test_transforms"]:
            budget.check()
            values = test_prediction(root, output, arm, transform, plan, parent, auxiliary, budget, use_cache=True)
            array_path = output / f"test_{arm}_{transform}.npy"
            save_array(array_path, values)
            csv_path = output / f"submission_{arm}_{transform}.csv"
            report["arms"][arm]["test_candidates"][transform] = parent.write_submission(root, csv_path, values)
            report["artifacts_sha256"][array_path.name] = sha256(array_path)
            report["artifacts_sha256"][csv_path.name] = sha256(csv_path)
            del values
    report["completed_utc"] = utc_now()
    report["budget_consumed_seconds"] = budget.elapsed()
    report["memory"] = memory_usage()
    write_json(report_path, report)
    return report


def smoke(parent: Any, auxiliary: Any) -> dict[str, Any]:
    """Exercise real preprocessing, tiny CPU fits, stopping, and model reload."""
    import lightgbm as lgb
    budget = Budget(seconds=120)
    rng = np.random.default_rng(73)
    times = np.arange(40, dtype=np.float64)
    slopes = np.linspace(-0.2, 0.3, 136)
    annual = 4.0 + times[None, :, None] * slopes[None, None, :]
    annual = np.repeat(annual, 3, axis=0)
    climate = climate_features(annual)
    shifted = climate_features(annual + 100.0)
    assert climate.shape == (3, 519) and len(GOOD) == 111
    assert np.allclose(climate[:, :111], (4 + 19.5 * slopes)[list(GOOD)], atol=1e-6, rtol=1e-6)
    assert np.allclose(climate[:, 111:247], np.std(times) * np.abs(slopes), atol=1e-6, rtol=1e-6)
    assert np.allclose(climate[:, 247:383], slopes, atol=1e-7, rtol=1e-6)
    assert np.allclose(climate[:, 383:], 30 * slopes, atol=1e-6, rtol=1e-6)
    assert np.allclose(climate[:, 111:], shifted[:, 111:], atol=1e-6, rtol=1e-6)
    rejected = 0
    for invalid in (np.nan, np.inf, -np.inf):
        malformed = annual.copy()
        malformed[0, 0, 0] = invalid
        try:
            climate_features(malformed)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Nonfinite forcing was accepted")
    try:
        climate_features(np.zeros((1, 39, 136)))
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("Wrong forcing shape was accepted")
    state = rng.uniform(0, 20, size=(640, 15, 7)).astype(np.float32)
    features = state_features_at_age(state, 4, parent, auxiliary, budget)
    expected = auxiliary.feature_matrix(state[:3], parent).reshape(3, 15, 116)[:, 4]
    assert np.array_equal(features[:3], expected)
    targets = 0.1 * features[:, 0] + 0.2 * features[:, 5]
    anchor = np.full(len(state), 0.3, dtype=np.float64)
    extra = rng.normal(size=(len(state), 519)).astype(np.float32)
    reload_maximum = 0.0
    with tempfile.TemporaryDirectory(prefix="age-specialist-smoke-") as temporary:
        for arm in ARMS:
            matrix = combine_features(features, extra if arm == ARMS[1] else None, arm)
            assert matrix.shape == (640, DIMENSIONS[arm])
            booster, history = train_booster(matrix[:480], targets[:480] - anchor[:480], matrix[480:],
                                             targets[480:], anchor[480:], parameters(4, 0), feature_names(arm),
                                             budget, rounds=12, patience=4)
            assert "clamped_normalized_mse" in history["tune"]
            path = Path(temporary) / f"{arm}.txt"
            booster.save_model(str(path), num_iteration=booster.best_iteration)
            restored = lgb.Booster(model_file=str(path))
            first = booster.predict(matrix[480:], num_iteration=booster.best_iteration, num_threads=THREADS)
            second = restored.predict(matrix[480:], num_iteration=booster.best_iteration, num_threads=THREADS)
            error = float(np.max(np.abs(first - second)))
            reload_maximum = max(reload_maximum, error)
            assert np.allclose(first, second, atol=1e-12, rtol=1e-12)
            assert restored.feature_name() == feature_names(arm)
            assert np.all(np.maximum(second + anchor[480:], 0) >= 0)
        output = Path(temporary) / "budget.json"
        restricted = Budget(output, "synthetic", seconds=10)
        restricted.previous = 11
        try:
            restricted.check()
        except BudgetExceeded:
            pass
        else:
            raise AssertionError("The declared time budget was not enforced")
        restricted.close()
    budget.close()
    return {"passed": True, "study": STUDY, "device": "cpu", "gpu_used": False,
            "real_training_inputs_accessed": False, "plan_prepared": False,
            "synthetic_fits": 2, "maximum_rounds_per_synthetic_fit": 12,
            "invalid_climate_cases_rejected": rejected, "reload_maximum_absolute_difference": reload_maximum,
            "source_sha256": sha256(Path(__file__).resolve()), "environment": environment(),
            "seconds": budget.elapsed(), "memory": memory_usage(),
            "checks": ["519-feature dimensions and order", "population standard deviation",
                       "OLS slope and decade difference", "offset-invariant variation summaries",
                       "nonfinite and wrong-shape rejection", "exact frozen 116-feature reuse",
                       "both real LightGBM CPU paths", "clamped tune metric", "checkpoint reload",
                       "feature-name persistence", "cumulative budget enforcement"]}


def main() -> int:
    """Run only the explicitly requested stage with fixed resource boundaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "prepare", "run", "finalize", "predict"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--parent-script", type=Path, default=Path(__file__).with_name("transfer_experiment_20261113.py"))
    parser.add_argument("--auxiliary-script", type=Path, default=Path(__file__).with_name("aux_shape_experiment_20261113.py"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--test-transform", choices=("raw", "community_affine"), default="community_affine")
    parser.add_argument("--prediction-file", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    budget: Budget | None = None
    try:
        memory_limit = enforce_memory_limit()
        parent, auxiliary = load_engines(args.parent_script.resolve(), args.auxiliary_script.resolve())
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        if args.action == "smoke":
            result = smoke(parent, auxiliary)
            result["memory_limit"] = memory_limit
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        if args.root is None:
            parser.error("--root is required outside the synthetic smoke test")
        root = args.root.resolve()
        output = (args.output or root / "research" / STUDY / "run").resolve()
        if output == root / "data" or (root / "data") in output.parents:
            raise ValueError("Study outputs must be outside the source data directory")
        if args.action == "predict":
            if args.arm is None or args.prediction_file is None:
                parser.error("predict requires --arm and --prediction-file")
            destination = args.prediction_file.resolve()
            if destination.exists() or (root / "data") in destination.parents:
                raise ValueError("Use a new prediction path outside the input data directory")
            # Saved-weight reproduction remains usable after the competition.
            # The CPU time and memory limits still apply to this public-input path.
            budget = Budget(seconds=MAXIMUM_SECONDS, respect_deadline=False)
            plan = verify_study(root, output, parent, budget, inference_arm=args.arm)
            values = test_prediction(root, output, args.arm, args.test_transform, plan,
                                     parent, auxiliary, budget, use_cache=False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            report = parent.write_submission(root, destination, values)
            report.update({"study": STUDY, "arm": args.arm, "transform": args.test_transform,
                           "saved_weights_inference": True, "training_performed": False,
                           "training_data_accessed": False, "gpu_used": False,
                           "competition_deadline_applied": False,
                           "code_sha256": plan["code_sha256"], "environment": environment()})
            write_json(destination.with_suffix(".reproduction.json"), report)
            print(json.dumps(report, indent=2))
            return 0
        with exclusive_run(output):
            if args.action == "prepare":
                if args.controls is None:
                    parser.error("prepare requires --controls pointing to the completed parent study")
                budget = Budget(seconds=MAXIMUM_SECONDS)
                result = prepare(root, output, args.controls.resolve(), parent, budget)
            else:
                plan_hash = sha256(output / "preregistered_plan.json")
                budget = Budget(output / "resource_budget.json", plan_hash)
                plan = verify_study(root, output, parent, budget)
                if args.action == "run":
                    train_all(root, output, plan, parent, auxiliary, budget)
                result = finalize(root, output, plan, parent, auxiliary, budget)
            print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (BudgetExceeded, MemoryError) as error:
        result = {"status": "PARTIAL", "reason": str(error), "completed_models_preserved": True,
                  "incomplete_models_not_accepted": True, "gpu_used": False, "kaggle_upload_performed": False}
        if args.output is not None and args.output.exists():
            write_json(args.output / "last_resource_stop.json", result)
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 3
    except Exception as error:
        print(json.dumps({"status": "ERROR", "error_type": type(error).__name__, "reason": str(error),
                          "gpu_used": False, "kaggle_upload_performed": False}, indent=2), file=sys.stderr)
        return 1
    finally:
        if budget is not None:
            budget.close()


if __name__ == "__main__":
    raise SystemExit(main())
