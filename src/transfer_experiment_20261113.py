#!/usr/bin/env python3
"""Run a frozen CarbonGlobe transfer experiment without changing historical jobs.

The two predeclared arms use an age-curve residual model: ``curve2`` has only
height/AGB inputs, while ``full`` retains the historical 136-driver summaries
and seven ecosystem states. Both use identical architecture and training rules.
Geographic folds 2--4 train, fold 0 selects the epoch, and fold 1 is read only
after all four training runs finish. Fold 1 is newly reserved for this experiment,
not claimed independent of every model or investigation in the project's history.

Two inference encodings are always exported, without choosing one on the audit:
raw inputs, and a frozen community-published affine decoder for height/AGB.
The latter is an empirical community hypothesis, NOT organizer-certified truth.
The remaining five initial-state channels are never silently recalibrated.

Commands:
  python transfer_experiment_20261113.py smoke
  python transfer_experiment_20261113.py prepare --root /path/to/ieee-carbon \
    --target-audit /path/to/verified_target_month_audit.json
  CUDA_VISIBLE_DEVICES=GPU-d60d12af-82b8-9913-7a5e-1177e774dafa \
    python transfer_experiment_20261113.py run --root /path/to/ieee-carbon
  python transfer_experiment_20261113.py predict --root /path/to/ieee-carbon \
    --checkpoints /path/to/checkpoint.pt --test-transform community_affine

``prepare`` and ``smoke`` never use a GPU. ``run`` and ``train`` refuse to share
the existing IEEE GPU lock. Prediction defaults to CPU and never retrains.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gc
import hashlib
import json
import logging
import os
import platform
import random
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "4"

import numpy as np


EXPERIMENT = "transfer_20261113_v1"
GPU_UUID = "GPU-d60d12af-82b8-9913-7a5e-1177e774dafa"
AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500], dtype=np.float32)
SCALES = np.array([10.90645, 3.39567], dtype=np.float32)
AFFINE_SLOPE = np.array([0.876341, 0.876], dtype=np.float32)
AFFINE_OFFSET = np.array([-0.900947, -0.401412], dtype=np.float32)
ARMS = ("curve2", "full")
SEEDS = (20261113, 20261114)
TRANSFORMS = ("raw", "community_affine")
SPLIT_SEED = 20261112
STOP = False
STARTED_MONOTONIC = time.monotonic()
OFFICIAL_CUTOFF = dt.datetime(2026, 11, 16, 4, 35, tzinfo=dt.timezone.utc)
LOG = logging.getLogger(EXPERIMENT)


def utc_now() -> str:
    """Return an unambiguous UTC timestamp for immutable evidence records."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Hash an artifact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    """Fingerprint an ordered array, including its shape and dtype."""
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((values.shape, values.dtype.str)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Commit one complete evidence record atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    """Load a required JSON object and fail clearly if it is malformed."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def clean_input(values: np.ndarray) -> np.ndarray:
    """Preserve historical handling of nonfinite driver/sentinel values."""
    return np.nan_to_num(np.clip(np.asarray(values, dtype=np.float32), -1e6, 1e6),
                         nan=0.0, posinf=1e6, neginf=-1e6)


def transform_state(state: np.ndarray, transform: str) -> np.ndarray:
    """Decode only height/AGB, or encode physical values for a known-label stress test."""
    output = clean_input(state).copy()
    if output.ndim != 3 or output.shape[1:] != (15, 7):
        raise ValueError(f"Expected site x 15 x 7 initial state, got {output.shape}")
    if transform == "community_affine":
        output[:, :, :2] = (output[:, :, :2] - AFFINE_OFFSET) / AFFINE_SLOPE
    elif transform == "encode_stress":
        output[:, :, :2] = output[:, :, :2] * AFFINE_SLOPE + AFFINE_OFFSET
    elif transform != "raw":
        raise ValueError(f"Unknown initial-state transform: {transform}")
    return output


def verify_test_encoding(state: np.ndarray) -> dict[str, Any]:
    """Fail closed if the two observed affine zero offsets no longer match test inputs."""
    minimum = np.asarray(state[:, :, :2]).min(axis=(0, 1))
    if not np.allclose(minimum, AFFINE_OFFSET, rtol=0, atol=2e-5):
        raise ValueError(f"Test encoding offset mismatch: minima={minimum.tolist()}; decoder not applied")
    return {"observed_minima": minimum.tolist(), "expected_offsets": AFFINE_OFFSET.tolist(),
            "tolerance": 2e-5, "passed": True,
            "limitation": "Offsets match; this does not establish the empirical slopes as ground truth"}


def check_runtime() -> None:
    """Enforce the bounded two-hour mission and the verified contest safety margin."""
    if STOP:
        raise InterruptedError("Stop requested; best checkpoints remain preserved")
    if time.monotonic() - STARTED_MONOTONIC > 7200:
        raise TimeoutError("The two-hour experimental runtime limit was reached")
    if dt.datetime.now(dt.timezone.utc) >= OFFICIAL_CUTOFF:
        raise TimeoutError("The official submission cutoff safety margin was reached")


def anchor_at_40(state: np.ndarray) -> np.ndarray:
    """Interpolate the supplied age curve at seed age + 40; cap beyond age 500."""
    logarithms = np.log(AGES)
    upper = np.clip(np.searchsorted(logarithms, np.log(AGES + 40)), 1, 14)
    lower = upper - 1
    weight = np.clip((np.log(AGES + 40) - logarithms[lower]) /
                     (logarithms[upper] - logarithms[lower]), 0, 1)
    return ((1 - weight)[None, :, None] * state[:, lower] +
            weight[None, :, None] * state[:, upper]).astype(np.float32)


def climate_summary(annual: np.ndarray) -> np.ndarray:
    """Reproduce the historical eight summaries of 136 annual forcing channels."""
    annual = clean_input(annual)
    if annual.ndim != 3 or annual.shape[1:] != (40, 136):
        raise ValueError(f"Unexpected forcing shape: {annual.shape}")
    centered_time = np.arange(40, dtype=np.float32) - 19.5
    first, last = annual[:, :10].mean(1), annual[:, 30:].mean(1)
    return clean_input(np.concatenate([
        annual.mean(1), first, last, annual.std(1), annual.min(1), annual.max(1),
        (annual * centered_time[None, :, None]).sum(1) / np.sum(centered_time ** 2),
        last - first,
    ], axis=1))


def features(annual: np.ndarray | None, state: np.ndarray, arm: str) -> np.ndarray:
    """Create inference-identical features; curve2 retains age identifiability."""
    state = clean_input(state)
    n_sites = len(state)
    anchor = anchor_at_40(state)
    age = np.broadcast_to(np.stack([np.log(AGES), np.log(AGES + 40)], axis=1)[None],
                          (n_sites, 15, 2))
    if arm == "curve2":
        curve = np.broadcast_to(state[:, :, :2].reshape(n_sites, 1, 30), (n_sites, 15, 30))
        blocks = [curve, state[:, :, :2], anchor[:, :, :2], age]
    elif arm == "full":
        if annual is None:
            raise ValueError("The full arm requires forcing data")
        climate = climate_summary(annual)
        curve = np.broadcast_to(state.reshape(n_sites, 1, 105), (n_sites, 15, 105))
        blocks = [np.broadcast_to(climate[:, None], (n_sites, 15, 1088)),
                  curve, state, anchor, age]
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return clean_input(np.concatenate(blocks, axis=-1)).reshape(n_sites * 15, -1)


def geographic_partition(mask: np.ndarray, n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    """Create balanced 10-degree block folds identical to the historical partition."""
    row, column = np.nonzero(mask.astype(bool))
    if len(row) != n_sites:
        raise ValueError("The mask cannot establish the expected number of training sites")
    blocks = (row // 20) * 100 + column // 20
    unique, counts = np.unique(blocks, return_counts=True)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(unique))
    totals = np.zeros(5, dtype=np.int64)
    mapping: dict[int, int] = {}
    for position in sorted(order, key=lambda index: -int(counts[index])):
        selected = int(np.argmin(totals))
        mapping[int(unique[position])] = selected
        totals[selected] += counts[position]
    folds = np.fromiter((mapping[int(block)] for block in blocks), dtype=np.int8, count=n_sites)
    return folds, blocks.astype(np.int32)


def source_paths(root: Path) -> dict[str, Path]:
    """List every data dependency, including the exact submission ID template."""
    names = ("train_y0.npy", "train_ydec.npy", "train_xann.npy", "mask2d.npy",
             "ssp126_test_y0.npy", "ssp585_test_y0.npy", "test_xann_ssp126.npy",
             "test_xann_ssp585.npy", "sample_submission.csv")
    paths = {name: root / "data" / name for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    return paths


def environment_fingerprint() -> dict[str, Any]:
    """Record software versions needed to verify an exact training resume."""
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "platform": platform.system(), "cpu_threads": 4}


def partition_hash(folds: np.ndarray, blocks: np.ndarray) -> str:
    """Hash the complete ordered site-to-block-to-fold mapping."""
    return array_hash(np.stack([np.arange(len(folds)), blocks, folds], axis=1).astype(np.int64))


def load_sources(root: Path) -> dict[str, np.ndarray]:
    """Open training tensors read-only and validate the target indexing contract."""
    arrays = {name: np.load(root / "data" / f"{name}.npy", mmap_mode="r")
              for name in ("train_y0", "train_ydec", "train_xann", "mask2d")}
    count = len(arrays["train_y0"])
    expected = {"train_y0": (count, 15, 7), "train_ydec": (count, 15, 41, 7),
                "train_xann": (count, 40, 136)}
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {arrays[name].shape}")
    return arrays


def prepare(root: Path, output: Path, decoder_source: str, target_audit_path: Path) -> dict[str, Any]:
    """Freeze the plan and source hashes before any model or audit evaluation."""
    output.mkdir(parents=True, exist_ok=True)
    paths = source_paths(root)
    sources = load_sources(root)
    target_audit = read_json(target_audit_path)
    required_audit = {"passed": True, "target_year": 40, "month_index": 11, "raw_source_comparison": True}
    if any(target_audit.get(key) != value for key, value in required_audit.items()):
        raise RuntimeError("A passing raw-source December year-40 audit is required before preregistration")
    if not target_audit.get("raw_source_files") or not target_audit.get("sampling"):
        raise RuntimeError("The target audit must identify raw source files and the comparison population")
    difference = float(target_audit.get("maximum_absolute_difference", float("inf")))
    tolerance = float(target_audit.get("tolerance", -1))
    if not np.isfinite(difference) or tolerance < 0 or difference > tolerance:
        raise RuntimeError("Cached targets do not pass the frozen raw-source temporal audit")
    cache_hash = sha256_file(paths["train_ydec.npy"])
    if target_audit.get("cache_sha256") != cache_hash:
        raise RuntimeError("The temporal audit does not certify the current cached target bytes")
    n_sites = len(sources["train_y0"])
    folds, blocks = geographic_partition(sources["mask2d"], n_sites)
    site_groups = {"train": np.flatnonzero(folds >= 2), "tune": np.flatnonzero(folds == 0),
                   "audit": np.flatnonzero(folds == 1)}
    for left, right in (("train", "tune"), ("train", "audit"), ("tune", "audit")):
        if np.intersect1d(blocks[site_groups[left]], blocks[site_groups[right]]).size:
            raise AssertionError("Geographic block leakage in the frozen partition")
    plan = {
        "experiment": EXPERIMENT,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "environment": environment_fingerprint(),
        "target_temporal_audit": target_audit,
        "target_temporal_audit_sha256": sha256_file(target_audit_path),
        "sources": {name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for name, path in paths.items()},
        "arms": list(ARMS), "seeds": list(SEEDS), "width": 512, "depth": 3,
        "dropout": 0.20, "learning_rate": 3e-4, "weight_decay": 0.04,
        "epochs": 30, "patience": 5, "batch_size": 2048,
        "feature_dimensions": {"curve2": 36, "full": 1209},
        "feature_schema": {"version": EXPERIMENT, "interpolation": "linear in log(age), clipped at 500 years",
                           "cleaning": "nan_to_num and sentinel clip +/-1e6", "network_activation": "GELU",
                           "normalization_layer": "LayerNorm", "dropout": 0.20, "output_clamp": "max(prediction,0)"},
        "feature_scaling": "training rows only; fixed sample; z-score clipped to [-8, 8]",
        "split_seed": SPLIT_SEED, "train_folds": [2, 3, 4], "tune_fold": 0, "audit_fold": 1,
        "site_hashes": {name: array_hash(ids) for name, ids in site_groups.items()},
        "site_counts": {name: len(ids) for name, ids in site_groups.items()},
        "site_block_fold_sha256": partition_hash(folds, blocks),
        "audit_policy": "Read after all training runs complete; no hyperparameter or decoder selection on audit",
        "audit_independence": "Reserved for this experiment, not independent of all historical project activity",
        "test_transforms": list(TRANSFORMS),
        "decoder": {"slope": AFFINE_SLOPE.tolist(), "offset": AFFINE_OFFSET.tolist(),
                    "formula": "physical=(stored-offset)/slope; only height and agb",
                    "provenance": decoder_source, "status": "community empirical hypothesis, not organizer-certified"},
        "unknown_encoding": "Other five initial-state channels and climate drivers are not recalibrated",
        "ensemble": "Equal arithmetic mean of the two fixed seeds within each arm; no weight fitting",
        "metric": "Mean squared error across year-40 height/10.90645 and agb/3.39567",
        "inference_clamp": "Nonnegative predictions applied both during evaluation and export",
        "bootstrap": {"unit": "10-degree geographic block", "replicates": 2000, "seed": 20261113},
        "climate_regimes": "Training-defined tertiles of RMS standardized 40-year mean forcing distance to the training centroid",
        "resource_limits": {"cpu_threads": 4, "maximum_seconds": 7200, "stop_before_utc": OFFICIAL_CUTOFF.isoformat()},
    }
    plan_path = output / "preregistered_plan.json"
    if plan_path.exists():
        existing = read_json(plan_path)
        if existing != plan:
            raise RuntimeError("Frozen plan/source hash mismatch; use a new experiment output directory")
        if (output / "site_partition.npz").exists() and (output / "climate_regimes.npz").exists():
            return existing
    atomic_json(plan_path, plan)
    np.savez(output / "site_partition.npz", folds=folds, blocks=blocks, **site_groups)
    climate = np.empty((n_sites, 136), dtype=np.float32)
    for start in range(0, n_sites, 256):
        climate[start:start + 256] = clean_input(sources["train_xann"][start:start + 256]).mean(1)
    mean = climate[site_groups["train"]].mean(0, dtype=np.float64)
    deviation = np.maximum(climate[site_groups["train"]].std(0, dtype=np.float64), 1e-4)
    distance = np.sqrt(np.mean(np.square((climate - mean) / deviation), axis=1))
    thresholds = np.quantile(distance[site_groups["train"]], [1 / 3, 2 / 3])
    np.savez(output / "climate_regimes.npz", regime=np.searchsorted(thresholds, distance),
             thresholds=thresholds, feature_mean=mean, feature_std=deviation)
    atomic_json(output / "prepared_receipt.json", {"prepared_utc": utc_now(), "plan_sha256": sha256_file(plan_path)})
    LOG.info("Frozen plan saved: %s", plan_path)
    return plan


def verify_plan(root: Path, output: Path) -> dict[str, Any]:
    """Verify complete data, code, environment, configuration and partition provenance."""
    plan = read_json(output / "preregistered_plan.json")
    if plan["experiment"] != EXPERIMENT or plan["code_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("The current script differs from the preregistered implementation")
    if plan["environment"] != environment_fingerprint():
        raise RuntimeError("Software environment changed after preregistration; refusing exact resume")
    for name, path in source_paths(root).items():
        if path.stat().st_size != plan["sources"][name]["bytes"]:
            raise RuntimeError(f"Source size changed after preregistration: {name}")
        if sha256_file(path) != plan["sources"][name]["sha256"]:
            raise RuntimeError(f"Source contents changed after preregistration: {name}")
    partition = np.load(output / "site_partition.npz")
    if partition_hash(partition["folds"], partition["blocks"]) != plan["site_block_fold_sha256"]:
        raise RuntimeError("The complete site/block/fold manifest changed after preregistration")
    for group in ("train", "tune", "audit"):
        if array_hash(partition[group]) != plan["site_hashes"][group]:
            raise RuntimeError(f"Partition membership changed: {group}")
    return plan


def feature_cache(root: Path, output: Path, arm: str, plan: dict[str, Any]) -> np.ndarray:
    """Build a separate atomic memory-mapped cache; never reuse historical feature files."""
    sources = load_sources(root)
    count = len(sources["train_y0"])
    path = output / f"features_{arm}.npy"
    receipt_path = output / f"features_{arm}_receipt.json"
    expected = (count * 15, plan["feature_dimensions"][arm])
    if path.exists():
        receipt = read_json(receipt_path)
        if (receipt.get("plan_sha256") != sha256_file(output / "preregistered_plan.json")
                or receipt.get("sha256") != sha256_file(path)):
            raise RuntimeError(f"Feature cache integrity/provenance mismatch: {path}")
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape != expected or matrix.dtype != np.float32:
            raise RuntimeError(f"Unexpected cached features: {path}")
        return matrix
    temporary = path.with_suffix(".partial.npy")
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=expected)
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        annual = sources["train_xann"][start:stop] if arm == "full" else None
        matrix[start * 15:stop * 15] = features(annual, sources["train_y0"][start:stop], arm)
    matrix.flush()
    del matrix
    os.replace(temporary, path)
    atomic_json(receipt_path, {"sha256": sha256_file(path), "shape": list(expected),
                               "plan_sha256": sha256_file(output / "preregistered_plan.json")})
    return np.load(path, mmap_mode="r")


def rows_for_sites(sites: np.ndarray) -> np.ndarray:
    """Return all 15 ages in stable site-major order."""
    return (sites[:, None] * 15 + np.arange(15)[None]).reshape(-1)


def feature_scaler(matrix: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate scaling only from the frozen training partition."""
    sample = np.random.default_rng(SPLIT_SEED).choice(train_rows, min(120000, len(train_rows)), replace=False)
    sums = np.zeros(matrix.shape[1], dtype=np.float64)
    squares = np.zeros_like(sums)
    for start in range(0, len(sample), 4096):
        part = np.asarray(matrix[sample[start:start + 4096]], dtype=np.float64)
        sums += part.sum(0)
        squares += np.square(part).sum(0)
    mean = sums / len(sample)
    deviation = np.sqrt(np.maximum(squares / len(sample) - mean ** 2, 1e-8))
    return mean.astype(np.float32), deviation.astype(np.float32)


def normalized_rows(matrix: np.ndarray, rows: np.ndarray, mean: np.ndarray, deviation: np.ndarray) -> np.ndarray:
    """Load one partition into bounded host memory for efficient GPU batches."""
    result = np.empty((len(rows), matrix.shape[1]), dtype=np.float32)
    for start in range(0, len(rows), 8192):
        part = matrix[rows[start:start + 8192]]
        result[start:start + len(part)] = np.clip((part - mean) / deviation, -8, 8)
    return result


def make_model(torch: Any, dimension: int, plan: dict[str, Any]) -> Any:
    """Build the same residual MLP architecture for both controlled arms."""
    layers = []
    current = dimension
    for _ in range(plan["depth"]):
        layers.extend([torch.nn.Linear(current, plan["width"]), torch.nn.LayerNorm(plan["width"]),
                       torch.nn.GELU(), torch.nn.Dropout(plan["dropout"])])
        current = plan["width"]
    layers.append(torch.nn.Linear(current, 2))
    return torch.nn.Sequential(*layers)


def predict_tensor(model: Any, matrix: Any, anchor: np.ndarray, torch: Any, device: str) -> np.ndarray:
    """Predict normalized targets, enforcing the same clamp as submitted CSV files."""
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(matrix), 8192):
            batch = matrix[start:start + 8192].to(device)
            delta = model(batch).float().cpu().numpy()
            values.append(np.maximum(delta + anchor[start:start + len(delta)], 0))
    return np.concatenate(values)


def save_checkpoint(path: Path, payload: dict[str, Any], torch: Any) -> None:
    """Write a complete self-describing inference checkpoint atomically."""
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_one(root: Path, output: Path, arm: str, seed: int, plan: dict[str, Any]) -> dict[str, Any]:
    """Train one fixed configuration; fold 1 labels are never inspected here."""
    import torch
    torch.set_num_threads(4)
    run_id = f"{arm}_seed{seed}"
    run_directory = output / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = run_directory / "checkpoint.pt"
    record_path = run_directory / "training_record.json"
    if record_path.exists():
        record = read_json(record_path)
        if not checkpoint.exists() or sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError(f"Completed checkpoint integrity failure: {run_id}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (payload.get("run_id") != run_id or payload.get("arm") != arm or payload.get("seed") != seed
                or payload.get("plan_sha256") != sha256_file(output / "preregistered_plan.json")
                or payload.get("feature_schema") != f"{EXPERIMENT}:{arm}"):
            raise RuntimeError(f"Completed checkpoint configuration/seed/schema mismatch: {run_id}")
        return record
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    sources = load_sources(root)
    partition = np.load(output / "site_partition.npz")
    train_rows = rows_for_sites(partition["train"])
    tune_rows = rows_for_sites(partition["tune"])
    # Index only training and tuning targets. The reserved audit labels remain unread.
    train_target = np.asarray(sources["train_ydec"][partition["train"], :, 40, :2], dtype=np.float32).reshape(-1, 2) / SCALES
    tune_target = np.asarray(sources["train_ydec"][partition["tune"], :, 40, :2], dtype=np.float32).reshape(-1, 2) / SCALES
    if not np.isfinite(train_target).all() or not np.isfinite(tune_target).all():
        raise ValueError("Nonfinite train/tune labels; do not silently change the frozen population")
    train_anchor = (anchor_at_40(sources["train_y0"][partition["train"]])[:, :, :2] / SCALES).reshape(-1, 2)
    tune_anchor = (anchor_at_40(sources["train_y0"][partition["tune"]])[:, :, :2] / SCALES).reshape(-1, 2)
    matrix = feature_cache(root, output, arm, plan)
    mean, deviation = feature_scaler(matrix, train_rows)
    x_train = torch.from_numpy(normalized_rows(matrix, train_rows, mean, deviation))
    x_tune = torch.from_numpy(normalized_rows(matrix, tune_rows, mean, deviation))
    del matrix
    y_train = torch.from_numpy(train_target)
    a_train = torch.from_numpy(train_anchor)
    model = make_model(torch, len(mean), plan).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=plan["epochs"])
    scaler = torch.amp.GradScaler("cuda")
    best, patience, best_epoch = float("inf"), 0, -1
    history = []
    for epoch in range(plan["epochs"]):
        check_runtime()
        model.train()
        total_loss, count = 0.0, 0
        for selected in torch.randperm(len(x_train)).split(plan["batch_size"]):
            check_runtime()
            batch = x_train[selected].to("cuda:0")
            labels = y_train[selected].to("cuda:0")
            anchor = a_train[selected].to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(batch).float() + anchor
                loss = (prediction - labels).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in {run_id}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(selected)
            count += len(selected)
        predicted = predict_tensor(model, x_tune, tune_anchor, torch, "cuda:0")
        mse = float(np.mean(np.square(predicted.astype(np.float64) - tune_target)))
        schedule.step()
        history.append({"epoch": epoch, "training_mse": total_loss / count, "tune_mse": mse})
        LOG.info("run=%s epoch=%d train_mse=%.8f tune_mse=%.8f", run_id, epoch, total_loss / count, mse)
        if mse < best - 1e-7:
            best, patience, best_epoch = mse, 0, epoch
            reload_probe = predict_tensor(model, x_tune[:128], tune_anchor[:128], torch, "cuda:0")
            save_checkpoint(checkpoint, {
                "experiment": EXPERIMENT, "run_id": run_id, "arm": arm, "seed": seed,
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "plan": plan, "plan_sha256": sha256_file(output / "preregistered_plan.json"),
                "feature_mean": mean, "feature_std": deviation, "feature_dimension": len(mean),
                "feature_schema": f"{EXPERIMENT}:{arm}", "test_state_transforms": list(TRANSFORMS),
                "site_hashes": plan["site_hashes"], "scales": SCALES,
                "best_tune_mse": best, "best_epoch": epoch, "saved_utc": utc_now(),
                "reload_probe_predictions": reload_probe,
                "reload_probe_tune_rows": tune_rows[:128].copy(),
            }, torch)
        else:
            patience += 1
            if patience >= plan["patience"]:
                break
    restored, payload = load_model(checkpoint, "cuda:0")
    probe = predict_tensor(restored, x_tune[:128], tune_anchor[:128], torch, "cuda:0")
    if not np.allclose(probe, payload["reload_probe_predictions"], atol=1e-6, rtol=1e-6):
        raise RuntimeError("Checkpoint reload failed atol=rtol=1e-6; do not mark the run complete")
    record = {"run_id": run_id, "completed_utc": utc_now(), "best_tune_mse": best,
              "best_epoch": best_epoch, "epochs_completed": len(history), "history": history,
              "checkpoint_sha256": sha256_file(checkpoint), "audit_accessed": False,
              "plan_sha256": sha256_file(output / "preregistered_plan.json"),
              "reload_check": {"passed": True, "atol": 1e-6, "rtol": 1e-6,
                               "max_abs_difference": float(np.max(np.abs(probe - payload["reload_probe_predictions"])))} }
    atomic_json(record_path, record)
    del model, restored, optimizer, schedule, scaler, x_train, x_tune, y_train, a_train
    gc.collect()
    torch.cuda.empty_cache()
    return record


def load_model(checkpoint: Path, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """Restore a locally produced checkpoint without fitting or changing parameters."""
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["experiment"] != EXPERIMENT or payload["arm"] not in ARMS:
        raise ValueError("Checkpoint does not belong to this frozen experiment")
    if ("code_sha256" in payload["plan"] and
            payload["plan"]["code_sha256"] != sha256_file(Path(__file__).resolve())):
        raise RuntimeError("Inference source differs from the frozen checkpoint implementation")
    model = make_model(torch, payload["feature_dimension"], payload["plan"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def infer_sites(model: Any, payload: dict[str, Any], annual: np.ndarray, state: np.ndarray,
                transform: str, device: str = "cpu", roundtrip: bool = False,
                diagnostics: dict[str, Any] | None = None, sites_per_chunk: int = 128) -> np.ndarray:
    """Infer in site chunks, optionally testing a known encode/decode round trip."""
    import torch
    outputs = []
    for start in range(0, len(state), sites_per_chunk):
        if device == "cuda:0":
            check_runtime()
        stop = min(start + sites_per_chunk, len(state))
        raw_state = state[start:stop]
        if roundtrip:
            raw_state = transform_state(raw_state, "encode_stress")
        transformed = transform_state(raw_state, transform)
        vector = features(annual[start:stop] if payload["arm"] == "full" else None,
                          transformed, payload["arm"])
        vector = (vector - payload["feature_mean"]) / payload["feature_std"]
        if diagnostics is not None:
            diagnostics["clipped_values"] = diagnostics.get("clipped_values", 0) + int(np.count_nonzero(np.abs(vector) > 8))
            diagnostics["total_feature_values"] = diagnostics.get("total_feature_values", 0) + int(vector.size)
            diagnostics["clipped_fraction"] = diagnostics["clipped_values"] / diagnostics["total_feature_values"]
        vector = np.clip(vector, -8, 8)
        anchor = (anchor_at_40(transformed)[:, :, :2] / SCALES).reshape(-1, 2)
        outputs.append(predict_tensor(model, torch.from_numpy(vector), anchor, torch, device))
    return np.concatenate(outputs).astype(np.float32)


def expected_ids() -> list[str]:
    """Construct the official scenario/site/seed-age order for a strict contract check."""
    return [f"{scenario}_{site}_{age}" for scenario in ("ssp126", "ssp585")
            for site in range(6171) for age in AGES.astype(int)]


def write_submission(root: Path, path: Path, predictions: np.ndarray) -> dict[str, Any]:
    """Write a finite, nonnegative, template-ordered normalized CSV without uploading it."""
    with (root / "data" / "sample_submission.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "height", "agb"]:
            raise ValueError("Unexpected sample submission columns")
        template = [row["id"] for row in reader]
    if template != expected_ids() or len(set(template)) != 185130:
        raise ValueError("Official template ordering or unique IDs do not match the frozen contract")
    if predictions.shape != (185130, 2) or not np.isfinite(predictions).all() or np.min(predictions) < 0:
        raise ValueError("Invalid prediction shape, finiteness, or nonnegative physical contract")
    temporary = path.with_suffix(".partial.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "height", "agb"])
        for identifier, value in zip(template, predictions):
            writer.writerow([identifier, format(float(value[0]), ".9g"), format(float(value[1]), ".9g")])
    os.replace(temporary, path)
    return {"file": str(path), "sha256": sha256_file(path), "rows": len(template),
            "columns": ["id", "height", "agb"], "finite": True, "template_exact": True,
            "normalization": "height/10.90645, agb/3.39567"}


def predict_test(root: Path, checkpoint: Path, transform: str, device: str,
                 diagnostics: dict[str, Any] | None = None) -> np.ndarray:
    """Reproduce a complete test prediction from weights and data alone."""
    model, payload = load_model(checkpoint, device)
    outputs = []
    for scenario in ("ssp126", "ssp585"):
        annual = np.load(root / "data" / f"test_xann_{scenario}.npy", mmap_mode="r")
        state = np.load(root / "data" / f"{scenario}_test_y0.npy", mmap_mode="r")
        if state.shape != (6171, 15, 7) or annual.shape != (6171, 40, 136):
            raise ValueError(f"Unexpected test shapes for {scenario}")
        if transform == "community_affine":
            verify_test_encoding(state)
        outputs.append(infer_sites(model, payload, annual, state, transform, device, diagnostics=diagnostics))
    return np.concatenate(outputs)


def legacy_benchmark(root: Path, output: Path, checkpoints: list[Path], device: str = "cpu") -> dict[str, Any]:
    """Isolate encoding effects on historical CUP weights without any retraining.

    Historical state-dict keys have a ``net.`` prefix. Removing that wrapper
    preserves every parameter and the layer order exactly. The raw branch must
    reproduce any existing CSV to tolerance before its decoded counterpart is
    regarded as a valid inference candidate.
    """
    import torch
    torch.set_num_threads(4)
    if device == "cuda:0":
        torch.backends.cuda.matmul.allow_tf32 = True
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"created_utc": utc_now(), "training_performed": False,
                              "gpu_used": device == "cuda:0", "device": device,
                              "decoder_status": "Frozen community hypothesis, not official",
                              "checkpoints": {}}
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not all(key in payload for key in ("trial", "feature_dim", "feature_mean", "feature_std", "model_state")):
            raise ValueError(f"Not a supported historical CUP checkpoint: {checkpoint}")
        if payload["feature_dim"] != 1209:
            raise ValueError("Historical full-forcing schema changed; do not guess its semantics")
        trial = payload["trial"]
        plan = {"width": trial["width"], "depth": trial["depth"], "dropout": trial["dropout"]}
        model = make_model(torch, 1209, plan).to(device).eval()
        state_dict = {key.removeprefix("net."): value for key, value in payload["model_state"].items()}
        model.load_state_dict(state_dict, strict=True)
        adapted = {"arm": "full", "feature_mean": payload["feature_mean"], "feature_std": payload["feature_std"]}
        result: dict[str, Any] = {"checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint),
                                  "trial": trial, "candidates": {}}
        predictions = {}
        for transform in TRANSFORMS:
            pieces = []
            for scenario in ("ssp126", "ssp585"):
                annual = np.load(root / "data" / f"test_xann_{scenario}.npy", mmap_mode="r")
                state = np.load(root / "data" / f"{scenario}_test_y0.npy", mmap_mode="r")
                if transform == "community_affine":
                    verify_test_encoding(state)
                pieces.append(infer_sites(model, adapted, annual, state, transform, device, sites_per_chunk=256))
            prediction = np.concatenate(pieces)
            predictions[transform] = prediction
            name = f"legacy_{checkpoint.stem}_{transform}"
            np.save(output / f"{name}.npy", prediction)
            result["candidates"][transform] = write_submission(root, output / f"{name}.csv", prediction)
        historical = root / "submissions" / f"{checkpoint.stem}.csv"
        if historical.exists():
            with historical.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            if [row["id"] for row in rows] != expected_ids():
                raise ValueError("Historical candidate ID order differs from the official template")
            old = np.asarray([[float(row["height"]), float(row["agb"])] for row in rows], dtype=np.float64)
            difference = float(np.max(np.abs(old - predictions["raw"])))
            atol, rtol = (1e-6, 1e-6) if device == "cuda:0" else (1e-4, 0.0)
            passed = bool(np.allclose(old, predictions["raw"], atol=atol, rtol=rtol))
            result["historical_raw_reproduction"] = {"source": str(historical), "max_abs_difference": difference,
                                                     "atol": atol, "rtol": rtol, "passed": passed,
                                                     "backend": device, "sites_per_chunk": 256}
            if not passed:
                atomic_json(output / f"legacy_{checkpoint.stem}_FAILED.json", result)
                raise RuntimeError(f"Historical raw inference could not be reproduced: maximum difference {difference}")
        else:
            result["historical_raw_reproduction"] = {"source": None, "passed": None,
                                                     "reason": "No historical CSV exists at the canonical path"}
        delta = predictions["community_affine"] - predictions["raw"]
        result["decoder_prediction_change"] = {"rmse": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
                                                  "mean_by_target": delta.mean(0).tolist()}
        report["checkpoints"][checkpoint.stem] = result
    atomic_json(output / "legacy_encoding_benchmark.json", report)
    return report


def metric(prediction: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Calculate the exact two-column normalized MSE in float64."""
    squares = np.square(prediction.astype(np.float64) - labels.astype(np.float64))
    return {"mse": float(squares.mean()), "height_mse": float(squares[:, 0].mean()),
            "agb_mse": float(squares[:, 1].mean())}


def stratified_metrics(prediction: np.ndarray, labels: np.ndarray, regimes: np.ndarray) -> dict[str, Any]:
    """Report every age and predeclared forcing-distance regime without fitting weights."""
    pred_sites, target_sites = prediction.reshape(-1, 15, 2), labels.reshape(-1, 15, 2)
    return {"by_age": {str(int(age)): metric(pred_sites[:, index], target_sites[:, index])
                        for index, age in enumerate(AGES)},
            "by_climate_distance_regime": {
                str(int(regime)): {"sites": int(np.count_nonzero(regimes == regime)),
                                   **metric(pred_sites[regimes == regime].reshape(-1, 2),
                                            target_sites[regimes == regime].reshape(-1, 2))}
                for regime in np.unique(regimes)}}


def paired_bootstrap(left: np.ndarray, right: np.ndarray, labels: np.ndarray, blocks: np.ndarray,
                     plan: dict[str, Any]) -> dict[str, Any]:
    """Resample complete geographic blocks for the frozen full-minus-curve2 comparison."""
    differences = (np.square(left.astype(np.float64) - labels) -
                   np.square(right.astype(np.float64) - labels)).mean(1)
    row_blocks = np.repeat(blocks, 15)
    unique, inverse = np.unique(row_blocks, return_inverse=True)
    total = np.bincount(inverse, weights=differences)
    count = np.bincount(inverse)
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    values = []
    for _ in range(plan["bootstrap"]["replicates"]):
        selected = rng.integers(0, len(unique), size=len(unique))
        values.append(float(total[selected].sum() / count[selected].sum()))
    return {"comparison": "full minus curve2; negative favors full", "mean_delta": float(differences.mean()),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(), "blocks": len(unique),
            "replicates": len(values), "interpretation": "Experiment audit only; not proof of hidden SSP performance"}


def finalize(root: Path, output: Path, device: str = "cpu") -> dict[str, Any]:
    """Perform the one frozen audit and export every predeclared candidate and ensemble."""
    plan = verify_plan(root, output)
    final_path = output / "ensemble_summary.json"
    if final_path.exists():
        return read_json(final_path)
    runs = [(arm, seed, output / f"{arm}_seed{seed}") for arm in ARMS for seed in SEEDS]
    for _, _, directory in runs:
        record = read_json(directory / "training_record.json")
        if sha256_file(directory / "checkpoint.pt") != record["checkpoint_sha256"]:
            raise RuntimeError("An incomplete or modified training run blocks audit access")
    receipt = output / "audit_access_receipt.json"
    if not receipt.exists():
        atomic_json(receipt, {"first_access_utc": utc_now(), "plan_sha256": sha256_file(output / "preregistered_plan.json"),
                             "policy": "Frozen evaluation only; reruns may recover interrupted inference, not tune"})
    partition = np.load(output / "site_partition.npz")
    audit_sites = partition["audit"]
    sources = load_sources(root)
    labels = (np.asarray(sources["train_ydec"][audit_sites, :, 40, :2], dtype=np.float32) / SCALES).reshape(-1, 2)
    if not np.isfinite(labels).all():
        raise ValueError("Nonfinite audit labels; frozen population may not be changed silently")
    state, annual = sources["train_y0"][audit_sites], sources["train_xann"][audit_sites]
    regimes = np.load(output / "climate_regimes.npz")["regime"][audit_sites]
    audit_by_arm: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    test_by_arm = {arm: {transform: [] for transform in TRANSFORMS} for arm in ARMS}
    summary: dict[str, Any] = {"experiment": EXPERIMENT, "completed_utc": None, "runs": {}, "ensembles": {},
                              "decoder_status": plan["decoder"], "audit_independence": plan["audit_independence"]}
    for arm, seed, directory in runs:
        checkpoint = directory / "checkpoint.pt"
        model, payload = load_model(checkpoint, device)
        prediction_path = directory / "audit_predictions.npz"
        if prediction_path.exists():
            saved = np.load(prediction_path)
            clean, encoded, roundtrip = saved["clean"], saved["encoded"], saved["roundtrip"]
        else:
            clean = infer_sites(model, payload, annual, state, "raw", device)
            encoded = infer_sites(model, payload, annual, state, "raw", device, roundtrip=True)
            roundtrip = infer_sites(model, payload, annual, state, "community_affine", device, roundtrip=True)
            np.savez_compressed(prediction_path, clean=clean, encoded=encoded, roundtrip=roundtrip, site_ids=audit_sites)
        error = float(np.max(np.abs(clean - roundtrip)))
        if error > 1e-3:
            raise RuntimeError(f"Affine encoding/decoding round trip failed numerical tolerance: {error}")
        result = {"audit_clean": metric(clean, labels), "audit_strata": stratified_metrics(clean, labels, regimes),
                  "audit_encoded_without_decoder": metric(encoded, labels),
                  "audit_encoded_then_decoded": metric(roundtrip, labels), "roundtrip_max_abs": error,
                  "stress_interpretation": "Known-label encoding stress validates code consistency, not the empirical test decoder",
                  "test_candidates": {}}
        audit_by_arm[arm].append(clean)
        for transform in TRANSFORMS:
            saved_test = directory / f"test_predictions_{transform}.npy"
            diagnostics_path = directory / f"test_clipping_{transform}.json"
            if saved_test.exists():
                predicted = np.load(saved_test)
                diagnostics = read_json(diagnostics_path) if diagnostics_path.exists() else {"status": "Interrupted before diagnostics were committed"}
            else:
                diagnostics = {}
                predicted = predict_test(root, checkpoint, transform, device, diagnostics)
                np.save(saved_test, predicted)
                atomic_json(diagnostics_path, diagnostics)
            result["test_candidates"][transform] = write_submission(root, directory / f"submission_{transform}.csv", predicted)
            result["test_candidates"][transform]["feature_clipping"] = diagnostics
            test_by_arm[arm][transform].append(predicted)
        atomic_json(directory / "audit_and_inference_record.json", result)
        summary["runs"][f"{arm}_seed{seed}"] = result
    averaged = {}
    for arm in ARMS:
        averaged[arm] = np.mean(np.stack(audit_by_arm[arm]), axis=0, dtype=np.float64).astype(np.float32)
        np.save(output / f"ensemble_{arm}_audit.npy", averaged[arm])
        result = {"audit_metric": metric(averaged[arm], labels), "audit_strata": stratified_metrics(averaged[arm], labels, regimes),
                  "weights": [0.5, 0.5], "test_candidates": {}}
        for transform in TRANSFORMS:
            predicted = np.mean(np.stack(test_by_arm[arm][transform]), axis=0, dtype=np.float64).astype(np.float32)
            np.save(output / f"ensemble_{arm}_{transform}.npy", predicted)
            result["test_candidates"][transform] = write_submission(root, output / f"ensemble_{arm}_{transform}.csv", predicted)
        summary["ensembles"][arm] = result
    summary["paired_block_bootstrap"] = paired_bootstrap(averaged["full"], averaged["curve2"], labels,
                                                          partition["blocks"][audit_sites], plan)
    summary["completed_utc"] = utc_now()
    summary["official_submission"] = "None; this program never calls Kaggle or selects final submissions"
    atomic_json(final_path, summary)
    return summary


@contextlib.contextmanager
def exclusive_gpu(root: Path) -> Iterator[None]:
    """Respect the existing IEEE single-owner lock and the exact assigned physical GPU."""
    import fcntl
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES exactly to {GPU_UUID}")
    if not torch.cuda.is_available() or "RTX 3080" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The assigned RTX 3080 is not available")
    with (root / "gpu1_ieee.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The historical IEEE service still owns the GPU lock; coordinate its handover first") from error
        yield


def request_stop(signum: int, frame: Any) -> None:
    """Stop at the next epoch boundary without claiming an incomplete run finished."""
    global STOP
    STOP = True
    LOG.warning("Received signal %s; stopping at the next epoch boundary", signum)


def smoke_test() -> dict[str, Any]:
    """Exercise representation, partition, checkpoint reload, and inference on CPU only."""
    import torch
    torch.set_num_threads(2)
    rng = np.random.default_rng(17)
    state = rng.uniform(0, 35, size=(6, 15, 7)).astype(np.float32)
    annual = rng.normal(size=(6, 40, 136)).astype(np.float32)
    decoded = transform_state(transform_state(state, "encode_stress"), "community_affine")
    assert np.allclose(decoded, state, atol=4e-6, rtol=1e-6)
    assert np.array_equal(decoded[:, :, 2:], state[:, :, 2:])
    try:
        verify_test_encoding(state)
    except ValueError:
        pass
    else:
        raise AssertionError("An incompatible test encoding was not rejected")
    assert features(annual, state, "curve2").shape == (90, 36)
    assert features(annual, state, "full").shape == (90, 1209)
    changed_climate = features(annual * 10, state, "curve2")
    assert np.array_equal(changed_climate, features(annual, state, "curve2"))
    mask = np.ones((100, 100), dtype=bool)
    folds, blocks = geographic_partition(mask, int(mask.sum()))
    assert set(folds) == set(range(5))
    for block in np.unique(blocks):
        assert np.unique(folds[blocks == block]).size == 1
    plan = {"width": 16, "depth": 2, "dropout": 0.12}
    model = make_model(torch, 36, plan).eval()
    payload = {"experiment": EXPERIMENT, "arm": "curve2", "model_state": model.state_dict(),
               "feature_dimension": 36, "feature_mean": np.zeros(36, dtype=np.float32),
               "feature_std": np.ones(36, dtype=np.float32), "plan": plan}
    expected = infer_sites(model, payload, annual, state, "raw")
    with tempfile.TemporaryDirectory(prefix="carbon-transfer-smoke-") as temporary:
        temporary_root = Path(temporary)
        path = temporary_root / "checkpoint.pt"
        save_checkpoint(path, payload, torch)
        loaded, restored = load_model(path)
        actual = infer_sites(loaded, restored, annual, state, "raw")
        (temporary_root / "data").mkdir()
        ids = expected_ids()
        template = temporary_root / "data" / "sample_submission.csv"
        with template.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "height", "agb"])
            writer.writerows([identifier, 0, 0] for identifier in ids)
        invalid = np.zeros((185130, 2), dtype=np.float32)
        invalid[0, 0] = np.nan
        try:
            write_submission(temporary_root, temporary_root / "invalid.csv", invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Nonfinite submission values were not rejected")
        ids[-1] = ids[0]
        with template.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "height", "agb"])
            writer.writerows([identifier, 0, 0] for identifier in ids)
        try:
            write_submission(temporary_root, temporary_root / "duplicate.csv", np.zeros((185130, 2), dtype=np.float32))
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicated submission IDs were not rejected")
    assert np.array_equal(expected, actual)
    assert np.isfinite(actual).all() and (actual >= 0).all()
    assert len(expected_ids()) == 185130 and len(set(expected_ids())) == 185130
    return {"passed": True, "device": "cpu", "gpu_used": False,
            "checks": ["affine roundtrip", "incompatible offset rejected", "five untouched channels", "feature shapes",
                       "curve climate independence", "geographic isolation", "checkpoint exact reload",
                       "finite clamped inference", "nonfinite predictions rejected", "duplicate IDs rejected", "official ID contract"]}


def main() -> None:
    """Expose separate preparation, training, audit, and no-retraining inference actions."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("prepare", "train", "run", "finalize", "predict", "legacy-benchmark", "smoke"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decoder-source", default="Frozen community-published affine hypothesis: height=(stored+0.900947)/0.876341; agb=(stored+0.401412)/0.876; not official")
    parser.add_argument("--target-audit", type=Path, help="Passing December year-40 raw-source/cache comparison JSON")
    parser.add_argument("--arm", choices=ARMS, default="curve2")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--checkpoints", type=Path, nargs="+")
    parser.add_argument("--test-transform", choices=TRANSFORMS, default="community_affine")
    parser.add_argument("--prediction-path", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.action == "smoke":
        print(json.dumps(smoke_test(), indent=2))
        return
    if args.root is None:
        parser.error("--root is required except for smoke")
    root = args.root.resolve()
    output = (args.output or root / "research" / EXPERIMENT).resolve()
    if args.action == "prepare":
        if args.target_audit is None:
            parser.error("prepare requires --target-audit; raw-source temporal alignment must be verified first")
        print(json.dumps(prepare(root, output, args.decoder_source, args.target_audit), indent=2))
        return
    if args.action == "legacy-benchmark":
        if not args.checkpoints:
            parser.error("legacy-benchmark requires --checkpoints")
        if args.device == "cuda:0":
            with exclusive_gpu(root):
                result = legacy_benchmark(root, output / "legacy_benchmark", args.checkpoints, args.device)
        else:
            result = legacy_benchmark(root, output / "legacy_benchmark", args.checkpoints)
        print(json.dumps(result, indent=2))
        return
    if args.action == "predict":
        if not args.checkpoints:
            parser.error("predict requires --checkpoints")
        if args.device != "cpu":
            parser.error("Standalone predict is CPU-only; GPU use is reserved for the locked run action")
        predictions = [predict_test(root, path, args.test_transform, "cpu") for path in args.checkpoints]
        average = np.mean(np.stack(predictions), axis=0, dtype=np.float64).astype(np.float32)
        destination = args.prediction_path or output / f"reproduced_{args.test_transform}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(write_submission(root, destination, average), indent=2))
        return
    plan = verify_plan(root, output)
    if args.action == "finalize":
        if args.device != "cpu":
            parser.error("Standalone finalize is CPU-only; locked run performs GPU inference")
        print(json.dumps(finalize(root, output, "cpu"), indent=2))
        return
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with exclusive_gpu(root):
        if args.action == "train":
            print(json.dumps(train_one(root, output, args.arm, args.seed, plan), indent=2))
        else:
            for arm in ARMS:
                for seed in SEEDS:
                    train_one(root, output, arm, seed, plan)
            if not STOP:
                print(json.dumps(finalize(root, output, "cuda:0"), indent=2))


if __name__ == "__main__":
    main()
