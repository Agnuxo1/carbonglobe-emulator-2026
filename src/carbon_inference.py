#!/usr/bin/env python3
"""Portable inference primitives for the frozen CarbonGlobe model families.

This module contains no training loop, hidden-label access, calendar cutoff or
Kaggle client. Feature mathematics, state decoding, auxiliary shapes, model
construction, prediction batching and CSV formatting are copied unchanged from
the identified frozen sources. Scenario iteration uses its own monotonic runtime
budget and signal handling. It never imports, modifies or patches those engines.

Portable mode supports an organizer's CPU or CUDA device. Mission mode preserves
the reserved GPU UUID and the existing gpu1_ieee.lock on the research machine.
"""
from __future__ import annotations

import contextlib
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import platform
import re
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "4"

import numpy as np

TRANSFER_SHA256 = "230680807f1ec9a3ccd89c8739ac149284cb5c1b5eaa5b11707fcbd213fa2905"
AUXILIARY_SHA256 = "5684f74e7254d045249b68bcec4d0fa889caf4ad50867c7b48e690f16483350c"
MISSION_GPU_UUID = "GPU-d60d12af-82b8-9913-7a5e-1177e774dafa"
EXPERIMENT = "transfer_20261113_v1"
STUDY = "curve2_aux_shape_20261113_v1"
ARM = "curve2_aux_shape"
ARMS = ("curve2", "full")
FAMILIES = ("transfer", "aux_shape")
TRANSFORMS = ("raw", "community_affine")
SEEDS = (20261113, 20261114)
AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500], dtype=np.float32)
SCALES = np.array([10.90645, 3.39567], dtype=np.float32)
AFFINE_SLOPE = np.array([0.876341, 0.876], dtype=np.float32)
AFFINE_OFFSET = np.array([-0.900947, -0.401412], dtype=np.float32)
AUXILIARY_NAMES = ("soil", "lai", "gpp", "npp", "rh")
FEATURE_DIMENSION = 116
CHECKPOINT_CONTRACT = SimpleNamespace(EXPERIMENT=EXPERIMENT, ARMS=ARMS, STUDY=STUDY, ARM=ARM)


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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Commit one complete evidence record atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def environment_fingerprint() -> dict[str, Any]:
    """Record software versions needed to verify an exact training resume."""
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "platform": platform.system(), "cpu_threads": 4}


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


def auxiliary_shape(auxiliary: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return variable-major 75 shapes followed by five precision-aware flags.

    np.spacing(float32_max) is infinite. Only those entries use the backward
    representable spacing, computed by a float64 subtraction. Subnormals and
    zero use their actual positive float32 spacing, without an epsilon floor.
    """
    values = np.asarray(auxiliary)
    if values.ndim != 3 or values.shape[1:] != (15, 5):
        raise ValueError(f"Expected (sites,15,5) auxiliary profiles, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Auxiliary profile contains a nonfinite original value")
    values64 = values.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        values32 = values.astype(np.float32)
    if not np.isfinite(values32).all():
        raise ValueError("Auxiliary profile cannot be represented as finite float32")
    absolute32 = np.abs(values32)
    with np.errstate(over="ignore", invalid="ignore"):
        spacing = np.spacing(absolute32).astype(np.float64)
    overflow_spacing = np.isinf(spacing)
    if overflow_spacing.any():
        lower = np.nextafter(absolute32[overflow_spacing], np.float32(0)).astype(np.float64)
        spacing[overflow_spacing] = absolute32[overflow_spacing].astype(np.float64) - lower
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise FloatingPointError("Float32 spacing is not finite and strictly positive")
    minimum = values64.min(axis=1)
    extent = values64.max(axis=1) - minimum
    threshold = 8.0 * spacing.max(axis=1)
    degenerate = extent <= threshold
    numerator = values64 - minimum[:, None, :]
    shape = np.zeros_like(values64)
    np.divide(numerator, extent[:, None, :], out=shape, where=~degenerate[:, None, :])
    np.clip(shape, 0.0, 1.0, out=shape)
    ordered = shape.transpose(0, 2, 1).reshape(len(values), 75)
    result = np.concatenate([ordered, degenerate.astype(np.float64)], axis=1).astype(np.float32)
    diagnostics = {
        "sites": len(values), "profiles": int(degenerate.size),
        "degenerate_by_variable": {name: int(degenerate[:, index].sum())
                                   for index, name in enumerate(AUXILIARY_NAMES)},
        "backward_spacing_fallback_values": int(overflow_spacing.sum()),
    }
    return result, diagnostics


def feature_matrix(state: np.ndarray, engine: Any) -> np.ndarray:
    """Append auxiliary shape to the exact frozen curve2 features for each age."""
    state = np.asarray(state)
    if state.ndim != 3 or state.shape[1:] != (15, 7) or not np.isfinite(state).all():
        raise ValueError("Expected a finite site x 15 x 7 initial state")
    # Calculate shapes from the original values, before the parent helper's
    # unrelated sentinel handling of its height/AGB feature block.
    shape, _ = auxiliary_shape(state[:, :, 2:7])
    curve = engine.features(None, state, "curve2").reshape(len(state), 15, 36)
    repeated = np.broadcast_to(shape[:, None, :], (len(state), 15, 80))
    return np.concatenate([curve, repeated], axis=2).reshape(-1, FEATURE_DIMENSION).astype(np.float32)


def decode_initial_state(state: np.ndarray, transform: str, engine: Any) -> np.ndarray:
    """Decode target channels only, preserving every auxiliary input without clamping."""
    state = np.asarray(state)
    if state.shape[1:] != (15, 7) or not np.isfinite(state).all():
        raise ValueError("Initial states must be finite and have shape (sites,15,7)")
    result = state.copy()
    if transform == "community_affine":
        engine.verify_test_encoding(state)
        result[:, :, :2] = (result[:, :, :2] - engine.AFFINE_OFFSET) / engine.AFFINE_SLOPE
    elif transform != "raw":
        raise ValueError(f"Unknown target-channel transform: {transform}")
    return result


def normalize_sha256(value: str | None) -> str | None:
    """Validate a complete SHA-256 digest rather than accepting partial prefixes."""
    if value is None:
        return None
    value = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected SHA-256 must contain exactly 64 hexadecimal characters")
    return value


def checkpoint_records(paths: list[Path], family: str, engine: Any, auxiliary: Any,
                       torch: Any) -> tuple[list[dict[str, Any]], str]:
    """Require the two declared seeds, one arm and one frozen plan."""
    if len(paths) != 2 or len(set(paths)) != 2:
        raise ValueError("Provide exactly two distinct checkpoint files")
    records = []
    for path in paths:
        digest = sha256_file(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if family == "transfer":
            if payload.get("experiment") != engine.EXPERIMENT or payload.get("arm") not in engine.ARMS:
                raise ValueError(f"Not a frozen transfer checkpoint: {path}")
            plan = payload["plan"]
            if plan.get("code_sha256") != TRANSFER_SHA256:
                raise RuntimeError(f"Checkpoint does not identify the pinned transfer engine: {path}")
            plan_hash = payload["plan_sha256"]
            sources = plan["sources"]
            configuration = {key: plan[key] for key in ("width", "depth", "dropout")}
        else:
            if (payload.get("study") != auxiliary.STUDY or payload.get("arm") != auxiliary.ARM
                    or payload.get("code_sha256") != AUXILIARY_SHA256
                    or payload.get("engine_sha256") != TRANSFER_SHA256):
                raise ValueError(f"Not a frozen auxiliary-shape checkpoint: {path}")
            plan_hash = payload["study_plan_sha256"]
            sources = payload["sources"]
            configuration = payload["configuration"]
        if normalize_sha256(str(plan_hash)) is None:
            raise ValueError("Checkpoint is missing its frozen plan hash")
        records.append({"path": str(path), "sha256": digest, "seed": int(payload["seed"]),
                        "arm": payload["arm"], "plan_sha256": plan_hash,
                        "feature_dimension": int(payload["feature_dimension"]),
                        "configuration": configuration, "sources": sources})
        del payload
    if sorted(record["seed"] for record in records) != list(SEEDS):
        raise ValueError(f"The ensemble must contain exactly seeds {SEEDS}")
    for key in ("arm", "plan_sha256", "feature_dimension", "configuration"):
        if records[0][key] != records[1][key]:
            raise ValueError(f"The two checkpoints disagree on {key}")
    records.sort(key=lambda record: record["seed"])
    return records, records[0]["arm"]


def verify_test_inputs(root: Path, family: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify only public inputs used by the selected inference engine."""
    names = ["ssp126_test_y0.npy", "ssp585_test_y0.npy", "sample_submission.csv"]
    if family == "transfer":
        names += ["test_xann_ssp126.npy", "test_xann_ssp585.npy"]
    verified = {}
    for name in names:
        path = root / "data" / name
        actual = sha256_file(path)
        for record in records:
            source = record["sources"].get(name)
            if not isinstance(source, dict) or source.get("sha256") != actual:
                raise RuntimeError(f"Public input differs from checkpoint provenance: {name}")
        verified[name] = {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}
    return verified


def csv_difference(actual_path: Path, reference_path: Path) -> dict[str, Any]:
    """Compare prediction CSV values and IDs without using labels or a score service."""
    maximum = [0.0, 0.0]
    squares = [0.0, 0.0]
    changed = [0, 0]
    count = 0
    with actual_path.open(newline="", encoding="utf-8-sig") as actual_stream, \
            reference_path.open(newline="", encoding="utf-8-sig") as reference_stream:
        actual, reference = csv.DictReader(actual_stream), csv.DictReader(reference_stream)
        if actual.fieldnames != ["id", "height", "agb"] or reference.fieldnames != actual.fieldnames:
            return {"comparable": False, "reason": "Column names differ from the submission contract"}
        for first, second in itertools.zip_longest(actual, reference):
            if first is None or second is None:
                return {"comparable": False, "reason": "Row counts differ", "matching_prefix_rows": count}
            if first["id"] != second["id"]:
                return {"comparable": False, "reason": "ID order differs", "first_mismatch_row": count}
            for index, key in enumerate(("height", "agb")):
                values = (float(first[key]), float(second[key]))
                if not all(np.isfinite(value) for value in values):
                    return {"comparable": False, "reason": "Nonfinite reference or generated prediction"}
                delta = values[0] - values[1]
                maximum[index] = max(maximum[index], abs(delta))
                squares[index] += delta * delta
                changed[index] += int(delta != 0)
            count += 1
    return {"comparable": True, "rows": count, "target_order": ["height", "agb"],
            "maximum_absolute_difference": maximum,
            "rmse_by_target": [float(np.sqrt(value / count)) if count else 0.0 for value in squares],
            "different_values_by_target": changed,
            "interpretation": "Prediction-to-prediction diagnostics, not validation against hidden labels"}


# The extracted auxiliary functions receive only these local mathematical
# primitives; this object is not an imported training engine.
PRIMITIVES = SimpleNamespace(features=features, verify_test_encoding=verify_test_encoding,
                             AFFINE_OFFSET=AFFINE_OFFSET, AFFINE_SLOPE=AFFINE_SLOPE)


class RuntimeBudget:
    """Limit elapsed execution time and allow cancellation without a calendar deadline."""

    def __init__(self, seconds: float = 7200.0) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Maximum runtime must be a finite positive number of seconds")
        self.seconds = float(seconds)
        self.started = time.monotonic()
        self.cancelled_by: int | None = None

    def request_stop(self, signum: int, frame: Any) -> None:
        """Record a termination request for the next inference chunk boundary."""
        self.cancelled_by = signum

    def check(self) -> None:
        """Raise on cancellation or elapsed-budget exhaustion; never inspect wall-clock dates."""
        if self.cancelled_by is not None:
            raise InterruptedError(f"Inference cancelled by signal {self.cancelled_by}")
        if time.monotonic() - self.started > self.seconds:
            raise TimeoutError(f"Inference exceeded its {self.seconds:g}-second runtime budget")

    @contextlib.contextmanager
    def signals(self) -> Iterator[None]:
        """Install temporary process handlers and restore the caller's handlers afterward."""
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, self.request_stop)
        try:
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


@contextlib.contextmanager
def execution_context(root: Path, mode: str, device: str, budget: RuntimeBudget) -> Iterator[None]:
    """Enforce explicit device ownership while keeping organizer inference portable."""
    if mode not in ("portable", "mission") or (device != "cpu" and re.fullmatch(r"cuda:[0-9]+", device) is None):
        raise ValueError("Unsupported execution mode or device")
    with budget.signals():
        budget.check()
        if device == "cpu":
            yield
            return
        if mode == "mission" and (device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != MISSION_GPU_UUID):
            raise RuntimeError("Mission CUDA inference requires the reserved UUID and logical cuda:0")
        if mode == "portable" and os.environ.get("CUDA_VISIBLE_DEVICES") == MISSION_GPU_UUID:
            raise RuntimeError("The reserved research GPU must be used through explicit mission mode")
        import torch
        index = int(device.split(":", 1)[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
        if mode == "portable":
            yield
            return
        if torch.cuda.device_count() != 1 or "RTX 3080" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Mission inference requires exactly the reserved RTX 3080")
        import fcntl
        with (root / "gpu1_ieee.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another IEEE process owns the reserved GPU lock") from error
            yield


def load_prediction_model(checkpoint: Path, family: str, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """Restore identified frozen weights without requiring the training engines at runtime."""
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if family == "transfer":
        if (payload.get("experiment") != EXPERIMENT or payload.get("arm") not in ARMS
                or payload["plan"].get("code_sha256") != TRANSFER_SHA256):
            raise ValueError("Checkpoint does not identify the supported frozen transfer family")
        expected_dimension = 36 if payload["arm"] == "curve2" else 1209
        configuration = payload["plan"]
    elif family == "aux_shape":
        if (payload.get("study") != STUDY or payload.get("arm") != ARM
                or payload.get("code_sha256") != AUXILIARY_SHA256
                or payload.get("engine_sha256") != TRANSFER_SHA256):
            raise ValueError("Checkpoint does not identify the supported frozen auxiliary-shape family")
        expected_dimension = FEATURE_DIMENSION
        configuration = payload["configuration"]
    else:
        raise ValueError(f"Unsupported checkpoint family: {family}")
    if payload["feature_dimension"] != expected_dimension:
        raise ValueError("Checkpoint feature dimension does not match its declared family and arm")
    model = make_model(torch, expected_dimension, configuration).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def infer_transfer(model: Any, payload: dict[str, Any], annual: np.ndarray, state: np.ndarray,
                   transform: str, device: str, budget: RuntimeBudget,
                   diagnostics: dict[str, Any] | None = None) -> np.ndarray:
    """Preserve the frozen transfer arithmetic in fixed 128-site chunks."""
    import torch
    outputs = []
    for start in range(0, len(state), 128):
        budget.check()
        stop = min(start + 128, len(state))
        transformed = transform_state(state[start:stop], transform)
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


def infer_auxiliary(model: Any, payload: dict[str, Any], state: np.ndarray, device: str,
                    budget: RuntimeBudget, diagnostics: dict[str, Any] | None = None) -> np.ndarray:
    """Preserve the frozen auxiliary arithmetic without altering the raw auxiliary channels."""
    import torch
    predictions = []
    for start in range(0, len(state), 128):
        budget.check()
        values = state[start:start + 128]
        vector = (feature_matrix(values, PRIMITIVES) - payload["feature_mean"]) / payload["feature_std"]
        if diagnostics is not None:
            diagnostics["clipped"] = diagnostics.get("clipped", 0) + int(np.count_nonzero(np.abs(vector) > 8))
            diagnostics["feature_values"] = diagnostics.get("feature_values", 0) + int(vector.size)
            diagnostics["clipped_fraction"] = diagnostics["clipped"] / diagnostics["feature_values"]
            _, shape_info = auxiliary_shape(values[:, :, 2:])
            counts = diagnostics.setdefault("degenerate_by_variable", {name: 0 for name in AUXILIARY_NAMES})
            for name in AUXILIARY_NAMES:
                counts[name] += shape_info["degenerate_by_variable"][name]
        vector = np.clip(vector, -8, 8)
        anchor = (anchor_at_40(values)[:, :, :2] / SCALES).reshape(-1, 2)
        predictions.append(predict_tensor(model, torch.from_numpy(vector), anchor, torch, device))
    return np.concatenate(predictions)


def predict_test(root: Path, checkpoint: Path, family: str, transform: str, device: str,
                 budget: RuntimeBudget, diagnostics: dict[str, Any] | None = None) -> np.ndarray:
    """Predict both public scenarios from their prepared arrays and a single checkpoint."""
    if transform not in TRANSFORMS:
        raise ValueError(f"Unsupported initial-state transform: {transform}")
    budget.check()
    model, payload = load_prediction_model(checkpoint, family, device)
    outputs = []
    for scenario in ("ssp126", "ssp585"):
        budget.check()
        state = np.load(root / "data" / f"{scenario}_test_y0.npy", mmap_mode="r")
        if state.shape != (6171, 15, 7):
            raise ValueError(f"Unexpected initial-state shape for {scenario}")
        if family == "transfer":
            annual = np.load(root / "data" / f"test_xann_{scenario}.npy", mmap_mode="r")
            if annual.shape != (6171, 40, 136):
                raise ValueError(f"Unexpected forcing shape for {scenario}")
            if transform == "community_affine":
                verify_test_encoding(state)
            outputs.append(infer_transfer(model, payload, annual, state, transform, device, budget, diagnostics))
        else:
            decoded = decode_initial_state(state, transform, PRIMITIVES)
            scenario_info: dict[str, Any] = {}
            outputs.append(infer_auxiliary(model, payload, decoded, device, budget, scenario_info))
            if diagnostics is not None:
                diagnostics[scenario] = scenario_info
    return np.concatenate(outputs)
