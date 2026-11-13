#!/usr/bin/env python3
"""Reproduce one frozen, equal-weight two-seed ensemble from its checkpoints.

This helper reads only checkpoints, public test inputs and the submission
template. It does not train, access training labels, refit a decoder, change
the frozen engines, or upload anything to Kaggle.

CPU is the default. CUDA requires the exact reserved CUDA_VISIBLE_DEVICES UUID
and acquires the frozen engine's existing mission lock. GPU inference explicitly
enables TF32 and retains the engines' 128-site chunks and 8192-row inner batches.

Use a new --output-file path so the reference submission remains untouched.
--expected-sha256 makes exact CSV identity a required gate. A mismatch preserves
the generated CSV, writes a .sha256_diff.json report, and exits with status 2.
An optional --reference-csv adds numerical difference diagnostics; its hash must
agree with --expected-sha256, or supplies the expected hash when omitted.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gc
import hashlib
import importlib.util
import itertools
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "4"

import numpy as np


TRANSFER_SHA256 = "230680807f1ec9a3ccd89c8739ac149284cb5c1b5eaa5b11707fcbd213fa2905"
AUXILIARY_SHA256 = "5684f74e7254d045249b68bcec4d0fa889caf4ad50867c7b48e690f16483350c"
GPU_UUID = "GPU-d60d12af-82b8-9913-7a5e-1177e774dafa"
SEEDS = (20261113, 20261114)
FAMILIES = ("transfer", "aux_shape")
TRANSFORMS = ("raw", "community_affine")


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sha256(value: str | None) -> str | None:
    """Validate a complete SHA-256 digest rather than accepting partial prefixes."""
    if value is None:
        return None
    value = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected SHA-256 must contain exactly 64 hexadecimal characters")
    return value


def locate_engine(explicit: Path | None, root: Path, filename: str) -> Path:
    """Locate an engine only in the declared project locations."""
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = (Path(__file__).resolve().parent / filename,
                  root / "research" / "transfer_20261113" / filename,
                  root / "src" / filename)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Cannot locate {filename}; specify its engine path explicitly")


def load_verified_engine(path: Path, expected_hash: str, name: str) -> Any:
    """Verify the exact frozen source bytes before importing the engine."""
    actual = sha256_file(path)
    if actual != expected_hash:
        raise RuntimeError(f"Frozen engine hash mismatch for {path}: {actual}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import frozen engine: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


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


def reproduce(root: Path, family: str, checkpoints: list[Path], transform: str, output_file: Path,
              device: str, engine: Any, auxiliary: Any = None, expected_sha256: str | None = None,
              reference_csv: Path | None = None) -> dict[str, Any]:
    """Run pinned inference, average in float64, write the exact engine CSV and apply the hash gate."""
    if family not in FAMILIES or transform not in TRANSFORMS or device not in ("cpu", "cuda:0"):
        raise ValueError("Unsupported family, initial-state transform or execution device")
    root, output_file = root.resolve(), output_file.resolve()
    if output_file.suffix.lower() != ".csv":
        raise ValueError("Output file must have a .csv extension")
    if output_file.exists():
        raise FileExistsError(f"Use a new output path; existing artifacts are preserved: {output_file}")
    if output_file.is_relative_to(root / "data"):
        raise ValueError("A reproduction CSV must not be written inside the input data directory")
    expected = normalize_sha256(expected_sha256)
    if reference_csv is not None:
        reference_csv = reference_csv.resolve()
        reference_hash = sha256_file(reference_csv)
        if reference_csv == output_file:
            raise ValueError("The reference CSV and output file must be different")
        if expected is not None and expected != reference_hash:
            raise ValueError("The reference CSV does not match the supplied expected SHA-256")
        expected = reference_hash
    if device == "cuda:0" and os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError(f"CUDA inference requires CUDA_VISIBLE_DEVICES exactly equal to {GPU_UUID}")
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    # The frozen models contain Linear/LayerNorm/GELU layers, not convolutions.
    # This explicit setting also preserves the training runtime's default.
    torch.backends.cudnn.allow_tf32 = True
    records, arm = checkpoint_records([path.resolve() for path in checkpoints], family, engine, auxiliary, torch)
    inputs = verify_test_inputs(root, family, records)
    report: dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "family": family, "arm": arm,
        "transform": transform, "device": device, "gpu_used": device == "cuda:0",
        "training_performed": False, "training_data_accessed": False, "kaggle_upload_performed": False,
        "helper_sha256": sha256_file(Path(__file__).resolve()),
        "transfer_engine_sha256": TRANSFER_SHA256,
        "auxiliary_engine_sha256": AUXILIARY_SHA256 if family == "aux_shape" else None,
        "environment": engine.environment_fingerprint(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        "inference_autocast": False, "sites_per_chunk": 128, "inner_row_batch_size": 8192,
        "ensemble": {"seeds": list(SEEDS), "weights": [0.5, 0.5],
                     "arithmetic": "np.mean(np.stack(predictions), axis=0, dtype=np.float64).astype(np.float32)"},
        "checkpoints": [{key: value for key, value in record.items() if key != "sources"} for record in records],
        "public_inputs": inputs, "expected_sha256": expected,
        "reference_csv": str(reference_csv) if reference_csv else None,
        "diagnostics": {},
    }
    context = engine.exclusive_gpu(root) if device == "cuda:0" else contextlib.nullcontext()
    values = []
    with context:
        if device == "cuda:0":
            engine.check_runtime()
            if torch.cuda.device_count() != 1:
                raise RuntimeError("Exactly one reserved CUDA device must be visible")
            report["gpu_name"] = torch.cuda.get_device_name(0)
        for record in records:
            checkpoint = Path(record["path"])
            if family == "transfer":
                diagnostics: dict[str, Any] = {}
                prediction = engine.predict_test(root, checkpoint, transform, device, diagnostics=diagnostics)
            else:
                prediction, diagnostics = auxiliary.test_predictions(root, checkpoint, transform, engine, device)
            if prediction.shape != (185130, 2) or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid predictions from seed {record['seed']}")
            if sha256_file(checkpoint) != record["sha256"]:
                raise RuntimeError("A checkpoint changed while inference was running")
            values.append(prediction)
            report["diagnostics"][str(record["seed"])] = diagnostics
            gc.collect()
        averaged = np.mean(np.stack(values), axis=0, dtype=np.float64).astype(np.float32)
        del values
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report["submission"] = engine.write_submission(root, output_file, averaged)
    actual = report["submission"]["sha256"]
    matched = None if expected is None else actual == expected
    report["sha256_gate"] = {"required": expected is not None, "passed": matched,
                             "expected": expected, "actual": actual}
    report["status"] = "GENERATED_UNCHECKED" if matched is None else ("PASS" if matched else "FAIL")
    report["exit_code"] = 2 if matched is False else 0
    if matched is False:
        difference: dict[str, Any] = {
            "status": "FAIL", "expected_sha256": expected, "actual_sha256": actual,
            "generated_csv": str(output_file), "generated_csv_preserved": output_file.is_file(),
            "reason": "Generated CSV bytes do not match the required reference hash",
            "device": device, "sites_per_chunk": 128, "tf32_matmul_allowed": True,
        }
        if reference_csv is None:
            difference["numeric_comparison"] = {"available": False, "reason": "Only a reference hash was provided"}
        else:
            try:
                difference["numeric_comparison"] = csv_difference(output_file, reference_csv)
            except (ValueError, OSError) as error:
                difference["numeric_comparison"] = {"available": False, "reason": str(error)}
        difference_path = output_file.with_suffix(".sha256_diff.json")
        engine.atomic_json(difference_path, difference)
        report["difference_report"] = str(difference_path)
    report["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    receipt_path = output_file.with_suffix(".reproduction.json")
    report["receipt_file"] = str(receipt_path)
    engine.atomic_json(receipt_path, report)
    return report


def main() -> int:
    """Parse the explicit inference request and return a machine-readable gate status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Project root containing the public data directory")
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=2, required=True, metavar=("SEED_ONE", "SEED_TWO"))
    parser.add_argument("--transform", choices=TRANSFORMS, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--transfer-engine", type=Path)
    parser.add_argument("--auxiliary-engine", type=Path)
    args = parser.parse_args()
    try:
        # Validate the CUDA boundary before importing torch or loading a checkpoint.
        if args.device == "cuda:0" and os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
            raise RuntimeError(f"CUDA inference requires CUDA_VISIBLE_DEVICES exactly equal to {GPU_UUID}")
        parent_path = locate_engine(args.transfer_engine, args.root, "transfer_experiment_20261113.py")
        engine = load_verified_engine(parent_path, TRANSFER_SHA256, "verified_transfer_reproduction_engine")
        auxiliary = None
        if args.family == "aux_shape":
            auxiliary_path = locate_engine(args.auxiliary_engine, args.root, "aux_shape_experiment_20261113.py")
            auxiliary = load_verified_engine(auxiliary_path, AUXILIARY_SHA256, "verified_auxiliary_reproduction_engine")
        if args.device == "cuda:0":
            signal.signal(signal.SIGTERM, engine.request_stop)
            signal.signal(signal.SIGINT, engine.request_stop)
        report = reproduce(args.root, args.family, args.checkpoints, args.transform, args.output_file,
                           args.device, engine, auxiliary, args.expected_sha256, args.reference_csv)
        print(json.dumps(report, indent=2, allow_nan=False))
        return int(report["exit_code"])
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "ERROR", "exit_code": 1, "error_type": type(error).__name__,
                          "error": str(error), "kaggle_upload_performed": False}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
