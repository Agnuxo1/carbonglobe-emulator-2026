#!/usr/bin/env python3
"""Generate the final two-seed CarbonGlobe submission without training.

This portable entrypoint uses a separately fingerprinted inference module.
It does not import the frozen training engines or enforce their historical
competition deadline. It reproduces their feature arithmetic, state decoder,
128-site chunks, TF32 setting and float64-to-float32 ensemble average.

Portable mode defaults to CPU and can use an organizer-selected CUDA device.
When running on the reserved research GPU, select mission mode explicitly;
that mode enforces the exact UUID and the existing mission lock.

Input layout: ROOT/data/{ssp126,ssp585}_test_y0.npy, sample_submission.csv,
and, for the transfer family, test_xann_{ssp126,ssp585}.npy. No training arrays
are needed. Use a fresh output path. An expected hash or a reference CSV
turns exact CSV identity into a gate; failure preserves the generated CSV,
writes a difference report and returns exit status 2.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

MODULE_SHA256 = "8d03ba50841828ca27e20661b9a02aa6aaeb034409defd6db3245a03f4b6fe5f"


def load_inference_module() -> Any:
    """Verify the standalone inference module before executing its code."""
    path = Path(__file__).resolve().with_name("carbon_inference.py")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != MODULE_SHA256:
        raise RuntimeError(f"Portable inference module hash mismatch: {actual}")
    specification = importlib.util.spec_from_file_location("verified_portable_carbon_inference", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import inference module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def run(args: argparse.Namespace, inference: Any) -> dict[str, Any]:
    """Verify inputs, reproduce the declared ensemble and persist its hash-gate result."""
    root, output = args.root.resolve(), args.output_file.resolve()
    if output.suffix.lower() != ".csv" or output.is_relative_to(root / "data"):
        raise ValueError("Output must be a CSV outside the input data directory")
    if output.exists():
        raise FileExistsError(f"Use a fresh output path; existing artifacts are preserved: {output}")
    expected = inference.normalize_sha256(args.expected_sha256)
    reference = args.reference_csv.resolve() if args.reference_csv else None
    if reference is not None:
        if reference == output:
            raise ValueError("Reference and output CSV paths must differ")
        reference_hash = inference.sha256_file(reference)
        if expected is not None and expected != reference_hash:
            raise ValueError("Reference CSV differs from the supplied expected SHA-256")
        expected = reference_hash
    if args.mode == "mission" and args.device != "cpu":
        if args.device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != inference.MISSION_GPU_UUID:
            raise RuntimeError("Mission CUDA inference requires the reserved UUID and logical cuda:0")
    budget = inference.RuntimeBudget(args.max_runtime_seconds)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    records, arm = inference.checkpoint_records(
        [path.resolve() for path in args.checkpoints], args.family,
        inference.CHECKPOINT_CONTRACT, inference.CHECKPOINT_CONTRACT, torch)
    inputs = inference.verify_test_inputs(root, args.family, records)
    report: dict[str, Any] = {
        "created_utc": inference.utc_now(), "family": args.family, "arm": arm,
        "transform": args.transform, "mode": args.mode, "device": args.device,
        "entrypoint_sha256": inference.sha256_file(Path(__file__).resolve()),
        "inference_module_sha256": MODULE_SHA256,
        "training_source_sha256": inference.TRANSFER_SHA256,
        "auxiliary_training_source_sha256": inference.AUXILIARY_SHA256 if args.family == "aux_shape" else None,
        "training_performed": False, "training_data_accessed": False,
        "frozen_training_engines_imported": False, "calendar_deadline_enforced": False,
        "kaggle_upload_performed": False,
        "runtime_budget_seconds": args.max_runtime_seconds,
        "environment": inference.environment_fingerprint(),
        "gpu_used": args.device != "cpu",
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        "inference_autocast": False, "sites_per_chunk": 128, "inner_row_batch_size": 8192,
        "ensemble": {"seeds": list(inference.SEEDS), "weights": [0.5, 0.5],
                     "arithmetic": "float64 mean followed by float32 conversion"},
        "checkpoints": [{key: value for key, value in record.items() if key != "sources"} for record in records],
        "public_inputs": inputs, "expected_sha256": expected,
        "reference_csv": str(reference) if reference else None, "diagnostics": {},
    }
    predictions = []
    with inference.execution_context(root, args.mode, args.device, budget):
        if args.device != "cpu":
            report["gpu_name"] = torch.cuda.get_device_name(int(args.device.split(":", 1)[1]))
        for record in records:
            budget.check()
            checkpoint = Path(record["path"])
            diagnostics: dict[str, Any] = {}
            prediction = inference.predict_test(root, checkpoint, args.family, args.transform,
                                                args.device, budget, diagnostics)
            if prediction.shape != (185130, 2) or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid predictions from seed {record['seed']}")
            if inference.sha256_file(checkpoint) != record["sha256"]:
                raise RuntimeError("A checkpoint changed while inference was running")
            predictions.append(prediction)
            report["diagnostics"][str(record["seed"])] = diagnostics
            gc.collect()
        budget.check()
        average = np.mean(np.stack(predictions), axis=0, dtype=np.float64).astype(np.float32)
        del predictions
        output.parent.mkdir(parents=True, exist_ok=True)
        report["submission"] = inference.write_submission(root, output, average)
    actual = report["submission"]["sha256"]
    matched = None if expected is None else actual == expected
    report["sha256_gate"] = {"required": expected is not None, "passed": matched,
                             "expected": expected, "actual": actual}
    report["status"] = "GENERATED_UNCHECKED" if matched is None else ("PASS" if matched else "FAIL")
    report["exit_code"] = 2 if matched is False else 0
    if matched is False:
        difference: dict[str, Any] = {
            "status": "FAIL", "expected_sha256": expected, "actual_sha256": actual,
            "generated_csv": str(output), "generated_csv_preserved": output.is_file(),
            "device": args.device, "sites_per_chunk": 128, "tf32_matmul_allowed": True,
            "reason": "Generated CSV bytes do not match the required reference hash",
        }
        if reference is None:
            difference["numeric_comparison"] = {"available": False, "reason": "Only a reference hash was provided"}
        else:
            try:
                difference["numeric_comparison"] = inference.csv_difference(output, reference)
            except (ValueError, OSError) as error:
                difference["numeric_comparison"] = {"available": False, "reason": str(error)}
        difference_path = output.with_suffix(".sha256_diff.json")
        inference.atomic_json(difference_path, difference)
        report["difference_report"] = str(difference_path)
    report["completed_utc"] = inference.utc_now()
    receipt = output.with_suffix(".reproduction.json")
    report["receipt_file"] = str(receipt)
    inference.atomic_json(receipt, report)
    return report


def main() -> int:
    """Execute explicit inference and expose a process exit code suitable for a release gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=("transfer", "aux_shape"), required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=2, required=True, metavar=("SEED_ONE", "SEED_TWO"))
    parser.add_argument("--transform", choices=("raw", "community_affine"), required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--mode", choices=("portable", "mission"), default="portable")
    parser.add_argument("--device", default="cpu", help="cpu or a CUDA device such as cuda:0")
    parser.add_argument("--max-runtime-seconds", type=float, default=7200.0)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--reference-csv", type=Path)
    args = parser.parse_args()
    try:
        inference = load_inference_module()
        report = run(args, inference)
        print(json.dumps(report, indent=2, allow_nan=False))
        return int(report["exit_code"])
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "ERROR", "exit_code": 1, "error_type": type(error).__name__,
                          "error": str(error), "kaggle_upload_performed": False}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

