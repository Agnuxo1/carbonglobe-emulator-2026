"""Run two saved-weight inferences and certify the resulting reference CSV.

This operational wrapper is intended for the research service on Linux. It does
not train, submit to Kaggle, change services, or modify the frozen release. Each
inference child acquires the mission GPU lock itself; do not wrap this program
in another flock on that same lock.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path, PurePosixPath

EXPECTED_ROWS = 185130
EXPECTED_HEADER = ["id", "height", "agb"]
SCENARIOS = ("ssp126", "ssp585")
SITE_COUNT = 6171
AGES = (1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500)
FROZEN_MANIFEST_SHA256 = "7a646394e23d81377f4385fdbcec8d9382095902eb941a2062cb21159f71d14c"
HISTORICAL_SHA256 = "58e4123e893837374f81c3a7b14e0bd85c9e18fbe294c9f491ea24ef7271f15a"
INPUT_NAMES = (
    "sample_submission.csv", "ssp126_test_y0.npy", "ssp585_test_y0.npy",
    "test_xann_ssp126.npy", "test_xann_ssp585.npy",
)
ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED", "INVOCATION_ID",
)


def utc_now():
    """Return an unambiguous UTC timestamp for the execution receipt."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    """Hash a file without loading the full artifact into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Replace a JSON receipt atomically, preserving the last complete update."""
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def file_inventory(paths):
    """Fingerprint explicit inputs using their basename as the stable key."""
    result = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return result


def identical_bytes(first, second):
    """Compare file contents literally, in addition to their SHA-256 hashes."""
    with Path(first).open("rb") as left, Path(second).open("rb") as right:
        while True:
            left_chunk, right_chunk = left.read(1 << 20), right.read(1 << 20)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def verify_bundle(bundle, expected_manifest_sha):
    """Verify the trusted manifest and every package asset before execution."""
    bundle = Path(bundle).resolve()
    manifest_path = bundle / "release_manifest.json"
    actual_manifest_sha = digest(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(f"Frozen manifest changed: {actual_manifest_sha}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = manifest.get("files")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("The release manifest has no file inventory.")
    for name, metadata in inventory.items():
        if not isinstance(name, str) or chr(92) in name or ":" in name:
            raise ValueError(f"Unsafe release path: {name!r}")
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
            raise ValueError(f"Unsafe release path: {name!r}")
        if name == "release_manifest.json":
            raise ValueError("The release manifest cannot hash itself.")
        path = bundle.joinpath(*relative.parts)
        resolved = path.resolve()
        if bundle not in resolved.parents or not resolved.is_file():
            raise ValueError(f"Missing or escaping release asset: {name}")
        if not isinstance(metadata, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("sha256", ""))):
            raise ValueError(f"Malformed fingerprint for {name}")
        if path.stat().st_size != metadata.get("bytes") or digest(path) != metadata["sha256"]:
            raise ValueError(f"Release asset changed: {name}")
    actual_files = {
        path.relative_to(bundle).as_posix() for path in bundle.rglob("*")
        if path.is_file() and path != manifest_path
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if actual_files != set(inventory):
        raise ValueError(f"Release inventory mismatch: {sorted(actual_files.symmetric_difference(inventory))}")
    if "reproduce_champion.py" not in inventory:
        raise ValueError("The frozen inference entry point is missing from the inventory.")
    return {"status": "PASS", "manifest_sha256": actual_manifest_sha,
            "verified_files": len(inventory), "checked_at": utc_now()}


def template_ids(path):
    """Validate the official template and preserve its exact row order."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != EXPECTED_HEADER:
            raise ValueError("The official template header must be id,height,agb.")
        identifiers = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 3 or not row[0]:
                raise ValueError(f"Malformed template row at line {line_number}")
            parts = row[0].split("_")
            if len(parts) != 3 or parts[0] not in SCENARIOS:
                raise ValueError(f"Invalid scenario/site/age ID at template line {line_number}")
            try:
                site, age = int(parts[1]), int(parts[2])
            except ValueError as error:
                raise ValueError(f"Invalid numeric ID at template line {line_number}") from error
            if not 0 <= site < SITE_COUNT or age not in AGES \
                    or row[0] != f"{parts[0]}_{site}_{age}":
                raise ValueError(f"ID lies outside the official scenario/site/age product at line {line_number}")
            identifiers.append(row[0])
    if len(identifiers) != EXPECTED_ROWS or len(set(identifiers)) != EXPECTED_ROWS:
        raise ValueError(f"Template must contain {EXPECTED_ROWS} unique IDs; found {len(identifiers)} rows.")
    # Membership plus this unique cardinality proves the complete Cartesian product.
    return identifiers


def inspect_prediction(path, identifiers):
    """Require the official schema, finite targets, exact IDs and exact order."""
    count = 0
    minimum = [math.inf, math.inf]
    maximum = [-math.inf, -math.inf]
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != EXPECTED_HEADER:
            raise ValueError(f"Prediction header mismatch: {path}")
        for index, row in enumerate(reader):
            if index >= len(identifiers) or len(row) != 3:
                raise ValueError(f"Unexpected prediction row {index + 1}: {path}")
            if row[0] != identifiers[index]:
                raise ValueError(f"Prediction ID/order mismatch at row {index + 1}: {path}")
            for target, text in enumerate(row[1:]):
                value = float(text)
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite prediction at row {index + 1}, target {target}: {path}")
                if value < 0:
                    raise ValueError(f"Negative prediction at row {index + 1}, target {target}: {path}")
                minimum[target] = min(minimum[target], value)
                maximum[target] = max(maximum[target], value)
            count += 1
    if count != len(identifiers):
        raise ValueError(f"Prediction row count mismatch: {count} != {len(identifiers)}")
    return {"path": str(path), "header": EXPECTED_HEADER, "rows": count,
            "unique_ids": count, "official_id_order": True, "finite": True, "nonnegative": True,
            "minimum": dict(zip(EXPECTED_HEADER[1:], minimum)),
            "maximum": dict(zip(EXPECTED_HEADER[1:], maximum)),
            "bytes": Path(path).stat().st_size, "sha256": digest(path)}


def compare_historical(generated, historical, identifiers, tolerance, expected_sha):
    """Compare normalized predictions separately from the inference process."""
    reference_schema = inspect_prediction(historical, identifiers)
    if expected_sha and reference_schema["sha256"] != expected_sha:
        raise ValueError("The historical reference does not match its protected SHA-256.")
    squared = [[], []]
    maximum = [0.0, 0.0]
    with Path(generated).open("r", encoding="utf-8-sig", newline="") as left, \
            Path(historical).open("r", encoding="utf-8-sig", newline="") as right:
        actual, expected = csv.reader(left), csv.reader(right)
        next(actual)
        next(expected)
        for actual_row, expected_row in zip(actual, expected, strict=True):
            for target in range(2):
                difference = float(actual_row[target + 1]) - float(expected_row[target + 1])
                square = difference * difference
                if not math.isfinite(difference) or not math.isfinite(square):
                    raise ValueError("Historical difference metrics overflowed despite finite CSV values.")
                maximum[target] = max(maximum[target], abs(difference))
                squared[target].append(square)
    means = [math.fsum(values) / len(identifiers) for values in squared]
    overall_maximum = max(maximum)
    matches = overall_maximum <= tolerance
    return {
        "scope": "historical_prediction_comparison_only",
        "classification": "historical_numerical_reproduction" if matches else "new_restored_candidate",
        "reference": str(historical), "reference_sha256": reference_schema["sha256"],
        "generated_sha256": digest(generated), "units": "submission-normalized height and agb",
        "maximum_absolute_difference": overall_maximum,
        "mean_squared_difference": math.fsum(means) / 2,
        "by_target": {name: {"maximum_absolute_difference": maximum[index],
                              "mean_squared_difference": means[index]}
                      for index, name in enumerate(EXPECTED_HEADER[1:])},
        "tolerance": tolerance, "matches_tolerance": matches,
        "exact_numeric_match": overall_maximum == 0.0,
        "byte_identical": digest(generated) == reference_schema["sha256"],
        "note": "These are differences between predictions, not predictive MSE against ground truth.",
    }


def stop_process_tree(process):
    """Terminate only this wrapper's timed-out child and its descendants."""
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    process.wait(timeout=10)


def execute_logged(command, environment, log_path, timeout_seconds):
    """Run one bounded inference with a persistent combined stdout/stderr log."""
    started = time.monotonic()
    record = {"command": command, "started_at": utc_now(), "log": str(log_path),
              "timeout_seconds": timeout_seconds, "status": "running"}
    with Path(log_path).open("w", encoding="utf-8") as log:
        options = {"start_new_session": True} if os.name == "posix" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, **options)
        record["pid"] = process.pid
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            stop_process_tree(process)
            record.update(status="timeout", returncode=process.returncode)
        except BaseException:
            stop_process_tree(process)
            raise
        else:
            record.update(status="complete" if return_code == 0 else "failed", returncode=return_code)
    record.update(seconds=time.monotonic() - started, ended_at=utc_now(), log_sha256=digest(log_path))
    return record


def probe_versions(python, environment):
    """Inspect the chosen interpreter without initializing a GPU context."""
    code = (
        "import importlib.metadata as m,json,platform,sys; import torch,numpy; "
        "torch.set_num_threads(2); "
        "names=('numpy','pandas','scipy','scikit-learn','lightgbm'); "
        "print(json.dumps({'python':sys.version,'executable':sys.executable,"
        "'platform':platform.platform(),'torch':torch.__version__,"
        "'cuda_build':torch.version.cuda,'gpu_initialized':torch.cuda.is_initialized(),"
        "'packages':{name:m.version(name) for name in names}}))"
    )
    safe_environment = dict(environment)
    safe_environment["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.run([str(python), "-c", code], capture_output=True, text=True,
                             env=safe_environment, timeout=120, check=True)
    result = json.loads(process.stdout)
    if result["gpu_initialized"]:
        raise RuntimeError("The CPU version probe unexpectedly initialized CUDA.")
    return result


def build_command(args, output):
    """Construct an argument list without invoking a shell."""
    return [str(args.python), str(args.bundle / "reproduce_champion.py"),
            "--data", str(args.data), "--bundle", str(args.bundle), "--output", str(output),
            "--device", "cuda", "--execution-mode", "mission", "--cuda-index", "0"]


def main():
    """Execute the two-pass gate and leave a receipt on success or failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="New empty directory containing run/, replay/, logs and receipts.")
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--expected-manifest-sha256", default=FROZEN_MANIFEST_SHA256)
    parser.add_argument("--expected-historical-sha256", default=HISTORICAL_SHA256)
    parser.add_argument("--historical-tolerance", type=float, default=1e-6)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--plan", action="store_true", help="Print commands without creating files or running inference.")
    args = parser.parse_args()
    for name in ("bundle", "data", "output_root", "historical"):
        setattr(args, name, getattr(args, name).resolve())
    # Following a venv interpreter symlink would silently select the base Python.
    args.python = Path(os.path.abspath(args.python))
    if args.timeout_seconds <= 0 or not math.isfinite(args.historical_tolerance) \
            or args.historical_tolerance < 0:
        parser.error("Timeout must be positive and tolerance finite and nonnegative.")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_manifest_sha256):
        parser.error("The expected manifest SHA-256 must contain 64 lowercase hexadecimal digits.")
    outputs = [args.output_root / "run", args.output_root / "replay"]
    commands = [build_command(args, output) for output in outputs]
    if args.plan:
        print(json.dumps({"status": "plan_only", "commands": commands,
                          "timeout_seconds_per_pass": args.timeout_seconds,
                          "expected_manifest_sha256": args.expected_manifest_sha256}, indent=2))
        return 0
    if args.output_root == args.bundle or args.bundle in args.output_root.parents:
        parser.error("Outputs must be outside the frozen release.")
    if args.output_root == args.data or args.data in args.output_root.parents:
        parser.error("Outputs must be outside the source dataset.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error("Use a new empty output root; previous runs are never overwritten.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise RuntimeError(f"Execution interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    receipt_path = args.output_root / "verified_replays_receipt.json"
    environment = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "2"
    started = time.monotonic()
    receipt = {
        "status": "running", "started_at": utc_now(), "wrapper_sha256": digest(__file__),
        "bundle": str(args.bundle), "data": str(args.data), "output_root": str(args.output_root),
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "historical_predictions_passed_to_inference": False, "training_requested": False,
        "environment": {key: environment.get(key) for key in ENVIRONMENT_KEYS},
        "wrapper_python": sys.version, "wrapper_platform": platform.platform(),
        "commands": commands, "passes": [], "bundle_checks": [],
    }
    write_json(receipt_path, receipt)
    try:
        receipt["bundle_checks"].append(verify_bundle(args.bundle, args.expected_manifest_sha256))
        identifiers = template_ids(args.data / "sample_submission.csv")
        receipt["template_contract"] = {
            "scenarios": list(SCENARIOS), "sites_per_scenario": SITE_COUNT, "ages": list(AGES),
            "rows": EXPECTED_ROWS, "exact_cartesian_product": True,
            "official_template_order_used": True,
            "canonical_nested_order": all(identifier == f"{scenario}_{site}_{age}"
                                          for identifier, (scenario, site, age) in zip(
                                              identifiers,
                                              ((scenario, site, age) for scenario in SCENARIOS
                                               for site in range(SITE_COUNT) for age in AGES), strict=True)),
        }
        data_inventory = file_inventory(args.data / name for name in INPUT_NAMES)
        receipt["test_inputs"] = data_inventory
        receipt["versions"] = probe_versions(args.python, environment)
        receipt["bundled_lightgbm_version"] = (args.bundle / "dependencies/lightgbm/VERSION.txt").read_text().strip()
        write_json(receipt_path, receipt)
        predictions = []
        for index, (output, command) in enumerate(zip(outputs, commands, strict=True)):
            name = "run" if index == 0 else "replay"
            receipt["bundle_checks"].append(verify_bundle(args.bundle, args.expected_manifest_sha256))
            if file_inventory(args.data / item for item in INPUT_NAMES) != data_inventory:
                raise ValueError(f"Test inputs changed before {name}.")
            print(f"Starting {name}; timeout {args.timeout_seconds} seconds", flush=True)
            record = execute_logged(command, environment, args.output_root / f"{name}.log", args.timeout_seconds)
            record["name"] = name
            receipt["passes"].append(record)
            write_json(receipt_path, receipt)
            if record["status"] != "complete":
                raise RuntimeError(f"Inference {name} failed: {record['status']}, returncode={record['returncode']}")
            receipt["bundle_checks"].append(verify_bundle(args.bundle, args.expected_manifest_sha256))
            if file_inventory(args.data / item for item in INPUT_NAMES) != data_inventory:
                raise ValueError(f"Test inputs changed during {name}.")
            child_receipt_path = output / "reproduction_receipt.json"
            child = json.loads(child_receipt_path.read_text(encoding="utf-8"))
            if child.get("status") != "complete" or child.get("training_performed") is not False \
                    or child.get("historical_predictions_loaded") is not False:
                raise ValueError(f"Invalid saved-weight inference receipt for {name}.")
            if child.get("release_inventory_verification", {}).get("manifest_sha256") != args.expected_manifest_sha256:
                raise ValueError(f"Child {name} did not verify the frozen manifest.")
            prediction = output / "final/S1109_restored_candidate.csv"
            schema = inspect_prediction(prediction, identifiers)
            if child.get("output_sha256") != schema["sha256"]:
                raise ValueError(f"Child {name} output fingerprint does not match its CSV.")
            record["prediction"] = schema
            record["child_receipt_sha256"] = digest(child_receipt_path)
            dynamic_manifest = output / "dynamics_predictions/dynamics_inference_manifest.json"
            record["dynamics_runtime"] = json.loads(dynamic_manifest.read_text(encoding="utf-8"))
            predictions.append(prediction)
            write_json(receipt_path, receipt)
        first, second = (record["prediction"] for record in receipt["passes"])
        if first["sha256"] != second["sha256"] or first["bytes"] != second["bytes"] \
                or not identical_bytes(predictions[0], predictions[1]):
            raise ValueError("The two independent saved-weight CSVs are not byte identical.")
        receipt["replay"] = {"status": "PASS", "classification": "candidate_replay_byte_identical",
                             "sha256": first["sha256"], "rows": EXPECTED_ROWS,
                             "maximum_absolute_difference": 0.0, "mean_squared_difference": 0.0}
        receipt["historical_comparison"] = compare_historical(
            predictions[0], args.historical, identifiers, args.historical_tolerance,
            args.expected_historical_sha256)
        write_json(args.output_root / "historical_comparison.json", receipt["historical_comparison"])
        receipt["bundle_checks"].append(verify_bundle(args.bundle, args.expected_manifest_sha256))
        receipt["status"] = "PASS"
        receipt["classification"] = receipt["historical_comparison"]["classification"]
    except Exception as error:
        receipt["status"] = "FAIL"
        receipt["error"] = {"type": type(error).__name__, "message": str(error),
                            "traceback": traceback.format_exc()}
    finally:
        receipt["ended_at"] = utc_now()
        receipt["seconds"] = time.monotonic() - started
        write_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path),
                      "classification": receipt.get("classification"), "seconds": receipt["seconds"]}), flush=True)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
