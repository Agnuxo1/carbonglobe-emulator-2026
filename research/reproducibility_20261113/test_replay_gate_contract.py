"""Exercise replay-gate contracts with synthetic CSVs; no emulator or GPU runs."""
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import run_verified_replays as gate

FAKE_ENTRYPOINT = '''import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('data','bundle','output','device','execution-mode','cuda-index'):p.add_argument('--'+name)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True)
(out/'final').mkdir();(out/'dynamics_predictions').mkdir()
target=out/'final/S1109_restored_candidate.csv'
shutil.copyfile(Path(a.data)/'sample_submission.csv',target)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
receipt={'status':'complete','training_performed':False,'historical_predictions_loaded':False,
 'release_inventory_verification':{'manifest_sha256':sha(Path(a.bundle)/'release_manifest.json')},
 'output_sha256':sha(target)}
(out/'reproduction_receipt.json').write_text(json.dumps(receipt))
(out/'dynamics_predictions/dynamics_inference_manifest.json').write_text(json.dumps({'synthetic_fixture_only':True,'gpu_used':False}))
'''


def rejected(function):
    try:
        function()
    except (ValueError, RuntimeError):
        return True
    return False


def main():
    results = {}
    with tempfile.TemporaryDirectory(prefix="replay_contract_") as directory:
        root = Path(directory)
        bundle, data = root / "bundle", root / "data"
        bundle.mkdir(); data.mkdir()
        (bundle / "dependencies/lightgbm").mkdir(parents=True)
        (bundle / "dependencies/lightgbm/VERSION.txt").write_text("4.6.0\n")
        (bundle / "reproduce_champion.py").write_text(FAKE_ENTRYPOINT)
        inventory = {p.relative_to(bundle).as_posix(): {"bytes": p.stat().st_size, "sha256": gate.digest(p)}
                     for p in bundle.rglob("*") if p.is_file()}
        manifest = bundle / "release_manifest.json"
        manifest.write_text(json.dumps({"files": inventory}))
        manifest_sha = gate.digest(manifest)
        template = data / "sample_submission.csv"
        with template.open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(gate.EXPECTED_HEADER)
            writer.writerows((f"{scenario}_{site}_{age}", 0, 0)
                             for scenario in gate.SCENARIOS for site in range(gate.SITE_COUNT) for age in gate.AGES)
        for name in gate.INPUT_NAMES[1:]:
            (data / name).write_bytes(b"synthetic input fingerprint fixture; never loaded as an array")
        identifiers = gate.template_ids(template)
        results["complete_cartesian_id_product"] = len(identifiers) == 185130
        results["finite_nonnegative_schema"] = gate.inspect_prediction(template, identifiers)["nonnegative"]
        gate.probe_versions = lambda python, environment: {"synthetic_fixture_only": True, "gpu_initialized": False}
        argv = ["run_verified_replays.py", "--bundle", str(bundle), "--data", str(data),
                "--historical", str(template), "--expected-historical-sha256", gate.digest(template),
                "--expected-manifest-sha256", manifest_sha, "--python", sys.executable,
                "--output-root", str(root / "success"), "--timeout-seconds", "30"]
        previous_argv = sys.argv
        sys.argv = argv
        with contextlib.redirect_stdout(io.StringIO()):
            result = gate.main()
        success = json.loads((root / "success/verified_replays_receipt.json").read_text())
        results["two_synthetic_passes_and_saved_receipt"] = result == 0 and success["status"] == "PASS" and len(success["passes"]) == 2
        results["literal_replay_identity"] = success["replay"]["classification"] == "candidate_replay_byte_identical"
        results["historic_reference_separate"] = success["historical_comparison"]["exact_numeric_match"]
        results["six_manifest_boundaries"] = len(success["bundle_checks"]) == 6
        small = root / "small.csv"
        def write_small(rows, header=gate.EXPECTED_HEADER):
            with small.open("w", newline="") as stream:
                writer = csv.writer(stream);writer.writerow(header);writer.writerows(rows)
        ids = ["ssp126_0_1", "ssp126_0_10"]
        write_small([(ids[0], 0, 0), (ids[1], 0, 0)])
        baseline = root / "small_reference.csv";baseline.write_bytes(small.read_bytes())
        write_small([(ids[0], 0, 1), (ids[1], 0, 0)])
        comparison = gate.compare_historical(small, baseline, ids, 1e-6, gate.digest(baseline))
        results["different_historical_prediction_is_new_candidate"] = comparison["classification"] == "new_restored_candidate" and comparison["mean_squared_difference"] == 0.25
        write_small([(ids[0], float("nan"), 0), (ids[1], 0, 0)])
        results["nonfinite_rejected"] = rejected(lambda: gate.inspect_prediction(small, ids))
        write_small([(ids[0], -0.1, 0), (ids[1], 0, 0)])
        results["negative_rejected"] = rejected(lambda: gate.inspect_prediction(small, ids))
        write_small([(ids[1], 0, 0), (ids[0], 0, 0)])
        results["id_order_rejected"] = rejected(lambda: gate.inspect_prediction(small, ids))
        write_small([(ids[0], 0, 0)])
        results["row_count_rejected"] = rejected(lambda: gate.inspect_prediction(small, ids))
        write_small([(ids[0], 0, 0), (ids[1], 0, 0)], ["id", "agb", "height"])
        results["header_rejected"] = rejected(lambda: gate.inspect_prediction(small, ids))
        (bundle / "reproduce_champion.py").write_text(FAKE_ENTRYPOINT + "\n# changed\n")
        results["manifest_asset_tampering_rejected"] = rejected(lambda: gate.verify_bundle(bundle, manifest_sha))
        sys.argv = [str(root / "failure") if item == str(root / "success") else item for item in argv]
        with contextlib.redirect_stdout(io.StringIO()):
            result = gate.main()
        failure = json.loads((root / "failure/verified_replays_receipt.json").read_text())
        results["integrity_failure_saved_without_launching"] = result == 1 and failure["status"] == "FAIL" and not failure["passes"]
        for tolerance in ("nan", "inf"):
            sys.argv = argv + ["--historical-tolerance", tolerance, "--plan"]
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    gate.main()
                except SystemExit as error:
                    results[f"tolerance_{tolerance}_rejected"] = error.code == 2
        timeout = gate.execute_logged([sys.executable, "-c", "import time; time.sleep(5)"],
                                      dict(os.environ), root / "timeout.log", 0.1)
        results["bounded_child_timeout"] = timeout["status"] == "timeout" and timeout["returncode"] != 0
        link = root / "venv_python";link.symlink_to(sys.executable)
        sys.argv = argv + ["--python", str(link), "--plan"]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            gate.main()
        results["venv_interpreter_symlink_preserved"] = json.loads(stdout.getvalue())["commands"][0][0] == str(link)
        sys.argv = previous_argv
    assert all(results.values()), results
    report = {"status": "PASS", "test_kind": "synthetic_orchestration_contract", "gpu_used": False,
              "emulator_inference_performed": False, "checks": results,
              "wrapper_sha256": gate.digest(gate.__file__)}
    destination = Path(__file__).with_name("replay_gate_contract_smoke.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
