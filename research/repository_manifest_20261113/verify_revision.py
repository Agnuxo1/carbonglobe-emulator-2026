"""Verify the manifest revision in an isolated scratch Git fixture, without commits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parents[2] / "manifest" / "build_manifest.py"


def run(command: list[str], root: Path, expect_success: bool = True) -> subprocess.CompletedProcess:
    """Capture a fixture-only command and assert its expected outcome."""
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if (result.returncode == 0) != expect_success:
        raise AssertionError({"command": command, "returncode": result.returncode,
                              "stdout": result.stdout, "stderr": result.stderr})
    return result


def main() -> None:
    """Check tracked scope, private exclusions, exact bytes, and atomic failures."""
    with tempfile.TemporaryDirectory(prefix="manifest-git-fixture-", dir=HERE) as temporary:
        root = Path(temporary)
        run(["git", "init", "-q"], root)
        files = {
            "README.md": b"Public README\r\n",
            "TECHNICAL_REPORT.md": b"Report\n",
            "ATTRIBUTION.md": b"Attribution\n",
            "LICENSE": b"License\n",
            "requirements.txt": b"numpy\n",
            "reproduce.sh": b"#!/bin/sh\n",
            ".gitattributes": b"* -text\n",
            ".gitignore": b"src/private.py\nresearch/private/\n*.npy\n",
            "manifest/MANIFEST.json": b"{}\n",
            "src/public.py": b"VALUE = 1\r\n",
            "tests/test_public.py": b"assert True\n",
            "servidor/public.py": b"VALUE = 2\n",
            "src/nested/outside_scope.py": b"VALUE = 3\n",
            "research/receipt.json": b'{"public": true}\n',
            "research/notes with spaces \u03a9.md": "Public notes.\n".encode("utf-8"),
            "docs/guide.md": b"Public guide\n",
            "research/raw.bin": b"Not a text/code artifact\n",
        }
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        shutil.copyfile(SOURCE, root / "manifest" / "build_manifest.py")
        run(["git", "add", "--all"], root)
        # These are synthetic sentinels, not credentials or real user data.
        for name in ("src/private.py", "research/private/credentials.json", "research/data.npy"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"SYNTHETIC_PRIVATE_SENTINEL\n")
            run(["git", "add", "--force", "--", name], root)
        (root / "src" / "untracked.py").write_bytes(b"UNTRACKED_SENTINEL\n")
        (root / "research" / "untracked.json").write_bytes(b'{"untracked": true}\n')
        index_before = (root / ".git" / "index").read_bytes()
        attributes_before = (root / ".gitattributes").read_bytes()
        command = [sys.executable, "-B", str(root / "manifest" / "build_manifest.py")]
        run(command, root)
        default = json.loads((root / "manifest" / "MANIFEST.json").read_text())
        expected_core = {"README.md", "TECHNICAL_REPORT.md", "ATTRIBUTION.md", "LICENSE",
                         "requirements.txt", "reproduce.sh", "manifest/build_manifest.py",
                         "src/public.py", "tests/test_public.py", "servidor/public.py"}
        assert set(default["files"]) == expected_core
        exact = hashlib.sha256(files["src/public.py"]).hexdigest()[:16]
        normalized = hashlib.sha256(files["src/public.py"].replace(b"\r\n", b"\n")).hexdigest()[:16]
        assert default["files"]["src/public.py"] == exact and exact != normalized
        run(command + ["--include-research-docs"], root)
        destination = root / "manifest" / "MANIFEST.json"
        expanded = json.loads(destination.read_text())
        expected_extra = {"research/receipt.json", "research/notes with spaces \u03a9.md", "docs/guide.md"}
        assert set(expanded["files"]) == expected_core | expected_extra
        assert expanded["selected_historical_submission"]["kaggle_submission_id"] == 58145080
        assert expanded["selected_historical_submission"]["model_reproduction_status"] == "pending"
        assert expanded["date"] == "2026-11-13" and expanded["version"] == "1.1.0"
        assert expanded["h1_verdict_date"] == "2026-10-30"
        # This is the same file-hash condition as the published CI job.
        assert all(hashlib.sha256((root / name).read_bytes()).hexdigest()[:16] == digest
                   for name, digest in expanded["files"].items())
        first_bytes = destination.read_bytes()
        run(command + ["--include-research-docs"], root)
        assert destination.read_bytes() == first_bytes
        assert (root / ".git" / "index").read_bytes() == index_before
        assert (root / ".gitattributes").read_bytes() == attributes_before
        missing = root / "src" / "public.py"
        saved = missing.read_bytes()
        missing.unlink()
        run(command, root, expect_success=False)
        assert destination.read_bytes() == first_bytes
        missing.write_bytes(saved)
        external = root.parent / (root.name + "-outside.py")
        external.write_bytes(b"SYNTHETIC_EXTERNAL_SENTINEL\n")
        try:
            link = root / "src" / "external.py"
            link.symlink_to(external)
            run(["git", "add", "--", "src/external.py"], root)
            run(command, root, expect_success=False)
            assert destination.read_bytes() == first_bytes
        finally:
            external.unlink()
    receipt = {
        "passed": True,
        "scope": "Isolated scratch fixture only; no Windows or remote Git operations",
        "commits_created": 0,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "default_fixture_files": len(expected_core),
        "expanded_fixture_files": len(expected_core | expected_extra),
        "checks": ["untracked code and JSON excluded", "tracked ignored private sentinels excluded",
                   "historical default scope preserved", "optional tracked research/docs scope",
                   "NUL-separated Unicode and space-containing paths", "manifest self-exclusion",
                   "raw CRLF bytes preserved", "published CI hash condition",
                   "deterministic output", "builder leaves index and .gitattributes unchanged",
                   "missing tracked file fails before replacing the manifest",
                   "external symlink rejected before replacing the manifest"],
    }
    (HERE / "verification_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
