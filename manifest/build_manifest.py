"""Rebuild the packaging manifest from Git-tracked, non-ignored repository files.

The default scope matches the historical builder. Use --include-research-docs to
also hash tracked text/code artifacts under research/ and docs/. Local caches,
ignored private files, and untracked work are never discovered by filesystem
globs. Hashes use working-tree bytes exactly as stored; no line-ending or text
normalization is performed. This script does not stage or commit any files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "manifest/MANIFEST.json"
CORE_FILES = frozenset({
    "README.md", "TECHNICAL_REPORT.md", "ATTRIBUTION.md", "LICENSE",
    "requirements.txt", "reproduce.sh", "manifest/build_manifest.py",
})
CODE_DIRECTORIES = frozenset({"src", "tests", "servidor"})
OPTIONAL_DIRECTORIES = frozenset({"research", "docs"})
OPTIONAL_SUFFIXES = frozenset({".md", ".rst", ".txt", ".py", ".sh", ".json", ".yaml", ".yml"})


def git_paths(root: Path, *options: str) -> set[str]:
    """Read NUL-separated index paths without modifying the index or repository."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", *options, "--"],
        capture_output=True, check=True,
    )
    return {os.fsdecode(value) for value in result.stdout.split(b"\0") if value}


def in_scope(name: str, include_research_docs: bool) -> bool:
    """Apply the explicit source/document scope, always excluding the manifest itself."""
    path = PurePosixPath(name)
    if name == OUTPUT:
        return False
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unexpected repository-relative Git path: {name!r}")
    if name in CORE_FILES:
        return True
    if len(path.parts) == 2 and path.parts[0] in CODE_DIRECTORIES and path.suffix == ".py":
        return True
    return bool(include_research_docs and len(path.parts) >= 2
                and path.parts[0] in OPTIONAL_DIRECTORIES
                and path.suffix.lower() in OPTIONAL_SUFFIXES)


def file_hash(root: Path, name: str) -> str:
    """Hash raw bytes of a regular repository file without following symlinks."""
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Refusing to hash a symbolic link: {name}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Expected a regular file inside the repository: {name}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def build_manifest(root: Path, include_research_docs: bool = False) -> dict:
    """Build the CI-compatible file map and explicitly historical score metadata."""
    root = root.resolve()
    tracked = git_paths(root)
    ignored = git_paths(root, "--ignored", "--exclude-standard")
    eligible = tracked - ignored
    missing = sorted(CORE_FILES - eligible)
    if missing:
        raise RuntimeError(f"Required core files must be tracked and non-ignored: {missing}")
    names = sorted(name for name in eligible if in_scope(name, include_research_docs))
    return {
        "project": "carbonglobe-emulator-2026",
        "version": "1.1.0",
        "date": "2026-11-13",
        "competition": "ieee-bigdata-cup-2026-ai-emulation-challenge",
        "metric": "MSE on height/10.90645 and agb/3.39567, December of year 40",
        "best_public_score": 0.060,
        "best_public_system": "S1109.csv (selected historical submission)",
        "score_scope": "Historical selected artifact; not a score certified for reconstructed model weights",
        "selected_historical_submission": {
            "file": "S1109.csv",
            "kaggle_submission_id": 58145080,
            "public_score": 0.060,
            "model_reproduction_status": "pending",
            "reproduction_note": "End-to-end historical identity from model weights has not been established",
        },
        "h1_verdict": ("rejected: Holographic-Reservoir 2.58x the control MSE; "
                       "hybrid 1.24x worse than the control"),
        "h1_verdict_date": "2026-10-30",
        "h1_verdict_scope": "Historical verdict retained from the 2026-10-30 manifest; not a new validation",
        "note": ("These hashes certify packaging only, not accuracy or complete model reproduction. "
                 "Score evidence and experiment qualifications are recorded in experiments/ and research/."),
        "file_inventory": {
            "source": "git ls-files -z --cached, excluding paths ignored by Git",
            "scope": "Historical core scope; optionally tracked research/docs text and code",
            "research_docs_included": include_research_docs,
            "hash": "First 16 hexadecimal characters of SHA-256 over exact working-tree bytes",
            "manifest_self_excluded": True,
        },
        "files": {name: file_hash(root, name) for name in names},
    }


def main() -> None:
    """Write only MANIFEST.json, leaving Git staging and commits to the caller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-research-docs", action="store_true",
                        help="Also hash already-tracked, non-ignored text/code under research/ and docs/.")
    args = parser.parse_args()
    manifest = build_manifest(ROOT, args.include_research_docs)
    destination = ROOT / OUTPUT
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=destination.parent, prefix="MANIFEST.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=1, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"{len(manifest['files'])} tracked files hashed")


if __name__ == "__main__":
    main()
