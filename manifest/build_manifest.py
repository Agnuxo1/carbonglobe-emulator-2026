"""Rebuild manifest/MANIFEST.json. Paths are always written with POSIX separators so
the hash check runs identically on Windows and on Linux CI."""
import glob
import hashlib
import json
import os

TRACKED = (sorted(glob.glob("src/*.py")) + sorted(glob.glob("tests/*.py")) +
           ["README.md", "TECHNICAL_REPORT.md", "ATTRIBUTION.md", "LICENSE", "requirements.txt",
            "reproduce.sh", "manifest/build_manifest.py"])

man = {
    "project": "carbonglobe-emulator-2026",
    "version": "1.0.1",
    "date": "2026-10-30",
    "competition": "ieee-bigdata-cup-2026-ai-emulation-challenge",
    "metric": "MSE on height/10.90645 and agb/3.39567, December of year 40",
    "best_public_score": 0.094,
    "best_public_system": "age-shifted anchor (no fitted parameters)",
    "h1_verdict": ("rejected: Holographic-Reservoir 2.58x the control MSE; "
                   "hybrid 1.24x worse than the control"),
    "note": ("These hashes certify packaging only. Every accuracy number in the README "
             "comes from the runs recorded in experiments/."),
    "files": {},
}
for f in TRACKED:
    key = f.replace(os.sep, "/")
    with open(f, "rb") as fh:
        man["files"][key] = hashlib.sha256(fh.read()).hexdigest()[:16]

with open("manifest/MANIFEST.json", "w", encoding="utf-8") as fh:
    json.dump(man, fh, indent=1)
print(f"{len(man['files'])} files hashed")
