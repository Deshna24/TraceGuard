"""Integrity verification for TRACEGUARD-X protected research artifacts.

Verifies SHA-256 hashes against protected_hashes_phase0.json.
Must be run after every phase to ensure no frozen artifacts were modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from traceguard_x.paths import INTEGRITY_DIR, REPO_ROOT


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file in 1MB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protected_artifacts() -> dict[str, Any]:
    """Verify all protected artifacts recorded in phase0.
    
    Returns a dict with:
        'ok': bool (True if all match, False otherwise)
        'total': int
        'checked': int
        'matches': int
        'discrepancies': list of dicts describing mismatches
        'missing': list of missing paths
    """
    record_path = INTEGRITY_DIR / "protected_hashes_phase0.json"
    if not record_path.exists():
        raise FileNotFoundError(f"Integrity record missing: {record_path}")

    with record_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    hashes: dict[str, str] = data.get("sha256", {})
    discrepancies: list[dict[str, str]] = []
    missing: list[str] = []
    matches = 0

    for rel_path_str, expected_hash in hashes.items():
        norm_path = Path(rel_path_str.replace("\\", "/"))
        full_path = REPO_ROOT / norm_path
        if not full_path.exists():
            missing.append(str(norm_path))
            continue

        actual_hash = compute_sha256(full_path)
        if actual_hash != expected_hash:
            discrepancies.append({
                "path": str(norm_path),
                "expected": expected_hash,
                "actual": actual_hash,
            })
        else:
            matches += 1

    ok = (len(discrepancies) == 0 and len(missing) == 0)
    return {
        "ok": ok,
        "total": len(hashes),
        "checked": matches + len(discrepancies),
        "matches": matches,
        "discrepancies": discrepancies,
        "missing": missing,
    }


def main() -> int:
    result = verify_protected_artifacts()
    print(f"[Integrity] Total protected: {result['total']}, Matched: {result['matches']}")
    if result["missing"]:
        print(f"[Integrity] MISSING files ({len(result['missing'])}):")
        for m in result["missing"]:
            print(f"  - {m}")
    if result["discrepancies"]:
        print(f"[Integrity] DISCREPANCIES ({len(result['discrepancies'])}):")
        for d in result["discrepancies"]:
            print(f"  - {d['path']}: expected {d['expected'][:8]}... got {d['actual'][:8]}...")
    if result["ok"]:
        print("[Integrity] ALL PROTECTED ARTIFACTS INTACT (SHA-256 verified).")
        return 0
    else:
        print("[Integrity] VERIFICATION FAILED!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
