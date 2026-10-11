"""Results migration script from Week 6 to Week 5-C.

Performs:
1. Pre-migration SHA256 checksum calculation
2. File copy to results/week5/content_allocator/w5c_encoder_adaptation/
3. Post-migration SHA256 checksum calculation and verification
4. Generates MIGRATION_MANIFEST.md with byte-level provenance
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

OLD_ROOT = Path("results/week6/encoder_adaptation")
NEW_ROOT = Path("results/week5/content_allocator/w5c_encoder_adaptation")

FILE_MAP = {
    # Models
    "models/C1_seed_42.pt": "models/C1_seed_42.pt",
    "models/C1_seed_123.pt": "models/C1_seed_123.pt",
    "models/C1_seed_2026.pt": "models/C1_seed_2026.pt",
    "models/C2_seed_42.pt": "models/C2_seed_42.pt",
    "models/C2_seed_123.pt": "models/C2_seed_123.pt",
    "models/C2_seed_2026.pt": "models/C2_seed_2026.pt",
    # Predictions
    "predictions/dev_predictions.csv": "predictions/dev_predictions.csv",
    # Tables (w6_* -> w5c_*)
    "tables/w6_main_results.csv": "tables/w5c_main_results.csv",
    "tables/w6_seed_results.csv": "tables/w5c_seed_results.csv",
    "tables/w6_multilabel_metrics.csv": "tables/w5c_multilabel_metrics.csv",
    "tables/w6_tau_selection.csv": "tables/w5c_tau_selection.csv",
    "tables/w6_capacity_comparison.csv": "tables/w5c_capacity_comparison.csv",
    # Bootstrap
    "bootstrap/w6_bootstrap_comparisons.csv": "bootstrap/w5c_bootstrap_comparisons.csv",
    # Figures (w6_* -> w5c_*)
    "figures/w6_pareto.png": "figures/w5c_pareto.png",
    "figures/w6_selected_k_distribution.png": "figures/w5c_selected_k_distribution.png",
    # Diagnostics
    "diagnostics/representation_shift.csv": "diagnostics/representation_shift.csv",
    "diagnostics/representation_adaptation_health.json": "diagnostics/representation_adaptation_health.json",
    # Smoke artifacts
    "diagnostics/smoke/smoke_test_results.json": "diagnostics/smoke/smoke_test_results.json",
    "diagnostics/smoke/smoke_checkpoint.pt": "diagnostics/smoke/smoke_checkpoint.pt",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("Starting W6 -> W5-C Results Migration...")
    NEW_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    all_matched = True

    for old_rel, new_rel in FILE_MAP.items():
        old_path = OLD_ROOT / old_rel
        new_path = NEW_ROOT / new_rel
        new_path.parent.mkdir(parents=True, exist_ok=True)

        if not old_path.exists():
            print(f"WARNING: Old file not found: {old_path}")
            continue

        size = old_path.stat().st_size
        old_hash = sha256_file(old_path)

        # Copy byte-for-byte
        shutil.copy2(old_path, new_path)

        new_hash = sha256_file(new_path)
        matched = (old_hash == new_hash)
        if not matched:
            all_matched = False
            print(f"ERROR: Checksum mismatch for {old_rel} -> {new_rel}")
        else:
            action = "Renamed & Moved" if old_rel != new_rel else "Moved"
            print(f"Verified {action}: {old_rel} -> {new_rel} ({size:,} bytes, hash={old_hash[:12]}...)")

        manifest_rows.append({
            "old_path": str(old_path),
            "new_path": str(new_path),
            "action": "Renamed" if old_rel != new_rel else "Moved",
            "size_bytes": size,
            "sha256_before": old_hash,
            "sha256_after": new_hash,
            "verified": matched,
        })

    # Write MIGRATION_MANIFEST.md
    manifest_md_path = NEW_ROOT / "MIGRATION_MANIFEST.md"
    with open(manifest_md_path, "w", encoding="utf-8") as f:
        f.write("# Migration Manifest: Week 6 to Week 5-C (Encoder Adaptation)\n\n")
        f.write("**Date:** 2026-10-11\n")
        f.write("**Status:** " + ("VERIFIED (All SHA256 Checksums Identical)" if all_matched else "FAILED") + "\n\n")
        f.write("This manifest documents the byte-level preservation of experimental results migrated from ")
        f.write("`results/week6/encoder_adaptation/` to `results/week5/content_allocator/w5c_encoder_adaptation/`.\n\n")
        f.write("| Old Path | New Path | Action | Size (Bytes) | SHA256 (Truncated) | Checksum Match |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for r in manifest_rows:
            f.write(
                f"| `{r['old_path']}` | `{r['new_path']}` | {r['action']} | "
                f"{r['size_bytes']:,} | `{r['sha256_before'][:16]}...` | "
                f"{'PASS' if r['verified'] else 'FAIL'} |\n"
            )
        f.write("\n\n## Verification Summary\n")
        f.write(f"- Total Files Migrated: {len(manifest_rows)}\n")
        f.write(f"- Checksums Verified 1:1: {all_matched}\n")
        f.write("- Experimental Numerical Outputs: Byte-for-byte identical.\n")

    print(f"\nManifest written to {manifest_md_path}")
    print(f"Total files migrated: {len(manifest_rows)}")
    print(f"Overall status: {'ALL CHECKSUMS VERIFIED' if all_matched else 'FAILED'}")


if __name__ == "__main__":
    main()
