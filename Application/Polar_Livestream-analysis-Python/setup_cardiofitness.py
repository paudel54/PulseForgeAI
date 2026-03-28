"""
One-time setup: downloads the 3 binary artefacts that are NOT tracked in git
(too large / binary) into your local cardiofitness-main/ copy.

Run once from the project root:
    python setup_cardiofitness.py

Files downloaded from github.com/sdimi/cardiofitness (main branch):

  cardiofitness-main/models/20201109-013142/
      weights-regression-improvement-8.89.hdf5   (~316 KB)

  cardiofitness-main/data/
      scaler_FI.save                             (~2 KB)
      PCA_FI_mapping_09999.save                  (~27 KB)

Everything else (model_architecture.json, extracted_features.csv, source code)
is already present in the local repo.

Reference:
  Spathis et al. (2022). Nature Digital Medicine.
  https://doi.org/10.1038/s41746-022-00719-1
"""

import sys
import urllib.request
from pathlib import Path

_ROOT    = Path(__file__).parent
_CF_DIR  = _ROOT / "cardiofitness-main"
_BASE    = "https://raw.githubusercontent.com/sdimi/cardiofitness/main"

FILES = [
    (
        "models/20201109-013142/weights-regression-improvement-8.89.hdf5",
        _CF_DIR / "models" / "20201109-013142" / "weights-regression-improvement-8.89.hdf5",
    ),
    (
        "data/scaler_FI.save",
        _CF_DIR / "data" / "scaler_FI.save",
    ),
    (
        "data/PCA_FI_mapping_09999.save",
        _CF_DIR / "data" / "PCA_FI_mapping_09999.save",
    ),
]


def _progress(count, block_size, total):
    pct = min(int(count * block_size * 100 / max(total, 1)), 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)


def download_all(force: bool = False) -> bool:
    all_ok = True
    for rel_path, dest in FILES:
        if dest.exists() and not force:
            print(f"  ✓  {dest.name}  (already present)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_BASE}/{rel_path}"
        print(f"\nDownloading {dest.name} …")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            print(f"\n  ✓  {dest}")
        except Exception as exc:
            print(f"\n  ✗  Failed ({exc})")
            all_ok = False
    return all_ok


def check_deps() -> bool:
    ok = True
    for pkg, install in [("tensorflow", "tensorflow"), ("sklearn", "scikit-learn"), ("joblib", "joblib")]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  ✗  {pkg} not found — pip install {install}")
            ok = False
    return ok


def check_repo() -> bool:
    if not _CF_DIR.is_dir():
        print(f"  ✗  cardiofitness-main/ not found at {_CF_DIR}")
        return False
    print(f"  ✓  cardiofitness-main/  found")
    return True


if __name__ == "__main__":
    force = "--force" in sys.argv
    print("=== CardioFitness Model Setup ===\n")

    print("Checking local repo …")
    if not check_repo():
        sys.exit(1)

    print("\nChecking Python dependencies …")
    if not check_deps():
        sys.exit(1)
    print("  ✓  All dependencies present\n")

    print("Downloading missing binary artefacts …\n")
    ok = download_all(force=force)

    if ok:
        print("\n✓ Done — Neural-net VO2max predictor is ready.")
    else:
        print("\n✗ Some files failed. Check your internet connection and retry.")
        sys.exit(1)
