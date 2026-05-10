"""Verify local dataset files are present before preprocessing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import AUGMENTED_CSV_GZ, AGG_CSV_GZ, VALIDATION_XLSX

REQUIRED = [AUGMENTED_CSV_GZ, AGG_CSV_GZ, VALIDATION_XLSX]


def verify() -> None:
    missing = []
    for p in REQUIRED:
        if p.exists():
            print(f"  Found {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  MISSING {p}")
            missing.append(p)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} required file(s). "
            "Place them under data/raw/ or run: python src/data/download_data.py"
        )
    print("Local dataset verification successful.")


def main() -> None:
    verify()


if __name__ == "__main__":
    main()
