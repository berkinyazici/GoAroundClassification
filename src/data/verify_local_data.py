from pathlib import Path
from typing import List

from src.config import RAW_DIR


def verify_local_data() -> List[Path]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

    candidates = [*RAW_DIR.glob("*.csv"), *RAW_DIR.glob("*.parquet"), *RAW_DIR.glob("*.feather")]
    if not candidates:
        raise FileNotFoundError(f"No raw data files found in {RAW_DIR}")

    return sorted(candidates)


if __name__ == "__main__":
    files = verify_local_data()
    print(f"Found {len(files)} raw data file(s)")
    for path in files:
        print(f"- {path}")
