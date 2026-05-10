from __future__ import annotations

from pathlib import Path

from src.config import RAW_DIR

REQUIRED_RAW_FILES = [
    "go_arounds_augmented.csv.gz",
    "go_arounds_agg.csv.gz",
    "validation_table.xlsx",
]


def verify_local_data(raw_dir: Path = RAW_DIR, strict: bool = False) -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        if strict:
            raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")
        return []

    found = [raw_dir / name for name in REQUIRED_RAW_FILES if (raw_dir / name).exists()]
    missing = [name for name in REQUIRED_RAW_FILES if not (raw_dir / name).exists()]
    if strict and missing:
        raise FileNotFoundError(
            "Missing required raw dataset files under "
            f"{raw_dir}: {', '.join(missing)}. Run `python -m src.data.download_data`."
        )
    for path in found:
        print(f"FOUND {path} ({path.stat().st_size / 1_048_576:.1f} MiB)")
    if missing:
        print("Missing optional/raw files: " + ", ".join(missing))
    return found


if __name__ == "__main__":
    files = verify_local_data(strict=True)
    print(f"Verified {len(files)} raw file(s).")
