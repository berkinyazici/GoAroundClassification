"""Download required Zenodo dataset files."""
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_RAW, AUGMENTED_CSV_GZ, AGG_CSV_GZ, VALIDATION_XLSX

ZENODO_BASE = "https://zenodo.org/records/7148117/files"
FILES = {
    "go_arounds_augmented.csv.gz": f"{ZENODO_BASE}/go_arounds_augmented.csv.gz?download=1",
    "go_arounds_agg.csv.gz":       f"{ZENODO_BASE}/go_arounds_agg.csv.gz?download=1",
    "validation_table.xlsx":       f"{ZENODO_BASE}/validation_table.xlsx?download=1",
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  Downloading {dest.name} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for fname, url in FILES.items():
        download_file(url, DATA_RAW / fname)
    print("Dataset download complete.")


if __name__ == "__main__":
    main()
