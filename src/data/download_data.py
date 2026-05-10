from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm

from src.config import RAW_DIR

ZENODO_FILES = {
    "go_arounds_augmented.csv.gz": "https://zenodo.org/records/7148117/files/go_arounds_augmented.csv.gz?download=1",
    "go_arounds_agg.csv.gz": "https://zenodo.org/records/7148117/files/go_arounds_agg.csv.gz?download=1",
    "validation_table.xlsx": "https://zenodo.org/records/7148117/files/validation_table.xlsx?download=1",
}


def download_data(raw_dir: Path = RAW_DIR) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for filename, url in ZENODO_FILES.items():
        output = raw_dir / filename
        if output.exists() and output.stat().st_size > 0:
            print(f"Skipping existing file: {output} ({output.stat().st_size / 1_048_576:.1f} MiB)")
            outputs.append(output)
            continue
        print(f"Downloading {filename} from Zenodo...")
        with urlopen(url, timeout=120) as response, output.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            with tqdm(total=total, unit="B", unit_scale=True, desc=filename) as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    progress.update(len(chunk))
        outputs.append(output)
        print(f"Wrote {output} ({output.stat().st_size / 1_048_576:.1f} MiB)")
    return outputs


if __name__ == "__main__":
    download_data()
