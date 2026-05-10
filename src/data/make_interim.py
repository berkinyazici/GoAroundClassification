from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.config import INTERIM_DIR, RAW_DIR
from src.data.verify_local_data import verify_local_data


def make_interim() -> Path:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    sources = verify_local_data()
    frames = []

    print(f"Found {len(sources)} raw source file(s) for interim dataset.")
    for source in tqdm(sources, desc="Loading raw sources", unit="file"):
        source_name = source.name.lower()
        if source_name.endswith((".csv", ".csv.gz")):
            frames.append(pd.read_csv(source, compression="infer", low_memory=False))
        elif source_name.endswith((".xlsx", ".xls")):
            frames.append(pd.read_excel(source))
        elif source_name.endswith(".parquet"):
            frames.append(pd.read_parquet(source))
        elif source_name.endswith(".feather"):
            frames.append(pd.read_feather(source))
        else:
            print(f"Skipping unsupported raw file: {source}")

    if not frames:
        raise RuntimeError("No interim data was created because no raw sources were loaded.")

    dataset = pd.concat(frames, ignore_index=True)
    output_path = INTERIM_DIR / "dataset.csv"
    print(f"Saving interim dataset with shape {dataset.shape} to {output_path}")
    dataset.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    output = make_interim()
    print(f"Interim dataset written to {output}")
