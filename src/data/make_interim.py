from pathlib import Path
import pandas as pd

from src.config import INTERIM_DIR, RAW_DIR
from src.data.verify_local_data import verify_local_data


def make_interim() -> Path:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    sources = verify_local_data()
    frames = []

    for source in sources:
        if source.suffix.lower() == ".csv":
            frames.append(pd.read_csv(source))
        elif source.suffix.lower() == ".parquet":
            frames.append(pd.read_parquet(source))
        elif source.suffix.lower() == ".feather":
            frames.append(pd.read_feather(source))

    if not frames:
        raise RuntimeError("No interim data was created because no raw sources were loaded.")

    dataset = pd.concat(frames, ignore_index=True)
    output_path = INTERIM_DIR / "dataset.parquet"
    dataset.to_parquet(output_path)
    return output_path


if __name__ == "__main__":
    output = make_interim()
    print(f"Interim dataset written to {output}")
