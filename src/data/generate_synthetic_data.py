"""
Generate a synthetic dataset that closely mimics the Zenodo go_arounds_augmented.csv.gz schema.

Real dataset statistics (from the paper):
  - ~9M landings total, ~33,000 go-arounds (~0.37% rate)
  - 176 airports, 44 countries, 2019 data from OpenSky
  - Features: ADS-B context + METAR weather

This script generates a smaller but statistically representative sample that can be
used when the real dataset is not available locally. For real experiments, replace
data/raw/go_arounds_augmented.csv.gz with the actual Zenodo file.
"""
import gzip
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_RAW

random.seed(42)
np.random.seed(42)

N_TOTAL = 500_000          # synthetic row count
GA_RATE  = 0.0037          # real dataset go-around rate ≈ 0.37%
N_GA     = int(N_TOTAL * GA_RATE)
N_NORMAL = N_TOTAL - N_GA

AIRPORTS = [
    "EDDF", "EGLL", "LFPG", "EHAM", "LEMD", "LIRF", "LSZH", "EDDM",
    "LEBL", "LOWW", "EFHK", "EKCH", "ENGM", "EPWA", "LKPR", "LTFM",
    "UUEE", "UKBB", "OMDB", "OTHH", "VIDP", "VHHH", "ZBAA", "WSSS",
    "YSSY", "YMML", "CYYZ", "CYUL", "KORD", "KJFK", "KLAX", "KATL",
    "KDFW", "KDEN", "KMIA", "KSFO", "KBOS", "KEWR", "KDTW", "KPHX",
]
RUNWAYS = ["07L","07R","10","10L","10R","18","18L","18R","25L","25R","27","27L","27R","34","34L","34R","01","01L","01R","19","22","22L","22R"]
WTC_CATS = ["L", "M", "H", "J"]
TYPECODES = ["A320","A321","A319","A330","A350","B737","B738","B777","B788","B789","B752","E190","CRJ9","AT76","DH8D"]
AIRCRAFT_TYPES = ["L2J","L4J","L2T","L1T","L2P"]
COUNTRIES = ["DE","GB","FR","NL","ES","IT","CH","AT","FI","DK","NO","PL","CZ","TR","RU","UA","AE","QA","IN","HK","CN","SG","AU","CA","US"]
REGIONS   = ["EU","NA","AS","OC","ME","AF"]
COUNTRY_REGION = {c: random.choice(REGIONS) for c in COUNTRIES}
WEATHER_INTENSITY = ["", "", "", "", "-", "+", "VC"]
WEATHER_PREC = ["", "", "", "RA", "SN", "DZ", "SG", "GR"]
WEATHER_DESC = ["", "", "", "TS", "BL", "FZ", "SH"]
WEATHER_OBS  = ["", "", "", "FG", "BR", "HZ", "FU"]
WEATHER_OTHER = ["", "", "SQ", "PO", "FC"]


def _ga_feature_bias(n: int) -> dict:
    """Go-around events have slightly different feature distributions."""
    return {
        "wind_speed_knts":  np.random.gamma(4, 3, n) + 8,
        "wind_gust_knts":   np.random.gamma(3, 5, n) + 5,
        "visibility_m":     np.clip(np.random.normal(5000, 3000, n), 200, 20000),
        "temperature_deg":  np.random.normal(10, 12, n),
        "press_sea_level_p": np.random.normal(1010, 6, n),
    }


def _normal_feature_bias(n: int) -> dict:
    return {
        "wind_speed_knts":  np.random.gamma(2, 3, n),
        "wind_gust_knts":   np.clip(np.random.gamma(1, 3, n), 0, None),
        "visibility_m":     np.clip(np.random.normal(9000, 2500, n), 200, 20000),
        "temperature_deg":  np.random.normal(12, 11, n),
        "press_sea_level_p": np.random.normal(1013, 4, n),
    }


def make_rows(n: int, is_ga: bool) -> pd.DataFrame:
    bias = _ga_feature_bias(n) if is_ga else _normal_feature_bias(n)
    airports = np.random.choice(AIRPORTS, n)
    countries = np.random.choice(COUNTRIES, n)
    op_countries = np.random.choice(COUNTRIES, n)

    # Generate timestamps throughout 2019
    start_ts = datetime(2019, 1, 1).timestamp()
    end_ts   = datetime(2020, 1, 1).timestamp()
    timestamps = [
        datetime.utcfromtimestamp(start_ts + r * (end_ts - start_ts))
        for r in np.random.uniform(0, 1, n)
    ]

    df = pd.DataFrame({
        "time":             timestamps,
        "icao24":           [f"{random.randint(0,0xFFFFFF):06x}" for _ in range(n)],
        "callsign":         [f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ',k=3))}{random.randint(100,9999)}" for _ in range(n)],
        "registration":     [f"D-{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ',k=4))}" for _ in range(n)],
        "airport":          airports,
        "runway":           np.random.choice(RUNWAYS, n),
        "has_ga":           [True if is_ga else False] * n,
        "n_approaches":     np.random.choice([1, 1, 1, 2], n),
        "n_rwy_approached": np.random.choice([1, 1, 2], n),
        "typecode":         np.random.choice(TYPECODES, n),
        "icaoaircrafttype": np.random.choice(AIRCRAFT_TYPES, n),
        "wtc":              np.random.choice(WTC_CATS, n, p=[0.05, 0.65, 0.25, 0.05]),
        "glide_slope_angle": np.random.normal(3.0, 0.3, n).clip(2.0, 5.0),
        "rwy_length":       np.random.choice([2400, 2800, 3000, 3200, 3500, 3800, 4000, 4200], n).astype(float),
        "has_intersection": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "airport_country":  countries,
        "airport_region":   [COUNTRY_REGION[c] for c in countries],
        "operator_country": op_countries,
        "operator_region":  [COUNTRY_REGION[c] for c in op_countries],
        "wind_speed_knts":  bias["wind_speed_knts"].clip(0),
        "wind_dir_deg":     np.random.uniform(0, 360, n),
        "wind_gust_knts":   bias["wind_gust_knts"].clip(0),
        "visibility_m":     bias["visibility_m"],
        "temperature_deg":  bias["temperature_deg"],
        "press_sea_level_p": bias["press_sea_level_p"],
        "press_p":          bias["press_sea_level_p"] - np.random.uniform(0, 5, n),
        "weather_intensity": np.random.choice(WEATHER_INTENSITY, n),
        "weather_precipitation": np.random.choice(WEATHER_PREC, n, p=[0.6,0.1,0.1,0.05,0.05,0.04,0.03,0.03]),
        "weather_desc":     np.random.choice(WEATHER_DESC, n, p=[0.6,0.1,0.1,0.08,0.04,0.04,0.04]),
        "weather_obscuration": np.random.choice(WEATHER_OBS, n, p=[0.65,0.1,0.1,0.05,0.05,0.03,0.02]),
        "weather_other":    np.random.choice(WEATHER_OTHER, n, p=[0.85,0.07,0.04,0.02,0.02]),
    })

    # Introduce realistic missingness
    for col, miss_rate in [
        ("wind_gust_knts", 0.40), ("weather_intensity", 0.15),
        ("weather_precipitation", 0.20), ("weather_desc", 0.30),
        ("weather_obscuration", 0.35), ("weather_other", 0.60),
        ("press_sea_level_p", 0.10), ("press_p", 0.12),
        ("glide_slope_angle", 0.08),
    ]:
        mask = np.random.random(n) < miss_rate
        df.loc[mask, col] = np.nan if df[col].dtype != object else None

    return df


def generate() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW / "go_arounds_augmented.csv.gz"
    agg_path = DATA_RAW / "go_arounds_agg.csv.gz"

    if out_path.exists():
        print(f"  {out_path.name} already exists ({out_path.stat().st_size / 1e6:.1f} MB) — skipping generation.")
        return

    print(f"Generating synthetic dataset: {N_TOTAL:,} rows ({N_GA:,} go-arounds) ...")
    ga_rows  = make_rows(N_GA,     is_ga=True)
    nl_rows  = make_rows(N_NORMAL, is_ga=False)
    df = pd.concat([ga_rows, nl_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Writing {out_path.name} ...")
    df.to_csv(out_path, index=False, compression="gzip")
    print(f"  Written: {out_path.stat().st_size / 1e6:.1f} MB  ({len(df):,} rows)")

    # Aggregate dataset (small summary file)
    agg = (
        df.groupby(["airport", "runway"])
        .agg(
            n_landings=("has_ga", "count"),
            n_ga=("has_ga", "sum"),
        )
        .reset_index()
    )
    agg["ga_rate"] = agg["n_ga"] / agg["n_landings"]
    agg.to_csv(agg_path, index=False, compression="gzip")
    print(f"  Aggregate file: {agg_path.name} ({len(agg)} airport-runway pairs)")

    # Minimal validation_table.xlsx
    val_path = DATA_RAW / "validation_table.xlsx"
    if not val_path.exists():
        summary = df[["airport", "has_ga"]].groupby("airport").agg(
            n_landings=("has_ga", "count"),
            n_ga=("has_ga", "sum"),
        ).reset_index()
        summary.to_excel(val_path, index=False)
        print(f"  Validation table: {val_path.name}")

    print(f"Synthetic data generation complete. GA rate: {100*N_GA/N_TOTAL:.3f}%")


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
