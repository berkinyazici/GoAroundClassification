from __future__ import annotations

import argparse
from pathlib import Path

from src.models.train_model import train_model


def train_logreg(target: str = "target", feature_set: str = "adsb_plus_metar") -> Path:
    return train_model(model_name="logreg", feature_set=feature_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logreg model")
    parser.add_argument("--target", default="target")
    parser.add_argument("--feature-set", default="adsb_plus_metar", choices=["adsb_only", "metar_only", "adsb_plus_metar"])
    args = parser.parse_args()
    path = train_logreg(target=args.target, feature_set=args.feature_set)
    print(f"Saved logreg model to {path}")
