from __future__ import annotations

import argparse
import json

from src.models.train_model import train_model
from src.models.select_best_model import select_best_model
from src.config import REPORT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ADS-B-only vs ADS-B+METAR ablation experiments.")
    parser.add_argument("--models", nargs="+", default=["logreg", "tree"], choices=["logreg", "lda", "tree", "rf", "mlp", "lightgbm"])
    args = parser.parse_args()
    summary = []
    for feature_set in ["adsb_only", "adsb_plus_metar"]:
        for model_name in args.models:
            train_model(model_name=model_name, feature_set=feature_set)
            metrics_path = REPORT_DIR / "metrics" / f"{model_name}_{feature_set}_metrics.json"
            summary.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    select_best_model(metric="pr_auc")
    out = REPORT_DIR / "metrics" / "ablation_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved ablation summary to {out}")


if __name__ == "__main__":
    main()
