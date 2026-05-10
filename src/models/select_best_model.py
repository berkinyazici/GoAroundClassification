"""Select the best model based on validation average precision and copy it to final_model.joblib."""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import METRICS_DIR, MODELS_DIR, FEATURE_SCHEMA, FINAL_MODEL


def select_best_model() -> dict:
    records = []
    for json_path in sorted(METRICS_DIR.glob("*.json")):
        if json_path.name in {"final_test_metrics.json", "model_comparison.csv"}:
            continue
        try:
            with open(json_path) as fh:
                m = json.load(fh)
        except Exception:
            continue

        record = {
            "model_name":              m.get("model", json_path.stem),
            "feature_set":             m.get("feature_set", "unknown"),
            "validation_roc_auc":      m.get("validation_roc_auc", 0.0),
            "validation_average_precision": m.get("validation_average_precision", 0.0),
            "validation_f1":           m.get("validation_f1", 0.0),
            "validation_precision":    m.get("validation_precision", 0.0),
            "validation_recall":       m.get("validation_recall", 0.0),
            "test_roc_auc":            m.get("test_roc_auc", 0.0),
            "test_average_precision":  m.get("test_average_precision", 0.0),
            "test_f1":                 m.get("test_f1", 0.0),
            "test_precision":          m.get("test_precision", 0.0),
            "test_recall":             m.get("test_recall", 0.0),
            "best_threshold":          m.get("best_threshold", 0.5),
            "source_file":             json_path.name,
        }
        records.append(record)

    if not records:
        raise RuntimeError("No metric JSON files found in reports/metrics/. Train models first.")

    df = pd.DataFrame(records).sort_values(
        ["validation_average_precision", "validation_f1"], ascending=False
    ).reset_index(drop=True)

    df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print("\n=== Model Comparison ===")
    print(df[["model_name", "feature_set", "validation_average_precision", "validation_roc_auc",
              "validation_f1", "test_average_precision", "test_roc_auc"]].to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest model: {best['model_name']}  [{best['feature_set']}]")
    print(f"  Val PR-AUC = {best['validation_average_precision']:.4f}")
    print(f"  Test PR-AUC = {best['test_average_precision']:.4f}")

    # Copy bundle to final_model.joblib
    source_key = f"{best['model_name']}_{best['feature_set']}"
    source_bundle = MODELS_DIR / f"{source_key}.joblib"
    if not source_bundle.exists():
        raise FileNotFoundError(f"Expected model bundle {source_bundle} not found.")
    shutil.copy2(source_bundle, FINAL_MODEL)
    print(f"  Copied {source_bundle.name} → final_model.joblib")

    # Save feature schema
    import joblib
    bundle = joblib.load(FINAL_MODEL)
    schema = bundle.get("feature_schema", {})
    schema["best_threshold"] = float(best["best_threshold"])
    schema["model_name"] = str(best["model_name"])
    schema["feature_set"] = str(best["feature_set"])
    with open(FEATURE_SCHEMA, "w") as fh:
        json.dump(schema, fh, indent=2)
    print(f"  Feature schema saved → feature_schema.json")

    return best.to_dict()


def main() -> None:
    select_best_model()


if __name__ == "__main__":
    main()
