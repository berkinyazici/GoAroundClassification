"""Error analysis: summarise FP/FN by airport, runway, aircraft, weather bins."""
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import FINAL_MODEL, FEATURE_SCHEMA, ERROR_ANALYSIS_DIR, TEST_PARQUET
from src.models.common import load_model_bundle
from src.features.build_features import load_data, build_feature_matrix


def _save(df: pd.DataFrame, name: str) -> None:
    path = ERROR_ANALYSIS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {path.name}")


def run_error_analysis() -> None:
    print("Running error analysis ...")
    bundle = load_model_bundle(FINAL_MODEL)
    model        = bundle["model"]
    preprocessor = bundle["preprocessor"]

    if FEATURE_SCHEMA.exists():
        with open(FEATURE_SCHEMA) as fh:
            schema = json.load(fh)
    else:
        schema = bundle["feature_schema"]

    feature_set = schema.get("feature_set", "context_metar")
    threshold   = schema.get("best_threshold", 0.5)

    raw = load_data(TEST_PARQUET)
    X_te, y_te, num_feats, cat_feats = build_feature_matrix(raw, feature_set)

    from sklearn.pipeline import Pipeline as _Pipeline
    if isinstance(model, _Pipeline):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        try:
            X_pre = preprocessor.transform(X_te)
        except Exception:
            X_pre = X_te.values
        y_prob = model.predict_proba(X_pre)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_te.values

    analysis_df = X_te.copy()
    analysis_df["y_true"]   = y_true
    analysis_df["y_pred"]   = y_pred
    analysis_df["y_prob"]   = y_prob
    analysis_df["outcome"]  = np.select(
        [
            (y_true == 1) & (y_pred == 1),
            (y_true == 0) & (y_pred == 0),
            (y_true == 1) & (y_pred == 0),
            (y_true == 0) & (y_pred == 1),
        ],
        ["TP", "TN", "FN", "FP"], default="?"
    )

    ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # By airport
    for grp_col in ["airport", "runway", "wtc", "icaoaircrafttype"]:
        if grp_col not in analysis_df.columns:
            continue
        summary = (
            analysis_df.groupby(grp_col)["outcome"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["TP", "TN", "FP", "FN"]:
            if col not in summary.columns:
                summary[col] = 0
        summary["total"] = summary[["TP", "TN", "FP", "FN"]].sum(axis=1)
        summary = summary.sort_values("FN", ascending=False)
        _save(summary, f"errors_by_{grp_col}")

    # Wind speed bins
    if "wind_speed_knts" in analysis_df.columns:
        analysis_df["wind_bin"] = pd.cut(
            pd.to_numeric(analysis_df["wind_speed_knts"], errors="coerce"),
            bins=[0, 5, 10, 15, 20, 30, 100], labels=["0-5","5-10","10-15","15-20","20-30","30+"]
        )
        wind_summary = (
            analysis_df.groupby("wind_bin")["outcome"]
            .value_counts().unstack(fill_value=0).reset_index()
        )
        _save(wind_summary, "errors_by_wind_speed")

    # Visibility bins
    if "visibility_m" in analysis_df.columns:
        analysis_df["vis_bin"] = pd.cut(
            pd.to_numeric(analysis_df["visibility_m"], errors="coerce"),
            bins=[0, 800, 3000, 5000, 10000, 99999],
            labels=["<800","800-3k","3k-5k","5k-10k","10k+"]
        )
        vis_summary = (
            analysis_df.groupby("vis_bin")["outcome"]
            .value_counts().unstack(fill_value=0).reset_index()
        )
        _save(vis_summary, "errors_by_visibility")

    # Print top FN airports
    if "airport" in analysis_df.columns:
        fn_df = analysis_df[analysis_df["outcome"] == "FN"]
        fn_counts = fn_df["airport"].value_counts().head(10)
        print("\nTop 10 airports with most False Negatives (missed go-arounds):")
        print(fn_counts.to_string())

        fp_df = analysis_df[analysis_df["outcome"] == "FP"]
        fp_counts = fp_df["airport"].value_counts().head(10)
        print("\nTop 10 airports with most False Positives (false alarms):")
        print(fp_counts.to_string())

    print("\nError analysis complete.")


def main() -> None:
    run_error_analysis()


if __name__ == "__main__":
    main()
