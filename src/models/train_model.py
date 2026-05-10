from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR
from src.features.build_features import TARGET_COLUMN, split_X_y
from src.models.common import build_model_pipeline


def _score_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return model.predict(X)


def best_f1_threshold(y_true, y_prob) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_model(model_name: str = "logreg", feature_set: str = "adsb_plus_metar") -> Path:
    train_path = PROCESSED_DIR / "train.parquet"
    valid_path = PROCESSED_DIR / "valid.parquet"
    test_path = PROCESSED_DIR / "test.parquet"
    for path in (train_path, valid_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing processed split: {path}. Run `python -m src.data.make_splits` first.")

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    test_df = pd.read_parquet(test_path)
    X_train, y_train, numeric, categorical = split_X_y(train_df, feature_set=feature_set)
    X_valid, y_valid, _, _ = split_X_y(valid_df, feature_set=feature_set)
    X_test, y_test, _, _ = split_X_y(test_df, feature_set=feature_set)

    pos = max(int(y_train.sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = neg / pos
    model = build_model_pipeline(model_name, numeric, categorical, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)

    valid_prob = _score_probabilities(model, X_valid)
    threshold = best_f1_threshold(y_valid, valid_prob)
    test_prob = _score_probabilities(model, X_test)
    metrics = {
        "model": model_name,
        "feature_set": feature_set,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "validation": metrics_at_threshold(y_valid, valid_prob, threshold),
        "test": metrics_at_threshold(y_test, test_prob, threshold),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DIR / f"{model_name}_{feature_set}.joblib"
    dump({"pipeline": model, "threshold": threshold, "feature_set": feature_set, "numeric_features": numeric, "categorical_features": categorical}, output_path)
    metrics_path = REPORT_DIR / "metrics" / f"{model_name}_{feature_set}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["test"], indent=2))
    print(f"Saved model to {output_path}")
    print(f"Saved metrics to {metrics_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a go-around classifier.")
    parser.add_argument("--model", default="logreg", choices=["logreg", "lda", "tree", "rf", "mlp", "lightgbm"])
    parser.add_argument("--feature-set", default="adsb_plus_metar", choices=["adsb_only", "metar_only", "adsb_plus_metar"])
    parser.add_argument("--target", default=TARGET_COLUMN, help="Kept for CLI compatibility; the canonical target is 'target'.")
    args = parser.parse_args()
    train_model(model_name=args.model, feature_set=args.feature_set)


if __name__ == "__main__":
    main()
