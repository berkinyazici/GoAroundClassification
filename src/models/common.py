"""Shared model utilities: preprocessing, evaluation, serialisation."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import TRAIN_PARQUET, VALID_PARQUET, TEST_PARQUET
from src.features.build_features import load_data, build_feature_matrix

RISK_ENCODING_COLUMNS = {
    "airport": "airport_risk_train",
    "runway": "runway_risk_train",
    "typecode": "typecode_risk_train",
}


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

def load_splits(
    feature_set: str = "context_metar",
    sample_frac: float | None = None,
    negative_ratio: float | None = 10.0,
) -> tuple:
    """Return (X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats)."""
    print(f"  Loading splits (feature_set={feature_set}) ...")
    t0 = time.time()

    train = load_data(TRAIN_PARQUET)
    valid = load_data(VALID_PARQUET)
    test  = load_data(TEST_PARQUET)

    if sample_frac is not None and sample_frac < 1.0:
        # Stratified sample to preserve go-around rate
        pos_tr = train[train["target"] == 1]
        neg_tr = train[train["target"] == 0].sample(
            frac=sample_frac, random_state=42
        )
        train = pd.concat([pos_tr, neg_tr]).sample(frac=1, random_state=42)
        print(f"  Train sampled: {len(train):,} rows ({int(train['target'].sum())} go-arounds)")
    elif negative_ratio is not None:
        pos_tr = train[train["target"] == 1]
        neg_tr = train[train["target"] == 0]
        max_negatives = int(len(pos_tr) * negative_ratio)
        if len(pos_tr) > 0 and len(neg_tr) > max_negatives:
            neg_tr = neg_tr.sample(n=max_negatives, random_state=42)
            train = pd.concat([pos_tr, neg_tr]).sample(frac=1, random_state=42)
            print(
                f"  Train downsampled: {len(train):,} rows "
                f"({len(pos_tr):,} go-arounds, negative_ratio={negative_ratio:g}:1)"
            )

    print(f"  Train: {len(train):,}  Valid: {len(valid):,}  Test: {len(test):,}")

    X_tr, y_tr, num_feats, cat_feats = build_feature_matrix(train, feature_set)
    X_va, y_va, _,         _         = build_feature_matrix(valid, feature_set)
    X_te, y_te, _,         _         = build_feature_matrix(test,  feature_set)

    if feature_set == "context_metar_engineered":
        risk_schema = fit_risk_encoders(X_tr, y_tr)
        X_tr = apply_risk_encoders(X_tr, risk_schema)
        X_va = apply_risk_encoders(X_va, risk_schema)
        X_te = apply_risk_encoders(X_te, risk_schema)
        X_tr.attrs["risk_schema"] = risk_schema

    print(f"  Features: {len(num_feats)} numeric + {len(cat_feats)} categorical  [{time.time()-t0:.1f}s]")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats


def fit_risk_encoders(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_samples: int = 100,
) -> dict[str, Any]:
    global_rate = float(y_train.mean())
    schema: dict[str, Any] = {"global_rate": global_rate, "columns": {}}
    train_with_target = X_train.copy()
    train_with_target["_target"] = y_train.values

    for source_col, feature_col in RISK_ENCODING_COLUMNS.items():
        if source_col not in train_with_target.columns:
            continue
        stats = train_with_target.groupby(source_col)["_target"].agg(["sum", "count"])
        smoothed = (stats["sum"] + global_rate * min_samples) / (stats["count"] + min_samples)
        schema["columns"][source_col] = {
            "feature": feature_col,
            "mapping": {str(key): float(value) for key, value in smoothed.items()},
        }
    return schema


def apply_risk_encoders(X: pd.DataFrame, risk_schema: dict[str, Any] | None) -> pd.DataFrame:
    X = X.copy()
    if not risk_schema:
        return X

    global_rate = float(risk_schema.get("global_rate", 0.0))
    for source_col, config in risk_schema.get("columns", {}).items():
        feature_col = config["feature"]
        mapping = config.get("mapping", {})
        if source_col in X.columns:
            X[feature_col] = X[source_col].astype(str).map(mapping).fillna(global_rate).astype("float32")
        else:
            X[feature_col] = global_rate
    return X


# --------------------------------------------------------------------------- #
# Preprocessing                                                                #
# --------------------------------------------------------------------------- #

def create_preprocessor(numeric_features: list[str], categorical_features: list[str],
                        ohe_min_frequency: int = 2000) -> ColumnTransformer:
    """
    ohe_min_frequency: a category must appear at least this many times in the
    training set to get its own OHE column (others → 'infrequent_sklearn').
    2000 out of ~5.6M rows ≈ top airports/runways only → keeps RAM manageable.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=ohe_min_frequency,
            sparse_output=False,
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #

def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    split_name: str = "test",
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics = {
        f"{split_name}_accuracy":           round(float(accuracy_score(y_true, y_pred)), 6),
        f"{split_name}_precision":          round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        f"{split_name}_recall":             round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        f"{split_name}_f1":                 round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        f"{split_name}_roc_auc":            round(float(roc_auc_score(y_true, y_prob)), 6),
        f"{split_name}_average_precision":  round(float(average_precision_score(y_true, y_prob)), 6),
        f"{split_name}_confusion_matrix":   cm,
        f"{split_name}_threshold":          threshold,
    }
    return metrics


def tune_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return tune_threshold_for_fbeta(y_true, y_prob, beta=1.0)


def tune_threshold_for_fbeta(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0,
    min_recall: float | None = None,
) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precision_values = precisions[:-1]
    recall_values = recalls[:-1]
    beta_squared = beta**2
    denom = (beta_squared * precision_values) + recall_values
    scores = np.where(
        denom > 0,
        (1 + beta_squared) * precision_values * recall_values / np.maximum(denom, 1e-9),
        0.0,
    )
    if min_recall is not None:
        scores = np.where(recall_values >= min_recall, scores, -1.0)
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx])


# --------------------------------------------------------------------------- #
# Serialisation                                                                #
# --------------------------------------------------------------------------- #

def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  Metrics saved → {path.name}")


def save_model_bundle(
    model: Any,
    preprocessor: Any,
    feature_schema: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":          model,
        "preprocessor":   preprocessor,
        "feature_schema": feature_schema,
    }
    joblib.dump(bundle, path)
    print(f"  Model bundle saved → {path.name}")


def load_model_bundle(path: Path) -> dict:
    return joblib.load(path)
