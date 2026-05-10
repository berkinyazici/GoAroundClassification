from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import MODEL_DIR


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)
    except TypeError:  # pragma: no cover - compatibility for older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse=True)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def get_estimator(name: str, scale_pos_weight: float | None = None):
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=None)
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name in {"tree", "decision_tree"}:
        return DecisionTreeClassifier(max_depth=12, min_samples_leaf=25, class_weight="balanced", random_state=42)
    if name in {"rf", "random_forest"}:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight or 1.0,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError("Unknown model '%s'. Available models: logreg, lda, tree, rf, mlp, lightgbm" % name)


def build_model_pipeline(
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    scale_pos_weight: float | None = None,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", get_estimator(model_name, scale_pos_weight=scale_pos_weight)),
        ]
    )


def get_model_registry() -> dict[str, str]:
    return {name: name for name in ["logreg", "lda", "tree", "rf", "mlp", "lightgbm"]}


def get_classifier(name: str) -> Any:
    return get_estimator(name)


def save_model(model: Any, output_path: Path) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, output_path)
    return output_path
