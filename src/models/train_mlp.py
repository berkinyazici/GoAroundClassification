"""Train MLP (Multi-Layer Perceptron) neural baseline for go-around classification."""
import argparse
import sys
import time
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR, METRICS_DIR
from src.models.common import (
    create_preprocessor, evaluate_binary_classifier,
    load_splits, save_metrics, save_model_bundle, tune_threshold_for_fbeta,
)

FEATURE_SETS = ["context_only", "context_metar", "context_metar_engineered"]


def train_mlp(
    feature_set: str = "context_metar",
    sample_frac: float | None = None,
    negative_ratio: float | None = 10.0,
) -> dict:
    print(f"\n=== MLP [{feature_set}] ===")
    X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats = load_splits(
        feature_set, sample_frac, negative_ratio
    )

    preprocessor = create_preprocessor(num_feats, cat_feats)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True,
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

    print(f"  Fitting MLP (early_stopping=True, verbose shows each epoch) ...")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    print(f"  Fit done [{time.time()-t0:.1f}s]  iterations={pipe.named_steps['clf'].n_iter_}")

    print("  Predicting on validation ...")
    va_prob = pipe.predict_proba(X_va)[:, 1]
    print("  Predicting on test ...")
    te_prob = pipe.predict_proba(X_te)[:, 1]
    best_thresh = tune_threshold_for_fbeta(y_va.values, va_prob, beta=2.0)

    metrics = {
        "model": "mlp",
        "feature_set": feature_set,
        "best_threshold": best_thresh,
        "threshold_metric": "f2",
        "negative_ratio": negative_ratio,
    }
    metrics.update(evaluate_binary_classifier(y_va.values, va_prob, threshold=0.5,         split_name="validation"))
    metrics.update(evaluate_binary_classifier(y_te.values, te_prob, threshold=best_thresh, split_name="test"))

    key = f"mlp_{feature_set}"
    save_metrics(metrics, METRICS_DIR / f"{key}.json")
    schema = {"numeric_features": num_feats, "categorical_features": cat_feats, "feature_set": feature_set, "risk_schema": X_tr.attrs.get("risk_schema", {})}
    save_model_bundle(pipe, preprocessor, schema, MODELS_DIR / f"{key}.joblib")

    print(f"  Val  ROC-AUC={metrics['validation_roc_auc']:.4f}  PR-AUC={metrics['validation_average_precision']:.4f}")
    print(f"  Test ROC-AUC={metrics['test_roc_auc']:.4f}  PR-AUC={metrics['test_average_precision']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--negative-ratio", type=float, default=10.0)
    args = parser.parse_args()
    for fs in FEATURE_SETS:
        train_mlp(fs, args.sample_frac, args.negative_ratio)


if __name__ == "__main__":
    main()
