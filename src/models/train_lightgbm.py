"""Train LightGBM (gradient-boosted trees) for go-around classification."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR, METRICS_DIR, FIGURES_DIR
from src.models.common import (
    create_preprocessor, evaluate_binary_classifier,
    load_splits, save_metrics, save_model_bundle, tune_threshold_for_f1,
)

FEATURE_SETS = ["context_only", "context_metar"]


class LGBMSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Thin sklearn-compatible wrapper around lgb.Booster."""

    def __init__(self, **params):
        self.params = params
        self.booster_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, eval_set=None):
        pos = int(np.sum(y))
        neg = len(y) - pos
        scale_pos_weight = neg / max(pos, 1)
        params = {
            "objective":        "binary",
            "metric":           "average_precision",
            "scale_pos_weight": scale_pos_weight,
            "learning_rate":    0.05,
            "num_leaves":       63,
            "min_child_samples": 20,
            "verbose":          -1,
        }
        params.update(self.params)
        train_data = lgb.Dataset(X, label=y)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        valid_sets = [train_data]
        if eval_set is not None:
            X_va, y_va = eval_set
            valid_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
            valid_sets = [train_data, valid_data]
        self.booster_ = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X):
        prob1 = self.booster_.predict(X)
        return np.column_stack([1 - prob1, prob1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_lightgbm(feature_set: str = "context_metar") -> dict:
    print(f"\n=== LightGBM [{feature_set}] ===")
    X_tr, y_tr, X_va, y_va, X_te, y_te, num_feats, cat_feats = load_splits(feature_set)

    preprocessor = create_preprocessor(num_feats, cat_feats)
    X_tr_pre = preprocessor.fit_transform(X_tr, y_tr)
    X_va_pre = preprocessor.transform(X_va)
    X_te_pre = preprocessor.transform(X_te)

    clf = LGBMSklearnWrapper()
    clf.fit(X_tr_pre, y_tr.values, eval_set=(X_va_pre, y_va.values))

    va_prob = clf.predict_proba(X_va_pre)[:, 1]
    te_prob = clf.predict_proba(X_te_pre)[:, 1]
    best_thresh = tune_threshold_for_f1(y_va.values, va_prob)

    metrics = {"model": "lightgbm", "feature_set": feature_set, "best_threshold": best_thresh}
    metrics.update(evaluate_binary_classifier(y_va.values, va_prob, threshold=0.5,         split_name="validation"))
    metrics.update(evaluate_binary_classifier(y_te.values, te_prob, threshold=best_thresh, split_name="test"))

    key = f"lightgbm_{feature_set}"
    save_metrics(metrics, METRICS_DIR / f"{key}.json")
    schema = {"numeric_features": num_feats, "categorical_features": cat_feats, "feature_set": feature_set}
    save_model_bundle(clf, preprocessor, schema, MODELS_DIR / f"{key}.joblib")

    # Feature importance
    try:
        booster = clf.booster_
        try:
            ohe_names = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_feats).tolist()
        except Exception:
            ohe_names = []
        feat_names = num_feats + ohe_names
        importances = booster.feature_importance(importance_type="gain")
        if len(feat_names) == len(importances):
            top_n = min(20, len(feat_names))
            idx = np.argsort(importances)[-top_n:][::-1]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh([feat_names[i] for i in reversed(idx)], importances[idx[::-1]])
            ax.set_title(f"LightGBM Feature Importance (gain) [{feature_set}]")
            ax.set_xlabel("Gain")
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f"lightgbm_feature_importance_{feature_set}.png", dpi=100)
            plt.close(fig)
    except Exception as e:
        print(f"  Feature importance plot skipped: {e}")

    print(f"  Val  ROC-AUC={metrics['validation_roc_auc']:.4f}  PR-AUC={metrics['validation_average_precision']:.4f}")
    print(f"  Test ROC-AUC={metrics['test_roc_auc']:.4f}  PR-AUC={metrics['test_average_precision']:.4f}")
    return metrics


def main() -> None:
    for fs in FEATURE_SETS:
        train_lightgbm(fs)


if __name__ == "__main__":
    main()
