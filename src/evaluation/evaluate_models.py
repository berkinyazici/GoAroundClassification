from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.config import MODEL_PATH, PROCESSED_DIR, REPORT_DIR
from src.features.build_features import TARGET_COLUMN, split_X_y
from src.models.train_model import _score_probabilities, metrics_at_threshold


def _load_bundle(path: Path):
    obj = load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj
    return {"pipeline": obj, "threshold": 0.5, "feature_set": "adsb_plus_metar"}


def evaluate_model(target: str = TARGET_COLUMN, model_path: Path = MODEL_PATH) -> dict[str, float]:
    test_path = PROCESSED_DIR / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError("Test split not found. Run `python -m src.data.make_splits` first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training and model selection first.")

    bundle = _load_bundle(model_path)
    test_df = pd.read_parquet(test_path)
    X_test, y_true, _, _ = split_X_y(test_df, feature_set=bundle.get("feature_set", "adsb_plus_metar"))
    y_prob = _score_probabilities(bundle["pipeline"], X_test)
    threshold = float(bundle.get("threshold", 0.5))
    metrics = metrics_at_threshold(y_true, y_prob, threshold)

    metrics_dir = REPORT_DIR / "metrics"
    figures_dir = REPORT_DIR / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], display_labels=["Landing", "Go-around"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    if len(set(y_true)) > 1:
        PrecisionRecallDisplay.from_predictions(y_true, y_prob)
        plt.title("Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig(figures_dir / "precision_recall_curve.png", dpi=160)
        plt.close()

        RocCurveDisplay.from_predictions(y_true, y_prob)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(figures_dir / "roc_curve.png", dpi=160)
        plt.close()

    print(json.dumps(metrics, indent=2))
    return {k: v for k, v in metrics.items() if isinstance(v, float)}


if __name__ == "__main__":
    evaluate_model()
