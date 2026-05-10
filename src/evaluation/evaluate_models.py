from pathlib import Path
import json

import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import MODEL_DIR, MODEL_PATH, PROCESSED_DIR, REPORT_DIR
from src.features.build_features import resolve_target_column


def get_model_path() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    candidate = next(MODEL_DIR.glob("*.joblib"), None)
    if candidate is not None:
        return candidate
    raise FileNotFoundError(
        f"No model file found at {MODEL_PATH} and no .joblib file found in {MODEL_DIR}"
    )


def calculate_metrics(model, test_df: pd.DataFrame, target: str) -> dict[str, float]:
    target = resolve_target_column(test_df, target)
    y_true = test_df[target]
    X_test = test_df.drop(columns=[target])
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def evaluate_all_models(target: str = "target") -> dict[str, dict[str, float]]:
    test_path = PROCESSED_DIR / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError("Test split not found. Run src.data.make_splits first.")

    test_df = pd.read_parquet(test_path)

    all_metrics = {}
    for model_path in sorted(MODEL_DIR.glob("*.joblib")):
        if model_path.stem == "best_model":
            continue
        all_metrics[model_path.stem] = calculate_metrics(load(model_path), test_df, target)

    metrics_dir = REPORT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "all_models_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


def evaluate_model(target: str = "target") -> dict[str, float]:
    test_path = PROCESSED_DIR / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError("Test split not found. Run src.data.make_splits first.")

    model_path = get_model_path()
    metrics = calculate_metrics(load(model_path), pd.read_parquet(test_path), target)

    metrics_path = REPORT_DIR / "metrics" / "evaluation_metrics.yaml"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("\n".join(f"{key}: {value:.4f}" for key, value in metrics.items()))
    evaluate_all_models(target=target)
    return metrics


if __name__ == "__main__":
    results = evaluate_model()
    print("Evaluation metrics:")
    for name, value in results.items():
        print(f"- {name}: {value:.4f}")
