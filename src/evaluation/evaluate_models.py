from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import MODEL_PATH, PROCESSED_DIR, REPORT_DIR


def evaluate_model(target: str = "target") -> dict[str, float]:
    test_path = PROCESSED_DIR / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError("Test split not found. Run src.data.make_splits first.")

    model = load(MODEL_PATH)
    test_df = pd.read_parquet(test_path)
    y_true = test_df[target]
    X_test = test_df.drop(columns=[target])
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "metrics" / "evaluation_metrics.yaml"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("\n".join(f"{key}: {value:.4f}" for key, value in metrics.items()))
    return metrics


if __name__ == "__main__":
    results = evaluate_model()
    print("Evaluation metrics:")
    for name, value in results.items():
        print(f"- {name}: {value:.4f}")
