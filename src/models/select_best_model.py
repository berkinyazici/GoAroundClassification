from pathlib import Path

from joblib import dump
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import pandas as pd

from src.config import MODEL_DIR, MODEL_PATH, PROCESSED_DIR
from src.features.build_features import build_features, resolve_target_column
from src.models.common import get_model_registry


def select_best_model(target: str = "target") -> Path:
    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed train/test splits not found. Run src.data.make_splits first.")

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    target = resolve_target_column(train, target)

    best_score = -1.0
    best_model = None
    best_name = None

    for name, model in get_model_registry().items():
        X_train, y_train, feature_pipeline = build_features(train, target)
        y_test = test[target]
        model.fit(X_train, y_train)
        full_pipeline = Pipeline([("features", feature_pipeline), ("model", model)])
        X_test = test.drop(columns=[target])

        if hasattr(full_pipeline, "predict_proba"):
            score = roc_auc_score(y_test, full_pipeline.predict_proba(X_test)[:, 1])
        else:
            score = roc_auc_score(y_test, full_pipeline.predict(X_test))

        if score > best_score:
            best_score = score
            best_model = full_pipeline
            best_name = name

    if best_model is None or best_name is None:
        raise RuntimeError("No model could be selected.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_PATH
    dump(best_model, output_path)
    return output_path


if __name__ == "__main__":
    path = select_best_model()
    print(f"Selected best model and saved to {path}")
