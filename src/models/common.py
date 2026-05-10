from pathlib import Path
from typing import Any

import lightgbm as lgb
from joblib import dump
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import MODEL_DIR


def get_model_registry() -> dict[str, Any]:
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "lda": LinearDiscriminantAnalysis(),
        "tree": DecisionTreeClassifier(random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "lightgbm": lgb.LGBMClassifier(random_state=42),
    }


def get_classifier(name: str) -> Any:
    registry = get_model_registry()
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Available models: {list(registry)}")
    return registry[name]


def save_model(model: Any, output_path: Path) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, output_path)
    return output_path
