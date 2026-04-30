from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def get_model(name: str):
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(max_iter=200, class_weight="balanced")
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    raise ValueError(f"Unsupported model: {name}")
