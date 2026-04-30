from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
