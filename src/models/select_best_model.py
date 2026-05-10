from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.config import MODEL_DIR, MODEL_PATH, REPORT_DIR


def select_best_model(metric: str = "pr_auc") -> Path:
    metrics_dir = REPORT_DIR / "metrics"
    candidates = []
    for path in metrics_dir.glob("*_metrics.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        score = float(data.get("test", {}).get(metric, -1.0))
        model = data.get("model")
        feature_set = data.get("feature_set")
        artifact = MODEL_DIR / f"{model}_{feature_set}.joblib"
        if artifact.exists():
            candidates.append((score, path, artifact, data))
    if not candidates:
        raise FileNotFoundError("No trained model metrics/artifacts were found. Train at least one model first.")
    score, metrics_path, artifact, data = max(candidates, key=lambda item: item[0])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, MODEL_PATH)
    summary = {"selected_model": str(artifact), "deployed_model": str(MODEL_PATH), "metric": metric, "score": score, "metrics_file": str(metrics_path)}
    (metrics_dir / "selected_model.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return MODEL_PATH


if __name__ == "__main__":
    select_best_model()
