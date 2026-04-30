from __future__ import annotations

import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import Any

import joblib
import yaml
from sklearn.pipeline import Pipeline

from goaround.data.io import load_dataset, basic_clean
from goaround.data.split import random_split, group_split, time_split
from goaround.features.pipeline import build_preprocessor
from goaround.models.factory import get_model
from goaround.eval.metrics import compute_metrics


def infer_columns(df, target_col: str):
    xdf = df.drop(columns=[target_col])
    numeric_cols = xdf.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in xdf.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def apply_feature_selection(df, features: list[str] | None, target_col: str):
    if not features:
        return df
    cols = [c for c in features if c in df.columns]
    if target_col not in cols:
        cols.append(target_col)
    return df[cols].copy()


def create_split(df, cfg: dict[str, Any], target_col: str):
    split_mode = cfg.get("split_mode", "random")
    seed = int(cfg.get("seed", 42))
    if split_mode == "random":
        return random_split(df, target_col=target_col, seed=seed)
    if split_mode == "airport":
        group_col = cfg.get("group_col", "airport")
        return group_split(df, group_col=group_col, target_col=target_col, seed=seed)
    if split_mode == "time":
        time_col = cfg.get("time_col", "timestamp")
        return time_split(df, time_col=time_col, target_col=target_col)
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="go_around")
    parser.add_argument("--model", default="logreg", choices=["logreg", "lda", "rf", "mlp"])
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_col = cfg.get("target", args.target)
    model_name = cfg.get("model", args.model)

    df = basic_clean(load_dataset(args.data), target_col=target_col)
    df = apply_feature_selection(df, cfg.get("features"), target_col=target_col)
    split = create_split(df, cfg, target_col=target_col)

    numeric_cols, categorical_cols = infer_columns(split.train, target_col)
    pre = build_preprocessor(numeric_cols, categorical_cols)
    model = get_model(model_name)
    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train = split.train.drop(columns=[target_col])
    y_train = split.train[target_col]
    X_val = split.val.drop(columns=[target_col])
    y_val = split.val[target_col]

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_proba)

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.outdir) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, outdir / "model.joblib")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({"run_id": run_id, **metrics}, indent=2))


if __name__ == "__main__":
    main()
