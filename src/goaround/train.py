from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
from sklearn.pipeline import Pipeline

from goaround.data.io import load_dataset, basic_clean
from goaround.data.split import random_split
from goaround.features.pipeline import build_preprocessor
from goaround.models.factory import get_model
from goaround.eval.metrics import compute_metrics


def infer_columns(df, target_col: str):
    xdf = df.drop(columns=[target_col])
    numeric_cols = xdf.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in xdf.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="go_around")
    parser.add_argument("--model", default="logreg", choices=["logreg", "lda", "rf", "mlp"])
    parser.add_argument("--outdir", default="artifacts")
    args = parser.parse_args()

    df = basic_clean(load_dataset(args.data), target_col=args.target)
    split = random_split(df, target_col=args.target)

    numeric_cols, categorical_cols = infer_columns(split.train, args.target)
    pre = build_preprocessor(numeric_cols, categorical_cols)
    model = get_model(args.model)
    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train = split.train.drop(columns=[args.target])
    y_train = split.train[args.target]
    X_val = split.val.drop(columns=[args.target])
    y_val = split.val[args.target]

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_proba)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, outdir / "model.joblib")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
