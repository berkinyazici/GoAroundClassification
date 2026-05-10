from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import MODEL_DIR
from src.data.download_data import download_data
from src.data.make_interim import make_interim
from src.data.make_splits import make_splits
from src.data.verify_local_data import verify_local_data
from src.evaluation.evaluate_models import evaluate_model
from src.models.select_best_model import select_best_model
from src.models.train_model import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI for go-around classification workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download-data", help="Download Zenodo dataset files into data/raw")
    subparsers.add_parser("verify", help="Verify raw dataset files are present locally")
    subparsers.add_parser("make-interim", help="Convert raw CSV.GZ files to cleaned parquet")
    subparsers.add_parser("prepare", help="Alias for make-interim")

    split_parser = subparsers.add_parser("make-splits", help="Create train/validation/test splits")
    split_parser.add_argument("--target", default="target")
    split_parser.add_argument("--sample-frac", type=float, default=None)
    split_parser.add_argument("--split-mode", choices=["time", "random"], default="time")
    subparsers.add_parser("split", help="Alias for make-splits")

    train_parser = subparsers.add_parser("train", help="Train a selected model")
    train_parser.add_argument("--model", default="logreg", choices=["logreg", "lda", "tree", "rf", "mlp", "lightgbm"])
    train_parser.add_argument("--feature-set", default="adsb_plus_metar", choices=["adsb_only", "metar_only", "adsb_plus_metar"])
    train_parser.add_argument("--target", default="target")

    subparsers.add_parser("select-best", help="Select the best trained model by test PR-AUC")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the deployed best model")
    eval_parser.add_argument("--target", default="target")

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", default="8000")

    args = parser.parse_args()

    if args.command == "download-data":
        download_data()
    elif args.command == "verify":
        files = verify_local_data(strict=False)
        print(f"Found {len(files)} expected raw data file(s).")
    elif args.command in {"make-interim", "prepare"}:
        output = make_interim()
        print(f"Interim dataset created at {output}")
    elif args.command in {"make-splits", "split"}:
        train_path, valid_path, test_path = make_splits(
            target=getattr(args, "target", "target"),
            sample_frac=getattr(args, "sample_frac", None),
            split_mode=getattr(args, "split_mode", "time"),
        )
        print(f"Train: {train_path}\nValidation: {valid_path}\nTest: {test_path}")
    elif args.command == "train":
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        train_model(model_name=args.model, feature_set=args.feature_set)
    elif args.command == "select-best":
        select_best_model()
    elif args.command == "evaluate":
        metrics = evaluate_model(target=args.target)
        print("Evaluation results:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")
    elif args.command == "serve":
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", args.host, "--port", str(args.port)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )


if __name__ == "__main__":
    main()
