import argparse
import subprocess
from pathlib import Path

from src.config import MODEL_DIR
from src.data.make_interim import make_interim
from src.data.make_splits import make_splits
from src.data.verify_local_data import verify_local_data
from src.evaluation.evaluate_models import evaluate_model
from src.models import common
from src.models.select_best_model import select_best_model


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI for go-around classification workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="Verify raw dataset is present locally")
    subparsers.add_parser("prepare", help="Build interim dataset from local raw data")

    split_parser = subparsers.add_parser("split", help="Create train/test splits")
    split_parser.add_argument("--target", default="target", help="Target column name")

    train_parser = subparsers.add_parser("train", help="Train the selected model")
    train_parser.add_argument("--model", default="logreg", choices=list(common.get_model_registry().keys()))
    train_parser.add_argument("--target", default="target", help="Target column name")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the saved model")
    eval_parser.add_argument("--target", default="target", help="Target column name")

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", default="8000")

    args = parser.parse_args()

    if args.command == "verify":
        files = verify_local_data()
        print(f"Found {len(files)} raw data file(s):")
        for f in files:
            print(f"- {f}")
    elif args.command == "prepare":
        output = make_interim()
        print(f"Interim dataset created at {output}")
    elif args.command == "split":
        train_path, test_path = make_splits(target=args.target)
        print(f"Train split: {train_path}\nTest split: {test_path}")
    elif args.command == "train":
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_script = Path(__file__).resolve().parent / "models" / f"train_{args.model}.py"
        subprocess.run(["python", str(model_script), "--target", args.target], check=True)
    elif args.command == "evaluate":
        metrics = evaluate_model(target=args.target)
        print("Evaluation results:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")
    elif args.command == "serve":
        subprocess.run(
            [
                "uvicorn",
                "app.main:app",
                "--host",
                args.host,
                "--port",
                str(args.port),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
