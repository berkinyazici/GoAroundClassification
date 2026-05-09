from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import REPORT_DIR


def plot_confusion_matrix(confusion_matrix, labels, output_name: str = "confusion_matrix.png") -> Path:
    output_dir = REPORT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    output_path = output_dir / output_name
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path
