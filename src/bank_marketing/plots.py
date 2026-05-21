from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix


def save_metric_bar(metrics: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = metrics.set_index("model")[["roc_auc", "f1", "recall", "precision"]]
    ax = plot_frame.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), width=0.78)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance on Bank Marketing Test Set")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_roc_curves(curve_data: list[tuple[str, object, object, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, ax = plt.subplots(figsize=(7, 6))
    for model_name, estimator, x_test, y_test in curve_data:
        RocCurveDisplay.from_estimator(estimator, x_test, y_test, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC Curves")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_confusion_matrix(y_true, y_score, threshold: float, model_name: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=["no", "yes"])
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
