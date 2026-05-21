from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ConfusionCounts:
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> dict[str, int]:
        return {"tn": self.tn, "fp": self.fp, "fn": self.fn, "tp": self.tp}


def summarize_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 6),
    }


def confusion_counts(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> ConfusionCounts:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ConfusionCounts(int(tn), int(fp), int(fn), int(tp))


def choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> tuple[float, dict[str, float]]:
    thresholds = np.round(np.arange(start, stop + step / 2, step), 2)
    best_threshold = float(thresholds[0])
    best_metrics = summarize_predictions(y_true, y_score, best_threshold)

    for threshold in thresholds[1:]:
        metrics = summarize_predictions(y_true, y_score, float(threshold))
        if (metrics["f1"], metrics["recall"], threshold) > (
            best_metrics["f1"],
            best_metrics["recall"],
            best_threshold,
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics
