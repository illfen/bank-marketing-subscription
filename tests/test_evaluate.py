import numpy as np

from bank_marketing.evaluate import choose_threshold, summarize_predictions


def test_choose_threshold_maximizes_f1_on_validation_scores():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.10, 0.40, 0.35, 0.80])

    threshold, metrics = choose_threshold(y_true, y_score)

    assert threshold == 0.35
    assert metrics["f1"] == 0.8


def test_summarize_predictions_returns_core_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.10, 0.20, 0.70, 0.90])

    metrics = summarize_predictions(y_true, y_score, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
