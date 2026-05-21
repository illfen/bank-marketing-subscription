from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bank_marketing.data import dataset_profile, load_bank_marketing
from bank_marketing.evaluate import choose_threshold, confusion_counts, summarize_predictions
from bank_marketing.models import RANDOM_STATE, build_models
from bank_marketing.plots import save_confusion_matrix, save_metric_bar, save_roc_curves
from bank_marketing.preprocess import split_features_target


OUTPUT_DIR = PROJECT_ROOT / "outputs"


def predict_scores(estimator, features: pd.DataFrame):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    decision = estimator.decision_function(features)
    return 1 / (1 + pd.Series(-decision).map(lambda value: pow(2.718281828, value)))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_bank_marketing()
    (OUTPUT_DIR / "dataset_profile.json").write_text(
        json.dumps(dataset_profile(frame), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    features, target = split_features_target(frame, include_duration=False)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )

    positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    results = []
    curve_data = []
    fitted = {}

    for spec in build_models(x_train, positive_weight=positive_weight):
        start = time.perf_counter()
        estimator = spec.estimator
        estimator.fit(x_train, y_train)
        elapsed = time.perf_counter() - start

        valid_scores = predict_scores(estimator, x_valid)
        threshold, valid_metrics = choose_threshold(y_valid.to_numpy(), valid_scores)
        test_scores = predict_scores(estimator, x_test)
        test_metrics = summarize_predictions(y_test.to_numpy(), test_scores, threshold)
        counts = confusion_counts(y_test.to_numpy(), test_scores, threshold).as_dict()

        row = {
            "model": spec.name,
            "threshold": round(threshold, 2),
            "valid_f1": valid_metrics["f1"],
            "average_precision": round(
                float(average_precision_score(y_test, test_scores)), 6
            ),
            "fit_seconds": round(float(elapsed), 3),
        }
        row.update(test_metrics)
        row.update(counts)
        results.append(row)
        fitted[spec.name] = (estimator, test_scores, threshold)
        curve_data.append((spec.name, estimator, x_test, y_test))
        print(f"finished {spec.name}: AUC={row['roc_auc']}, F1={row['f1']}")

    metrics = pd.DataFrame(results).sort_values(
        ["roc_auc", "f1"], ascending=False
    )
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    save_metric_bar(metrics, OUTPUT_DIR / "metric_comparison.png")
    save_roc_curves(curve_data, OUTPUT_DIR / "roc_curves.png")

    best_model = metrics.iloc[0]["model"]
    estimator, test_scores, threshold = fitted[best_model]
    save_confusion_matrix(
        y_test.to_numpy(),
        test_scores,
        threshold,
        best_model,
        OUTPUT_DIR / "best_confusion_matrix.png",
    )

    leakage_features, leakage_target = split_features_target(frame, include_duration=True)
    leakage_train, leakage_test, leakage_y_train, leakage_y_test = train_test_split(
        leakage_features,
        leakage_target,
        test_size=0.2,
        stratify=leakage_target,
        random_state=RANDOM_STATE,
    )
    leakage_model = build_models(leakage_train, positive_weight=positive_weight)[0].estimator
    leakage_model.fit(leakage_train, leakage_y_train)
    leakage_scores = predict_scores(leakage_model, leakage_test)
    leakage_metrics = summarize_predictions(leakage_y_test.to_numpy(), leakage_scores, 0.5)
    pd.DataFrame(
        [
            {
                "setting": "without_duration_primary_experiment",
                "roc_auc": float(metrics.loc[metrics["model"] == "Logistic Regression", "roc_auc"].iloc[0]),
            },
            {
                "setting": "with_duration_leakage_check",
                "roc_auc": leakage_metrics["roc_auc"],
            },
        ]
    ).to_csv(OUTPUT_DIR / "duration_leakage_check.csv", index=False)


if __name__ == "__main__":
    main()
