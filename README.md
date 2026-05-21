# Bank Marketing Subscription Prediction

This repository contains the code and experiment outputs for predicting whether
a bank telemarketing customer will subscribe to a term deposit, using the UCI
Bank Marketing dataset.

The course report files are intentionally excluded from this repository.

## Contents

- `src/bank_marketing/`: data loading, preprocessing, model definitions,
  evaluation, and plotting utilities.
- `scripts/run_experiments.py`: end-to-end experiment runner.
- `tests/`: unit tests for preprocessing and evaluation logic.
- `outputs/`: generated metrics and figures from the completed experiment.
- `requirements.txt`: Python dependencies.

## Setup

Using conda:

```bash
conda create -y -p ./.conda python=3.10 pip
conda run -p ./.conda python -s -m pip install -r requirements.txt
```

Using an already activated Python environment:

```bash
python -m pip install -r requirements.txt
```

## Run Tests

```bash
conda run -p ./.conda python -s -m pytest tests -q
```

or:

```bash
python -m pytest tests -q
```

## Run Experiments

```bash
conda run -p ./.conda python -s scripts/run_experiments.py
```

or:

```bash
python scripts/run_experiments.py
```

The script downloads the UCI Bank Marketing dataset automatically into
`data/raw/`, trains the comparison models, and writes experiment artifacts to
`outputs/`.

## Outputs

- `outputs/metrics.csv`: model metrics including Accuracy, Precision, Recall,
  F1, ROC-AUC, and Average Precision.
- `outputs/dataset_profile.json`: basic dataset profile.
- `outputs/duration_leakage_check.csv`: comparison showing the leakage effect
  of using the `duration` field.
- `outputs/metric_comparison.png`: metric comparison chart.
- `outputs/roc_curves.png`: ROC curves.
- `outputs/best_confusion_matrix.png`: confusion matrix of the best model.

## Notes

The main experiment excludes the `duration` feature because it is only known
after the phone call and would cause data leakage for a pre-call prediction
task.
