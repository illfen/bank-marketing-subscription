from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_marketing.preprocess import categorical_columns, numeric_columns


RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Pipeline


def make_preprocessor(frame: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    categorical = categorical_columns(frame)
    numeric = numeric_columns(frame)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )


def build_models(frame: pd.DataFrame, positive_weight: float) -> list[ModelSpec]:
    linear_preprocessor = make_preprocessor(frame, scale_numeric=True)
    tree_preprocessor = make_preprocessor(frame, scale_numeric=False)

    models = [
        ModelSpec(
            "Logistic Regression",
            Pipeline(
                [
                    ("preprocess", linear_preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            solver="lbfgs",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "Random Forest",
            Pipeline(
                [
                    ("preprocess", tree_preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=260,
                            max_depth=12,
                            min_samples_leaf=20,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "MLP",
            Pipeline(
                [
                    ("preprocess", linear_preprocessor),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            alpha=0.001,
                            batch_size=256,
                            learning_rate_init=0.001,
                            max_iter=120,
                            early_stopping=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]

    try:
        from xgboost import XGBClassifier

        xgb_model = XGBClassifier(
            n_estimators=360,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=positive_weight,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    except Exception:
        xgb_model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        )

    models.append(
        ModelSpec(
            "XGBoost",
            Pipeline(
                [
                    ("preprocess", tree_preprocessor),
                    ("model", xgb_model),
                ]
            ),
        )
    )

    return models
