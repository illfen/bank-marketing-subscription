from __future__ import annotations

from typing import Iterable

import pandas as pd


def split_features_target(
    frame: pd.DataFrame,
    target_col: str = "y",
    include_duration: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split raw Bank Marketing data into feature matrix and binary target.

    The `duration` column is excluded by default because it is only known after
    a phone call has happened, so using it would leak post-contact information
    into a pre-contact marketing prediction model.
    """
    if target_col not in frame.columns:
        raise ValueError(f"target column {target_col!r} not found")

    drop_cols = [target_col]
    if not include_duration and "duration" in frame.columns:
        drop_cols.append("duration")

    features = frame.drop(columns=drop_cols).copy()
    target = frame[target_col].map({"no": 0, "yes": 1})

    if target.isna().any():
        bad_values = sorted(frame.loc[target.isna(), target_col].astype(str).unique())
        raise ValueError(f"unexpected target values: {bad_values}")

    return features, target.astype(int)


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_categorical_dtype(frame[column])
    ]


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    cats = set(categorical_columns(frame))
    return [column for column in frame.columns if column not in cats]


def describe_columns(columns: Iterable[str]) -> str:
    return ", ".join(columns)
