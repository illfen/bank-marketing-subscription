import pandas as pd

from bank_marketing.preprocess import split_features_target


def test_split_features_target_drops_target_and_duration_by_default():
    frame = pd.DataFrame(
        {
            "age": [35, 48],
            "job": ["admin.", "technician"],
            "duration": [120, 240],
            "y": ["yes", "no"],
        }
    )

    features, target = split_features_target(frame)

    assert list(features.columns) == ["age", "job"]
    assert target.tolist() == [1, 0]


def test_split_features_target_can_keep_duration_for_leakage_check():
    frame = pd.DataFrame(
        {
            "age": [35, 48],
            "duration": [120, 240],
            "y": ["no", "yes"],
        }
    )

    features, target = split_features_target(frame, include_duration=True)

    assert list(features.columns) == ["age", "duration"]
    assert target.tolist() == [0, 1]
