from datetime import datetime

import numpy as np
import polars as pl
import pytest

from scoring.features import Features, build_features, tabular

COLS = ["COL1", "COL2", "COL3"]


@pytest.fixture
def feats() -> Features:
    apps = pl.DataFrame(
        {
            "IDX": [20, 20, 10, 10, 10],
            "NUM": [1, 0, 0, 1, 2],
            "ts": [
                datetime(2018, 11, 4, 12),
                datetime(2018, 11, 5),
                datetime(2018, 3, 10),
                datetime(2018, 3, 9),
                datetime(2018, 2, 7),
            ],
            "COL1": [1, 0, 1, 0, 1],
            "COL2": [0, 1, 0, 1, 0],
            "COL3": [1, 1, 0, 0, 1],
        },
        schema_overrides={"NUM": pl.UInt8, **dict.fromkeys(COLS, pl.UInt8)},
    )
    labels = pl.DataFrame({"IDX": [20, 10], "TRG": [1, 0]}, schema_overrides={"TRG": pl.UInt8})
    return build_features(apps, labels, cur_cols=COLS, hist_cols=COLS)


def agg_col(f: Features, name: str) -> np.ndarray:
    return f.agg[:, list(f.agg_cols).index(name)]


def test_rows_are_aligned_by_idx(feats: Features) -> None:
    np.testing.assert_array_equal(feats.idx, [10, 20])
    np.testing.assert_array_equal(feats.y, [0, 1])
    np.testing.assert_array_equal(feats.split, ["train", "test"])
    np.testing.assert_array_equal(feats.cur, [[1, 0, 0], [0, 1, 1]])


def test_history_slots_and_mask(feats: Features) -> None:
    np.testing.assert_array_equal(feats.hist[0, :2], [[0, 1, 0], [1, 0, 1]])  # NUM=1, NUM=2
    np.testing.assert_array_equal(feats.hist[1, 0], [1, 0, 1])
    assert feats.hist[0, 2:].sum() == 0 and feats.hist[1, 1:].sum() == 0
    np.testing.assert_array_equal(feats.hist_mask.sum(axis=1), [2, 1])
    assert feats.hist_mask[0, :2].all() and feats.hist_mask[1, 0]


def test_time_features(feats: Features) -> None:
    days = np.expm1(feats.hist_time)  # undo log1p
    np.testing.assert_allclose(days[0, 0], [1, 1])  # 9 Mar vs 10 Mar
    np.testing.assert_allclose(days[0, 1], [31, 30])  # 7 Feb: 31d before current, 30d before 9 Mar
    np.testing.assert_allclose(days[1, 0], [0.5, 0.5])  # 12 hours


def test_aggregates(feats: Features) -> None:
    np.testing.assert_array_equal(agg_col(feats, "n_prev"), [2, 1])
    np.testing.assert_allclose(agg_col(feats, "days_since_prev"), [1, 0.5])
    np.testing.assert_array_equal(agg_col(feats, "n_prev_30d"), [1, 1])
    np.testing.assert_allclose(agg_col(feats, "min_gap"), [1, 0.5])


def test_tabular_any_and_last(feats: Features) -> None:
    X, names = tabular(feats)
    assert X.shape[1] == len(names)
    assert X[0, names.index("any_COL1")] == 1  # COL1 = 1 in NUM=2
    assert X[0, names.index("last_COL1")] == 0  # but 0 in NUM=1
