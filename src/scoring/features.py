from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl

from scoring.config import FEATURE_COLS, MAX_HIST, SPLIT_BY_MONTH, WINDOWS

ROW_FIELDS = ("idx", "y", "split", "month", "cur", "hist", "hist_mask", "hist_time", "agg")


@dataclass
class Features:
    idx: np.ndarray  # [N] application id
    y: np.ndarray  # [N] target
    split: np.ndarray  # [N] "train" / "val" / "test"
    month: np.ndarray  # [N] month of the current application
    cur: np.ndarray  # [N, n_cur] current application, non-constant columns
    hist: np.ndarray  # [N, MAX_HIST, n_hist] previous applications, NUM=1 in slot 0
    hist_mask: np.ndarray  # [N, MAX_HIST] True for real previous applications
    hist_time: np.ndarray  # [N, MAX_HIST, 2] log1p(days before current), log1p(gap to newer app)
    agg: np.ndarray  # [N, n_agg] timing aggregates of the history
    cur_cols: np.ndarray
    hist_cols: np.ndarray
    agg_cols: np.ndarray

    def subset(self, split: str) -> Self:
        rows = self.split == split
        return type(self)(
            **{
                f.name: getattr(self, f.name)[rows]
                if f.name in ROW_FIELDS
                else getattr(self, f.name)
                for f in fields(self)
            }
        )

    def save(self, path: Path) -> None:
        np.savez(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: Path) -> Self:
        with np.load(path) as data:
            return cls(**{f.name: data[f.name] for f in fields(cls)})


def non_constant_columns(df: pl.DataFrame, cols: list[str]) -> list[str]:
    n_unique = df.select(pl.col(cols).n_unique()).row(0, named=True)
    return [c for c in cols if n_unique[c] > 1]


def _days(later: pl.Expr, earlier: pl.Expr) -> pl.Expr:
    return (later - earlier).dt.total_days(fractional=True)


def build_aggregates(history: pl.DataFrame) -> pl.DataFrame:
    """Timing aggregates per client, one row per `pos`."""
    days = pl.col("days_before_current")
    return history.group_by("pos").agg(
        *[(days <= w).sum().alias(f"n_prev_{w}d") for w in WINDOWS],
        n_prev=pl.len(),
        days_since_prev=days.min(),
        days_since_oldest=days.max(),
        min_gap=pl.col("gap_days").min(),
    )


def build_features(
    apps: pl.DataFrame,
    labels: pl.DataFrame,
    cur_cols: list[str] | None = None,
    hist_cols: list[str] | None = None,
) -> Features:
    clients = labels.sort("IDX").with_row_index("pos")
    apps = (
        apps.join(clients.select("IDX", "pos"), on="IDX", how="inner")
        .sort("pos", "NUM")
        .with_columns(
            cur_ts=pl.col("ts").first().over("pos"),  # NUM=0 comes first after sort
            newer_ts=pl.col("ts").shift(1).over("pos"),  # application with NUM-1
        )
        .with_columns(
            days_before_current=_days(pl.col("cur_ts"), pl.col("ts")),
            gap_days=_days(pl.col("newer_ts"), pl.col("ts")),
        )
    )
    current = apps.filter(pl.col("NUM") == 0)
    history = apps.filter(pl.col("NUM") > 0)

    n = clients.height
    assert (current["pos"].to_numpy() == np.arange(n)).all(), "one current app per client"

    cur_cols = cur_cols or non_constant_columns(current, FEATURE_COLS)
    hist_cols = hist_cols or non_constant_columns(history, FEATURE_COLS)

    # Scatter history rows into fixed-size tensors: client -> row, NUM -> slot
    pos = history["pos"].to_numpy()
    slot = history["NUM"].to_numpy().astype(np.intp) - 1

    hist = np.zeros((n, MAX_HIST, len(hist_cols)), dtype=np.uint8)
    hist[pos, slot] = history.select(hist_cols).to_numpy()
    hist_mask = np.zeros((n, MAX_HIST), dtype=bool)
    hist_mask[pos, slot] = True
    hist_time = np.zeros((n, MAX_HIST, 2), dtype=np.float32)
    hist_time[pos, slot] = np.log1p(history.select("days_before_current", "gap_days").to_numpy())

    agg = (
        clients.select("pos")
        .join(build_aggregates(history), on="pos", how="left")
        .sort("pos")
        .drop("pos")
    )
    month = current["ts"].dt.month()

    return Features(
        idx=clients["IDX"].to_numpy(),
        y=clients["TRG"].to_numpy(),
        split=month.replace_strict(SPLIT_BY_MONTH).to_numpy().astype(str),
        month=month.to_numpy(),
        cur=current.select(cur_cols).to_numpy(),
        hist=hist,
        hist_mask=hist_mask,
        hist_time=hist_time,
        agg=agg.to_numpy().astype(np.float32),
        cur_cols=np.array(cur_cols),
        hist_cols=np.array(hist_cols),
        agg_cols=np.array(agg.columns),
    )


def tabular(f: Features) -> tuple[np.ndarray, list[str]]:
    """Flat features for non-sequential baselines: current app + any/last previous + timing."""
    X = np.hstack([f.cur, f.hist.max(axis=1), f.hist[:, 0], f.agg]).astype(np.float32)
    names = [
        *f.cur_cols,
        *(f"any_{c}" for c in f.hist_cols),
        *(f"last_{c}" for c in f.hist_cols),
        *f.agg_cols,
    ]
    return X, names
