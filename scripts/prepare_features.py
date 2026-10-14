import time

import polars as pl

from scoring.config import APPS_PARQUET, FEATURES_NPZ, LABELS_PARQUET
from scoring.features import build_features


def main() -> None:
    t0 = time.perf_counter()
    feats = build_features(pl.read_parquet(APPS_PARQUET), pl.read_parquet(LABELS_PARQUET))
    feats.save(FEATURES_NPZ)
    print(f"built in {time.perf_counter() - t0:.0f}s -> {FEATURES_NPZ}")

    for name in ("cur", "hist", "hist_mask", "hist_time", "agg"):
        arr = getattr(feats, name)
        print(f"{name:10s} {arr.shape} {arr.dtype} {arr.nbytes / 2**20:.0f} MB")
    for split in ("train", "val", "test"):
        rows = feats.split == split
        print(f"{split:5s} n={rows.sum():_} target={feats.y[rows].mean():.4f}")


if __name__ == "__main__":
    main()
