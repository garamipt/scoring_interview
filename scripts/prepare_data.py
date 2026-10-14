import polars as pl

from scoring.config import APPS_PARQUET, FEATURE_COLS, LABELS_PARQUET, ROOT

RAW_DIR = ROOT / "data" / "sample_train_data"
APPS_SCHEMA = {
    "IDX": pl.Int64,
    "NUM": pl.UInt8,
    "COL0": pl.String,
    **dict.fromkeys(FEATURE_COLS, pl.UInt8),
}

if __name__ == "__main__":
    APPS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pl.scan_csv(RAW_DIR / "sample_train_data.csv", separator=";", schema=APPS_SCHEMA).select(
        "IDX",
        "NUM",
        pl.col("COL0").str.to_datetime("%Y.%m.%d %H:%M:%S").alias("ts"),
        *FEATURE_COLS,
    ).sink_parquet(APPS_PARQUET)
    pl.scan_csv(
        RAW_DIR / "sample_train_labels.csv",
        separator=";",
        schema={"IDX": pl.Int64, "TRG": pl.UInt8},
    ).sink_parquet(LABELS_PARQUET)
