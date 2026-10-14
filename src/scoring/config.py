from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
APPS_PARQUET = PROCESSED_DIR / "applications.parquet"
LABELS_PARQUET = PROCESSED_DIR / "labels.parquet"
FEATURES_NPZ = PROCESSED_DIR / "features.npz"

FEATURE_COLS = [f"COL{i}" for i in range(1, 159)]
MAX_HIST = 9  # up to 9 previous applications
WINDOWS = (1, 7, 30, 90, 180, 365)
SPLIT_BY_MONTH = {**dict.fromkeys(range(1, 10), "train"), 10: "val", 11: "test", 12: "test"}

ARTIFACTS_DIR = ROOT / "artifacts"
PREDS_DIR = ARTIFACTS_DIR / "preds"
RESULTS_JSONL = ROOT / "reports" / "experiments.jsonl"
