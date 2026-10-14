import json
from datetime import datetime

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scoring.config import PREDS_DIR, RESULTS_JSONL


def gini(y: np.ndarray, p: np.ndarray) -> float:
    return float(2 * roc_auc_score(y, p) - 1)


def evaluate(y: np.ndarray, p: np.ndarray, month: np.ndarray) -> dict:
    return {
        "gini": gini(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "gini_by_month": {int(m): gini(y[month == m], p[month == m]) for m in np.unique(month)},
    }


def report(
    name: str,
    params: dict,
    y: dict[str, np.ndarray],
    p: dict[str, np.ndarray],
    month: dict[str, np.ndarray],
) -> dict:
    """Evaluate predictions per split, append to the results log and save predictions."""
    record = {
        "name": name,
        "time": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        **{split: evaluate(y[split], p[split], month[split]) for split in p},
    }
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PREDS_DIR / f"{name}.npz", **{f"{split}_pred": p[split] for split in p})

    print(
        f"{name:20s} val gini={record['val']['gini']:.4f}  test gini={record['test']['gini']:.4f}"
    )
    return record
