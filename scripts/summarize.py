import json

import numpy as np
from scipy.stats import rankdata

from scoring.config import FEATURES_NPZ, PREDS_DIR, RESULTS_JSONL, ROOT
from scoring.metrics import gini, report

SPLITS = ("val", "test")
EXCLUDE = {"nn_smoke"}
COMPARISONS = [
    ("gbdt_all", "gbdt_current", "H1: history, GBDT"),
    ("nn_gru", "nn_current", "H1: history, NN"),
    ("nn_gru", "gbdt_all", "H2: sequence NN vs GBDT on aggregates"),
    ("gbdt_all", "gbdt_no_time", "H3: timing aggregates, GBDT"),
    ("nn_gru", "nn_no_time", "H3: time deltas, NN"),
    ("nn_gru", "nn_hist3", "H4: full history vs last 3"),
    ("blend_nn_gbdt", "nn_gru", "Blend vs NN"),
]
OUT_MD = ROOT / "reports" / "experiments.md"


def load_targets() -> tuple[dict, dict]:
    with np.load(FEATURES_NPZ) as d:  # npz reads only requested arrays
        split, y, month = d["split"], d["y"], d["month"]
    return {s: y[split == s] for s in SPLITS}, {s: month[split == s] for s in SPLITS}


def load_preds(name: str) -> dict[str, np.ndarray]:
    with np.load(PREDS_DIR / f"{name}.npz") as d:
        return {s: d[f"{s}_pred"] for s in SPLITS}


def latest_runs() -> dict[str, dict]:
    runs = {}
    for line in RESULTS_JSONL.read_text().splitlines():
        record = json.loads(line)
        if record["name"] not in EXCLUDE:
            runs[record["name"]] = record  # a rerun overwrites the earlier record
    return runs


def rank(p: np.ndarray) -> np.ndarray:
    return rankdata(p) / len(p)


def blend(a: dict, b: dict, y: dict) -> tuple[float, dict]:
    """Rank average with the weight chosen on val."""
    weights = np.linspace(0, 1, 11)
    scores = [gini(y["val"], w * rank(a["val"]) + (1 - w) * rank(b["val"])) for w in weights]
    w = float(weights[int(np.argmax(scores))])
    return w, {s: w * rank(a[s]) + (1 - w) * rank(b[s]) for s in SPLITS}


def paired_bootstrap(
    y: np.ndarray, p1: np.ndarray, p2: np.ndarray, n_boot: int = 200, seed: int = 0
) -> tuple[float, float, float]:
    """Gini(p1) - Gini(p2) with a 95% CI; both models scored on the same resampled rows."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(gini(y[i], p1[i]) - gini(y[i], p2[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return gini(y, p1) - gini(y, p2), float(lo), float(hi)


def main() -> None:
    y, month = load_targets()
    runs = latest_runs()

    w, p_blend = blend(load_preds("nn_gru"), load_preds("gbdt_all"), y)
    runs["blend_nn_gbdt"] = report("blend_nn_gbdt", {"w_nn": w}, y, p_blend, month)

    lines = [
        "## Results",
        "",
        "| Model | Val Gini | Test Gini | Test PR-AUC | Gini Nov | Gini Dec |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in sorted(runs.items(), key=lambda kv: kv[1]["test"]["gini"]):
        by_month = {int(m): g for m, g in r["test"]["gini_by_month"].items()}
        lines.append(
            f"| `{name}` | {r['val']['gini']:.4f} | {r['test']['gini']:.4f} "
            f"| {r['test']['pr_auc']:.4f} | {by_month[11]:.4f} | {by_month[12]:.4f} |"
        )

    lines += [
        "",
        "## Paired bootstrap on test (200 resamples)",
        "",
        "| Comparison | Δ Gini | 95% CI | Significant |",
        "|---|---|---|---|",
    ]
    for a, b, label in COMPARISONS:
        diff, lo, hi = paired_bootstrap(y["test"], load_preds(a)["test"], load_preds(b)["test"])
        significant = "yes" if lo > 0 or hi < 0 else "no"
        lines.append(
            f"| {label}: `{a}` − `{b}` | {diff:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {significant} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
