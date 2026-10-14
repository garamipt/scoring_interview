import time

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scoring.config import FEATURES_NPZ
from scoring.features import Features, tabular
from scoring.metrics import report

SEED = 42
SPLITS = ("train", "val", "test")
GROUPS = {
    "current": {"cur"},
    "no_time": {"cur", "hist"},
    "all": {"cur", "hist", "time"},
}
RUNS = [("logreg", "current"), ("gbdt", "current"), ("gbdt", "no_time"), ("gbdt", "all")]


def column_group(name: str, agg_cols: set[str]) -> str:
    if name in agg_cols:
        return "time"
    if name.startswith(("any_", "last_")):
        return "hist"
    return "cur"


def make_model(kind: str):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=1000,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        scoring="roc_auc",
        n_iter_no_change=30,
        random_state=SEED,
    )


def main() -> None:
    f = Features.load(FEATURES_NPZ)
    X, names = tabular(f)
    agg_cols = set(f.agg_cols)
    rows = {s: f.split == s for s in SPLITS}
    y = {s: f.y[rows[s]] for s in SPLITS}
    month = {s: f.month[rows[s]] for s in ("val", "test")}

    for kind, variant in RUNS:
        cols = [i for i, n in enumerate(names) if column_group(n, agg_cols) in GROUPS[variant]]
        Xs = {s: X[rows[s]][:, cols] for s in SPLITS}

        t0 = time.perf_counter()
        model = make_model(kind)
        if kind == "gbdt":
            model.fit(Xs["train"], y["train"], X_val=Xs["val"], y_val=y["val"])
        else:
            model.fit(Xs["train"], y["train"])
        elapsed = time.perf_counter() - t0

        params = {
            "model": kind,
            "variant": variant,
            "n_features": len(cols),
            "fit_s": round(elapsed),
        }
        if kind == "gbdt":
            params["n_iter"] = int(model.n_iter_)
        p = {s: model.predict_proba(Xs[s])[:, 1] for s in ("val", "test")}
        report(f"{kind}_{variant}", params, y, p, month)


if __name__ == "__main__":
    main()
