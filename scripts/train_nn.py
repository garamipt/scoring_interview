import argparse
import time

import keras

from scoring.config import ARTIFACTS_DIR, FEATURES_NPZ
from scoring.features import Features
from scoring.metrics import report
from scoring.models import build_model, make_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="nn_gru")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--no-time", action="store_true")
    parser.add_argument("--max-hist", type=int, default=9)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    f = Features.load(FEATURES_NPZ)
    data = {s: f.subset(s) for s in ("train", "val", "test")}
    del f
    flags = {
        "use_history": not args.no_history,
        "use_time": not args.no_time,
        "max_hist": args.max_hist,
    }
    x = {s: make_inputs(d, **flags) for s, d in data.items()}

    model = build_model(
        n_cur=data["train"].cur.shape[1],
        n_hist=data["train"].hist.shape[2],
        dim=args.dim,
        dropout=args.dropout,
        base_rate=float(data["train"].y.mean()),
        **flags,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    model.summary()

    t0 = time.perf_counter()
    history = model.fit(
        x["train"],
        data["train"].y,
        validation_data=(x["val"], data["val"].y),
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=3, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", factor=0.5, patience=1
            ),
        ],
        verbose=2,
    )
    elapsed = time.perf_counter() - t0

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACTS_DIR / f"{args.name}.keras")

    p = {s: model.predict(x[s], batch_size=8192, verbose=0).ravel() for s in ("val", "test")}
    params = {
        **vars(args),
        "fit_s": round(elapsed),
        "epochs_run": len(history.history["loss"]),
        "n_params": model.count_params(),
    }
    report(
        args.name,
        params,
        y={s: data[s].y for s in ("val", "test")},
        p=p,
        month={s: data[s].month for s in ("val", "test")},
    )


if __name__ == "__main__":
    main()
