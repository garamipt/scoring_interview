import keras
import numpy as np
from keras import layers, ops

from scoring.features import Features


def make_inputs(f: Features, use_history: bool, use_time: bool, max_hist: int) -> dict:
    x = {"cur": f.cur}
    if use_history:
        x["hist"] = f.hist[:, :max_hist]
        x["hist_mask"] = f.hist_mask[:, :max_hist]
        if use_time:
            x["hist_time"] = f.hist_time[:, :max_hist]
    return x


def build_model(
    n_cur: int,
    n_hist: int,
    *,
    use_history: bool = True,
    use_time: bool = True,
    max_hist: int = 9,
    dim: int = 64,
    dropout: float = 0.2,
    base_rate: float = 0.05,
) -> keras.Model:
    cur_in = keras.Input((n_cur,), dtype="uint8", name="cur")
    inputs = [cur_in]
    parts = [layers.Dense(dim, activation="relu", name="cur_dense")(ops.cast(cur_in, "float32"))]

    if use_history:
        hist_in = keras.Input((max_hist, n_hist), dtype="uint8", name="hist")
        mask_in = keras.Input((max_hist,), dtype="bool", name="hist_mask")
        inputs += [hist_in, mask_in]
        h = layers.Dense(dim, activation="relu", name="app_encoder")(ops.cast(hist_in, "float32"))
        if use_time:
            time_in = keras.Input((max_hist, 2), dtype="float32", name="hist_time")
            inputs.append(time_in)
            h = layers.Concatenate()([h, time_in])
        parts.append(layers.GRU(dim, name="history_gru")(h, mask=mask_in))

    z = layers.Concatenate()(parts) if len(parts) > 1 else parts[0]
    z = layers.Dense(dim, activation="relu")(z)
    z = layers.Dropout(dropout)(z)
    # Start from the base rate instead of 0.5: faster and more stable first epochs
    bias = keras.initializers.Constant(np.log(base_rate / (1 - base_rate)))
    out = layers.Dense(1, activation="sigmoid", bias_initializer=bias)(z)
    return keras.Model(inputs, out)
