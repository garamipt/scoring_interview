## Results

| Model | Val Gini | Test Gini | Test PR-AUC | Gini Nov | Gini Dec |
|---|---|---|---|---|---|
| `logreg_current` | 0.2582 | 0.2365 | 0.0796 | 0.2586 | 0.2187 |
| `nn_current` | 0.3399 | 0.3062 | 0.0922 | 0.3157 | 0.2984 |
| `gbdt_current` | 0.3411 | 0.3107 | 0.0936 | 0.3250 | 0.2988 |
| `nn_hist3` | 0.6046 | 0.5830 | 0.2077 | 0.5848 | 0.5814 |
| `gbdt_no_time` | 0.6072 | 0.5888 | 0.2043 | 0.5894 | 0.5881 |
| `gbdt_all` | 0.6109 | 0.5967 | 0.2171 | 0.6003 | 0.5937 |
| `nn_no_time` | 0.6334 | 0.6139 | 0.2345 | 0.6128 | 0.6148 |
| `nn_gru` | 0.6364 | 0.6166 | 0.2390 | 0.6153 | 0.6175 |
| `blend_nn_gbdt` | 0.6433 | 0.6258 | 0.2424 | 0.6258 | 0.6257 |

## Paired bootstrap on test (200 resamples)

| Comparison | Δ Gini | 95% CI | Significant |
|---|---|---|---|
| H1: history, GBDT: `gbdt_all` − `gbdt_current` | +0.2861 | [+0.2744, +0.2967] | yes |
| H1: history, NN: `nn_gru` − `nn_current` | +0.3104 | [+0.2992, +0.3213] | yes |
| H2: sequence NN vs GBDT on aggregates: `nn_gru` − `gbdt_all` | +0.0199 | [+0.0153, +0.0250] | yes |
| H3: timing aggregates, GBDT: `gbdt_all` − `gbdt_no_time` | +0.0080 | [+0.0052, +0.0110] | yes |
| H3: time deltas, NN: `nn_gru` − `nn_no_time` | +0.0026 | [-0.0007, +0.0058] | no |
| H4: full history vs last 3: `nn_gru` − `nn_hist3` | +0.0336 | [+0.0295, +0.0383] | yes |
| Blend vs NN: `blend_nn_gbdt` − `nn_gru` | +0.0092 | [+0.0077, +0.0106] | yes |
