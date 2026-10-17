# Ablation vs E0-smoke-public (source_eval)

| run | epoch | n_clips | sim_target_norm ↑ | sim_target ↑ | sim_source ↓ | dcer ↓ | f0_rmse_cents ↓ | octave_err ↓ | vuv_err ↓ | gpe50 ↓ | fpc ↑ | singmos ↑ | utmos ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **E0-smoke-public** (baseline) | 3 | 8 | 1.251 ±0.038 | 0.841 ±0.019 | 0.207 ±0.043 | +0.098 ±0.101 | 13.0 ±1.5 | 0.000 ±0.000 | 0.024 ±0.004 | 0.015 ±0.005 | 0.999 ±0.001 | 4.00 ±0.24 | 1.40 ±0.03 |
| Ref: target held-out (ceiling, LOO) |  |  | 1.000 | 0.715 | – | – | – | – | – | – | – | – | – |
| Ref: other singers (floor) |  |  | 0.000 | 0.215 | – | – | – | – | – | – | – | – | – |
| Ref: unconverted source vocals |  |  | 0.000 | 0.215 | 1.000 | +0.000 | 0.0 | 0.000 | 0.000 | 0.000 | 1.000 | 4.11 | 1.49 |

- Epoch: each run's best epoch by `sim_target_norm` on heldout, reported on source_eval.
- Inference setting: `f0-rmvpe_ir0.30_p0.33_ve1.00`.
- Cells show value (delta vs baseline). ▲ = better, ▼ = worse, (=) = unchanged at the shown precision, judged by each metric's direction (↑ higher / ↓ lower is better). ± = 95% CI over clips (paired by clip for deltas) when per-clip metrics exist: t-based for clip means, clip bootstrap for the corpus CER/ΔCER.
- Reference rows: target bounds from the calibration file; source vocals = the unconverted input (ΔCER is 0 by definition, MOS columns are source scores).
- Reproduce: `evaluation\env\python.exe scripts/report.py compare --baseline E0-smoke-public --runs E0-smoke-public`
