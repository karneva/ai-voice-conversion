# All runs at their best epoch (source_eval)

| run | track | best epoch | step | epochs evaluated | n_clips | sim_target_norm ↑ | sim_target ↑ | sim_source ↓ | dcer ↓ | f0_rmse_cents ↓ | octave_err ↓ | vuv_err ↓ | gpe50 ↓ | fpc ↑ | singmos ↑ | utmos ↑ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0-smoke-public | public | 3 | 1839 | 5 | 8 | 1.251 | 0.841 | 0.207 | +0.098 | 13.0 | 0.000 | 0.024 | 0.015 | 0.999 | 4.00 | 1.40 |
| Ref: target held-out (ceiling, LOO) | public |  |  |  |  | 1.000 | 0.715 | – | – | – | – | – | – | – | – | – |
| Ref: other singers (floor) | public |  |  |  |  | 0.000 | 0.215 | – | – | – | – | – | – | – | – | – |
| Ref: unconverted source vocals | public |  |  |  |  | 0.000 | 0.215 | 1.000 | +0.000 | 0.0 | 0.000 | 0.000 | 0.000 | 1.000 | 4.11 | 1.49 |

- Best epoch: each run's best epoch by `sim_target_norm` on heldout, reported on source_eval.
- Inference setting: `f0-rmvpe_ir0.30_p0.33_ve1.00`.
- ↑ higher is better, ↓ lower is better.
- Reproduce: `evaluation\env\python.exe scripts/report.py summary --set source_eval`
