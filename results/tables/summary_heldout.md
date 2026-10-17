# All runs at their best epoch (heldout)

| run | track | best epoch | step | epochs evaluated | n_clips | sim_target_norm ↑ | sim_target ↑ | dcer ↓ | f0_rmse_cents ↓ | octave_err ↓ | vuv_err ↓ | gpe50 ↓ | fpc ↑ | singmos ↑ | utmos ↑ | mcd ↓ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0-smoke-public | public | 3 | 1839 | 5 | 30 | 1.102 | 0.766 | +0.103 | 12.5 | 0.000 | 0.023 | 0.014 | 0.999 | 4.12 | 1.45 | 5.88 |
| Ref: target held-out (ceiling, LOO) | public |  |  |  |  | 1.000 | 0.715 | – | – | – | – | – | – | – | – | – |
| Ref: other singers (floor) | public |  |  |  |  | 0.000 | 0.215 | – | – | – | – | – | – | – | – | – |
| Ref: unconverted target vocals | public |  |  |  |  | 1.000 | 0.715 | +0.000 | 0.0 | 0.000 | 0.000 | 0.000 | 1.000 | 4.25 | 1.45 | 0.00 |

- Best epoch: each run's best epoch by `sim_target_norm` on this set (heldout). This selects the checkpoint on the evaluation set itself, so the scores are optimistic; pass --select-set with the other set for unbiased numbers.
- Inference setting: `f0-rmvpe_ir0.30_p0.33_ve1.00`.
- ↑ higher is better, ↓ lower is better.
- Reproduce: `evaluation\env\python.exe scripts/report.py summary --set heldout`
