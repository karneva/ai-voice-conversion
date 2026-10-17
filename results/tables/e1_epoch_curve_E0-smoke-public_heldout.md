# E0-smoke-public: metrics by epoch (heldout)

| epoch | step | n_clips | sim_target_norm ↑ | dcer ↓ | f0_rmse_cents ↓ | singmos ↑ | g/total (EMA) | g/mel (EMA) | d/adv (EMA) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 613 | 30 | 1.089 | +0.489 | 13.7 | 4.00 | 40.76 | 22.00 | 4.03 |
| 2 | 1226 | 30 | 1.080 | +0.177 | 13.4 | 4.10 | 40.18 | 21.60 | 4.00 |
| 3 | 1839 | 30 | **1.102** | +0.103 | 12.5 | 4.12 | 39.69 | 21.37 | 3.99 |
| 4 | 2452 | 30 | 1.045 | **+0.057** | 13.0 | **4.22** | 39.31 | 21.23 | 3.99 |
| 5 | 3065 | 30 | 1.069 | +0.108 | **12.4** | 4.09 | 38.95 | 21.10 | 4.00 |

- Inference setting: `f0-rmvpe_ir0.30_p0.33_ve1.00`. Bold = best epoch per metric (↑ higher / ↓ lower is better).
- Best epochs are selected on this evaluation set, so their scores are optimistic.
- Reproduce: `evaluation\env\python.exe scripts/report.py epoch-curve --run E0-smoke-public --set heldout`
