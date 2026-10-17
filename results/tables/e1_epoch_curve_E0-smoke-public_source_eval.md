# E0-smoke-public: metrics by epoch (source_eval)

| epoch | step | n_clips | sim_target_norm ↑ | dcer ↓ | f0_rmse_cents ↓ | singmos ↑ | g/total (EMA) | g/mel (EMA) | d/adv (EMA) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 613 | 8 | 1.242 | +0.254 | 12.9 | 3.93 | 40.76 | 22.00 | 4.03 |
| 2 | 1226 | 8 | 1.235 | +0.156 | 13.4 | 3.93 | 40.18 | 21.60 | 4.00 |
| 3 | 1839 | 8 | **1.251** | +0.098 | 13.0 | **4.00** | 39.69 | 21.37 | 3.99 |
| 4 | 2452 | 8 | 1.224 | **+0.092** | 13.6 | 3.89 | 39.31 | 21.23 | 3.99 |
| 5 | 3065 | 8 | 1.238 | +0.150 | **12.8** | 3.82 | 38.95 | 21.10 | 4.00 |

- Inference setting: `f0-rmvpe_ir0.30_p0.33_ve1.00`. Bold = best epoch per metric (↑ higher / ↓ lower is better).
- Best epochs are selected on this evaluation set, so their scores are optimistic.
- Reproduce: `evaluation\env\python.exe scripts/report.py epoch-curve --run E0-smoke-public --set source_eval`
