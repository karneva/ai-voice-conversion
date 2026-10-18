# Ablation vs E1-base (source_eval)

| run | epoch | n_clips | sim_target_norm ↑ | sim_target ↑ | sim_source ↓ | dcer ↓ | f0_rmse_cents ↓ | octave_err ↓ | vuv_err ↓ | gpe50 ↓ | fpc ↑ | singmos ↑ | utmos ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **E1-base** (baseline) | 500 | 18 | 1.125 ±0.052 | 0.787 ±0.020 | 0.382 ±0.080 | +0.145 ±0.084 | 35.8 ±19.2 | 0.002 ±0.002 | 0.062 ±0.010 | 0.037 ±0.018 | 0.988 ±0.010 | 4.13 ±0.23 | 1.47 ±0.09 |
| E4-no-masked | 340 | 18 | 1.129 (+0.003 ±0.036 ▲) | 0.788 (+0.001 ±0.014 ▲) | 0.379 (-0.003 ±0.022 ▲) | +0.100 (-0.044 ±0.071 ▲) | 32.2 (-3.5 ±4.6 ▲) | 0.001 (=) | 0.065 (+0.004 ±0.011 ▼) | 0.032 (-0.004 ±0.008 ▲) | 0.991 (+0.003 ±0.004 ▲) | 4.13 (+0.01 ±0.20 ▲) | 1.53 (+0.07 ±0.07 ▲) |
| Ref: target held-out (ceiling, LOO) |  |  | 1.000 | 0.738 | – | – | – | – | – | – | – | – | – |
| Ref: other singers (floor) |  |  | 0.000 | 0.347 | – | – | – | – | – | – | – | – | – |
| Ref: unconverted source vocals |  |  | 0.000 | 0.347 | 1.000 | +0.000 | 0.0 | 0.000 | 0.000 | 0.000 | 1.000 | 4.06 | 1.64 |

- Epoch: each run's best epoch by `sim_target_norm` on heldout, reported on source_eval.
- Inference setting: `f0-rmvpe_ir0.30_p0.33_ve1.00`.
- Cells show value (delta vs baseline). ▲ = better, ▼ = worse, (=) = unchanged at the shown precision, judged by each metric's direction (↑ higher / ↓ lower is better). ± = 95% CI over clips (paired by clip for deltas) when per-clip metrics exist: t-based for clip means, clip bootstrap for the corpus CER/ΔCER.
- Reference rows: target bounds from the calibration file; source vocals = the unconverted input (ΔCER is 0 by definition, MOS columns are source scores).
- CER reference: some clips have no lyrics, so their reference is Whisper's transcript of the source. For those clips cer_source is 0 and ΔCER is CER against the source ASR, not against the lyrics.
- Reproduce: `evaluation\env\python.exe scripts/report.py compare --baseline E1-base --runs E4-no-masked --set source_eval`
