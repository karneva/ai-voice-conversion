# E4-no-masked: paired scores against the real cover

| epoch | sim_ref ↑ | Δ vs source | mcd_ref ↓ | Δ vs source | f0_corr_ref |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.634 | +0.219 | 7.54 | +1.368 | 0.978 |
| 100 | 0.656 | +0.241 | 7.20 | +1.712 | 0.977 |
| 200 | 0.661 | +0.246 | 7.26 | +1.653 | 0.976 |
| 300 | 0.657 | +0.242 | 7.11 | +1.799 | 0.976 |
| 400 | **0.662** | +0.247 | 7.06 | +1.851 | 0.974 |
| 500 | 0.658 | +0.243 | **7.04** | +1.866 | 0.977 |

- Inference setting: `f0-rmvpe_ir0.30_p0.33_ve1.00`. Bold = best epoch. Δ is the improvement over the unconverted source (sim: converted − source, mcd: source − converted; positive = closer to the cover).
- Source baseline: sim_ref 0.415, mcd_ref 8.91; cover self-similarity (LOO centroid) 0.686. The self value is centroid-based (window vs the other cover windows); its like-for-like counterpart is sim_ref_centroid in runs/<run>/reference/metrics.csv, not the paired sim_ref.
- f0_corr_ref is information only: the cover is a different performance, so pitch differences are expected.
- Reproduce: `evaluation\env\python.exe scripts/report.py reference-curve --run E4-no-masked`
