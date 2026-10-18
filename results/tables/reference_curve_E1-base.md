# E1-base: paired scores against the real cover

| epoch | sim_ref ↑ | Δ vs source | mcd_ref ↓ | Δ vs source | f0_corr_ref |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.624 | +0.208 | 7.91 | +0.996 | 0.974 |
| 100 | 0.638 | +0.223 | 7.95 | +0.961 | 0.975 |
| 200 | 0.646 | +0.231 | 7.46 | +1.445 | 0.977 |
| 300 | 0.647 | +0.232 | 7.70 | +1.211 | 0.975 |
| 400 | **0.659** | +0.244 | 7.53 | +1.383 | 0.975 |
| 500 | 0.657 | +0.242 | **7.44** | +1.468 | 0.978 |

- Inference setting: `f0-rmvpe_ir0.30_p0.33_ve1.00`. Bold = best epoch. Δ is the improvement over the unconverted source (sim: converted − source, mcd: source − converted; positive = closer to the cover).
- Source baseline: sim_ref 0.415, mcd_ref 8.91; cover self-similarity (LOO centroid) 0.686. The self value is centroid-based (window vs the other cover windows); its like-for-like counterpart is sim_ref_centroid in runs/<run>/reference/metrics.csv, not the paired sim_ref.
- f0_corr_ref is information only: the cover is a different performance, so pitch differences are expected.
- Reproduce: `evaluation\env\python.exe scripts/report.py reference-curve --run E1-base`
