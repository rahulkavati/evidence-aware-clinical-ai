# Selective prediction evaluation

Validation-only evaluation using patient-disjoint threshold-selection and evaluation cohorts.

- Maximum selection risk: `0.100`
- Minimum selection coverage: `0.500`
- Selected confidence threshold: `0.991607`
- Held-out test accessed: `False`

| Scope | Patients | Positive labels | Coverage | Abstention | Selective risk | Selective Brier | Retained AUROC | Retained AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled_all | 3025 | 2090 | 0.50589 | 0.49411 | 0.00260 | 0.00260 | 0.61847 | 0.00459 |
| hospital_A | 1492 | 1258 | 0.41967 | 0.58033 | 0.00221 | 0.00221 | 0.75711 | 0.00556 |
| hospital_B | 1533 | 832 | 0.59123 | 0.40877 | 0.00288 | 0.00287 | 0.55371 | 0.00455 |
| hour_6 | 2979 | 56 | 0.35683 | 0.64317 | 0.00470 | 0.00468 | 0.66881 | 0.02731 |
| hour_12 | 2948 | 39 | 0.51866 | 0.48134 | 0.00262 | 0.00262 | 0.36475 | 0.00246 |
| hour_24 | 2323 | 17 | 0.66251 | 0.33749 | 0.00130 | 0.00131 | 0.48569 | 0.00222 |

This is an exploratory retrospective evaluation. The confidence threshold is not a clinically validated decision threshold.
