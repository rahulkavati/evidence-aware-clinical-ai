# Selective prediction evaluation

Validation-only evaluation using patient-disjoint threshold-selection and evaluation cohorts.

- Maximum selection risk: `0.100`
- Minimum selection coverage: `0.500`
- Selected confidence threshold: `0.500371`
- Held-out test accessed: `False`

| Scope | Patients | Positive labels | Coverage | Abstention | Selective risk | Selective Brier | Retained AUROC | Retained AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled_all | 3025 | 2090 | 1.00000 | 0.00000 | 0.01931 | 0.01750 | 0.83647 | 0.09344 |
| hospital_A | 1492 | 1258 | 1.00000 | 0.00000 | 0.02219 | 0.02032 | 0.83994 | 0.11388 |
| hospital_B | 1533 | 832 | 1.00000 | 0.00000 | 0.01647 | 0.01470 | 0.82481 | 0.07499 |
| hour_6 | 2979 | 56 | 1.00000 | 0.00000 | 0.01880 | 0.01808 | 0.77053 | 0.07476 |
| hour_12 | 2948 | 39 | 1.00000 | 0.00000 | 0.01323 | 0.01297 | 0.74433 | 0.04588 |
| hour_24 | 2323 | 17 | 1.00000 | 0.00000 | 0.00732 | 0.00717 | 0.83253 | 0.05096 |

This is an exploratory retrospective evaluation. The confidence threshold is not a clinically validated decision threshold.
