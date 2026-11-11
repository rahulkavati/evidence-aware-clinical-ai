# Selective prediction robustness gate

Validation-only sensitivity analysis with patient-disjoint threshold selection and evaluation.

- Held-out test accessed: `False`
- Patient-cluster bootstrap resamples: `1000`

## Pooled evaluation

| Max risk | Threshold | Coverage | Coverage 95% CI | Selective risk | Risk 95% CI | Abstention |
|---:|---:|---:|---|---:|---|---:|
| 0.01 | 0.958901 | 0.92006 | [0.90640, 0.93498] | 0.01033 | [0.00864, 0.01218] | 0.07994 |
| 0.05 | 0.500371 | 1.00000 | [1.00000, 1.00000] | 0.01931 | [0.01647, 0.02197] | 0.00000 |
| 0.10 | 0.500371 | 1.00000 | [1.00000, 1.00000] | 0.01931 | [0.01647, 0.02197] | 0.00000 |
| 0.20 | 0.500371 | 1.00000 | [1.00000, 1.00000] | 0.01931 | [0.01647, 0.02197] | 0.00000 |

## Subgroup evaluation

### Maximum risk 0.01

| Scope | Coverage | Risk | Risk 95% CI |
|---|---:|---:|---|
| pooled_all | 0.92006 | 0.01033 | [0.00864, 0.01218] |
| hospital_A | 0.91270 | 0.01209 | [0.00952, 0.01479] |
| hospital_B | 0.92735 | 0.00862 | [0.00647, 0.01094] |
| hour_6 | 0.94696 | 0.01524 | [0.01059, 0.01953] |
| hour_12 | 0.96540 | 0.01124 | [0.00740, 0.01513] |
| hour_24 | 0.97503 | 0.00618 | [0.00309, 0.00971] |

### Maximum risk 0.05

| Scope | Coverage | Risk | Risk 95% CI |
|---|---:|---:|---|
| pooled_all | 1.00000 | 0.01931 | [0.01647, 0.02197] |
| hospital_A | 1.00000 | 0.02219 | [0.01860, 0.02570] |
| hospital_B | 1.00000 | 0.01647 | [0.01259, 0.02133] |
| hour_6 | 1.00000 | 0.01880 | [0.01409, 0.02351] |
| hour_12 | 1.00000 | 0.01323 | [0.00916, 0.01764] |
| hour_24 | 1.00000 | 0.00732 | [0.00387, 0.01119] |

### Maximum risk 0.10

| Scope | Coverage | Risk | Risk 95% CI |
|---|---:|---:|---|
| pooled_all | 1.00000 | 0.01931 | [0.01647, 0.02197] |
| hospital_A | 1.00000 | 0.02219 | [0.01860, 0.02570] |
| hospital_B | 1.00000 | 0.01647 | [0.01259, 0.02133] |
| hour_6 | 1.00000 | 0.01880 | [0.01409, 0.02351] |
| hour_12 | 1.00000 | 0.01323 | [0.00916, 0.01764] |
| hour_24 | 1.00000 | 0.00732 | [0.00387, 0.01119] |

### Maximum risk 0.20

| Scope | Coverage | Risk | Risk 95% CI |
|---|---:|---:|---|
| pooled_all | 1.00000 | 0.01931 | [0.01647, 0.02197] |
| hospital_A | 1.00000 | 0.02219 | [0.01860, 0.02570] |
| hospital_B | 1.00000 | 0.01647 | [0.01259, 0.02133] |
| hour_6 | 1.00000 | 0.01880 | [0.01409, 0.02351] |
| hour_12 | 1.00000 | 0.01323 | [0.00916, 0.01764] |
| hour_24 | 1.00000 | 0.00732 | [0.00387, 0.01119] |

Thresholds and risk targets are exploratory and are not clinically validated.
