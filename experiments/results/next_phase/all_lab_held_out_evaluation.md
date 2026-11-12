# All-26-laboratory held-out evaluation

Clean next-phase evaluation using a newly trained mask-aware XGBoost model,
independent Platt calibration, and a threshold selected on patient-disjoint
validation data. The held-out test split was accessed once after the protocol
was frozen.

- Candidate laboratory features: **26**
- Frozen validation threshold: **0.9743948721**
- Validation risk target: **0.01**
- Held-out patients: **6,051**
- Held-out rows: **232,215**

## Held-out comparison

| State | Coverage | Selective risk | Brier score | Retained AUROC | Retained AUPRC |
|---|---:|---:|---:|---:|---:|
| All 26 labs hidden | 90.520% | 1.028% | 0.010118 | 0.71765 | 0.02544 |
| All 26 labs visible | 86.208% | 0.791% | 0.007797 | 0.75040 | 0.02132 |

## Comparison with locked G1

The original G1 policy used six candidate labs, a fixed one-step acquisition
policy, and threshold **0.9626458569**:

| Evaluation | Coverage | Risk | Brier score |
|---|---:|---:|---:|
| G1 baseline, six labs hidden | 92.401% | 1.067% | 0.010439 |
| G1 one-step acquisition assisted | 92.666% | 1.084% | 0.010603 |
| New all-26 model, all labs hidden | 90.520% | 1.028% | 0.010118 |
| New all-26 model, all labs visible | 86.208% | 0.791% | 0.007797 |

## Interpretation

The all-26 model improves retained-set risk and Brier score relative to G1 when
all labs are hidden, but it retains fewer predictions because its validation-
selected confidence threshold is stricter. Full laboratory visibility further
improves reliability and discrimination, but unexpectedly reduces coverage at
the same strict threshold. This is a headroom comparison, not yet a validated
patient-specific acquisition policy.

The GRU/TCN components and patient-specific recommendation scorer have passed
implementation tests and smoke checks, but were not included in this locked
numeric comparison.
