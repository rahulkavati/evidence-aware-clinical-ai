# Complete next-phase evaluation report

This report consolidates the completed all-26-laboratory XGBoost evaluation,
clinical-group action policies, wait/re-check evaluation, multi-horizon label
analysis, and structured narrative rendering on the locked held-out cohort.

## Dataset and protocol

- Patients: **6,051**
- Rows: **232,215**
- Candidate laboratories: **26**
- Frozen threshold: **0.9743948721**
- Held-out evaluation: one locked run after validation selection

## Group-policy comparison

| Policy/state | Coverage | Risk | Brier score | Eligible rows |
|---|---:|---:|---:|---:|
| All labs hidden | 90.520% | 1.028% | 0.010118 | — |
| Fixed clinical-group order | 86.696% | 0.831% | 0.008183 | 209,620 |
| Random clinical group | 88.714% | 0.950% | 0.009349 | 209,620 |
| Patient-specific recommendation | 86.696% | 0.831% | 0.008183 | 209,620 |
| Oracle group ceiling | 86.696% | 0.831% | 0.008183 | 209,620 |

The fixed, patient-specific, and oracle policies converge because the strict
threshold and the available group probabilities produced the same selected
group for this cohort. This means the patient-specific scorer did not add
measurable value under the current one-step group design.

## Wait/re-check

Rows with a next future observation: **226,164**.

| State | Coverage | Risk | Brier score |
|---|---:|---:|---:|
| Next-observation re-check | 85.907% | 0.783% | 0.007719 |

This is a retrospective future-row analysis, not prospective turnaround-time
evidence.

## Multi-horizon urgency targets

Using future labels strictly after the current row:

| Horizon | Positive rate | AUROC using the calibrated all-lab score |
|---|---:|---:|
| Within 6 hours | 2.544% | 0.80572 |
| Within 12 hours | 3.348% | 0.80576 |
| Within 24 hours | 4.678% | 0.80410 |

These are horizon-label analyses using the existing calibrated score; they are
not yet separately trained horizon-specific neural predictors.

## Structured language layer

The language layer renders structured decisions only:

```text
Action: abstain. Estimated sepsis probability: 0.120; confidence: 0.880. Structured output only; no language-layer decision was made.
```

It does not change probabilities or select actions.

## Comparison with original G1

| Evaluation | Coverage | Risk | Brier |
|---|---:|---:|---:|
| Original G1, six labs hidden | 92.401% | 1.067% | 0.010439 |
| Original G1, one-step acquisition | 92.666% | 1.084% | 0.010603 |
| New all-26 labs hidden | 90.520% | 1.028% | 0.010118 |
| New fixed group action | 86.696% | 0.831% | 0.008183 |

The all-26 model is more reliable but more conservative. The group-action
comparison improves retained-set risk and Brier score, but reduces coverage;
it does not demonstrate that the patient-specific policy beats a fixed order.

## Verification

The repository test suite passes with **111 tests and 12 subtests**. The
integrated runner is `experiments/run_complete_next_phase.py`.
