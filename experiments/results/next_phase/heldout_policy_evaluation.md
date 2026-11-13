# Next-phase locked held-out evaluation

This is the single locked held-out evaluation of the frozen next-phase
protocol. The test split was accessed only after validation selection and
approval to complete the project.

## Reproduction

```bash
PYTHONPATH=. \
ALL_LAB_MODEL_DIR=/private/tmp/evidence-aware-next-phase-state-calibrated \
NEXT_PHASE_POLICY_FIT_MAX_ROWS=50000 \
NEXT_PHASE_COMPLETE_OUTPUT=/private/tmp/evidence-aware-next-phase-heldout.json \
./.venv/bin/python experiments/run_complete_next_phase.py \
  --split test --policies-only
```

The frozen confidence threshold was `0.9669101582337782`. The held-out cohort
contained 6,051 patients and 232,215 hourly rows, including 4,203 positive
labels. Acquisition was restricted to baseline-abstained rows and historically
observed laboratory groups.

## Policy comparison

| Policy | Coverage | Risk | Normalized PhysioNet utility | Eligible rows |
|---|---:|---:|---:|---:|
| All labs hidden | 0.91186 | 0.01062 | 0.00803 | — |
| Fixed order | 0.92114 | 0.01083 | 0.00773 | 19,829 |
| Random | 0.92414 | 0.01109 | 0.00542 | 19,829 |
| Patient-specific | 0.92202 | 0.01088 | 0.00750 | 19,829 |
| Oracle upper bound | 0.92114 | 0.01083 | 0.01331 | 19,829 |

The patient-specific selector used 49,362 deterministic training rows under
the declared 50,000-row policy-fit cap. Relative to all-labs-hidden, its
paired patient-cluster bootstrap differences were:

| Metric difference | Estimate | 95% confidence interval |
|---|---:|---:|
| AUROC | +0.001279 | +0.000452 to +0.002066 |
| AUPRC | +0.006972 | -0.003197 to +0.015368 |
| Brier improvement | +0.0000894 | -0.0000373 to +0.0001854 |

Intervals use 200 patient-cluster bootstrap replicates with seed 42. The
oracle is an evaluation ceiling and is not deployable.

## Wait-and-re-check

There were 20,467 baseline-abstained rows. Of these, 19,905 had a strictly
future next observation and 562 had no future observation. The mean, median,
and 95th-percentile delay were all one hour. After the next observation,
coverage was 0.10962, risk was 0.04583, and 89.04% remained unresolved at the
frozen threshold.

## Descriptive horizon associations

These are not horizon-trained models; they use the current-label model on
currently negative rows only.

| Horizon | Rows | Positive rate | AUROC | AUPRC | Brier |
|---:|---:|---:|---:|---:|---:|
| 6h | 228,012 | 0.00940 | 0.80354 | 0.05050 | 0.00933 |
| 12h | 228,012 | 0.01760 | 0.80328 | 0.09475 | 0.01665 |
| 24h | 228,012 | 0.03114 | 0.80021 | 0.16710 | 0.02877 |

## Final interpretation

The patient-specific policy improved AUROC and AUPRC relative to the
all-labs-hidden state, but its utility was slightly lower than the baseline and
its AUPRC and Brier intervals include no difference. The evidence therefore
does not support a claim that patient-specific acquisition improves the final
clinical utility metric under this protocol. The result supports reporting
the policy as a tested, modest discrimination improvement with inconclusive
utility impact.

These are retrospective research results, not prospective clinical evidence.
