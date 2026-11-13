# Next-phase validation policy evaluation

This report records the validation-only run completed after the policy-runtime
stabilization. It does not access the held-out test split.

## Reproduction

```bash
PYTHONPATH=. \
ALL_LAB_MODEL_DIR=/private/tmp/evidence-aware-next-phase-state-calibrated \
NEXT_PHASE_POLICY_FIT_MAX_ROWS=50000 \
NEXT_PHASE_COMPLETE_OUTPUT=/private/tmp/evidence-aware-next-phase-validation.json \
./.venv/bin/python experiments/run_complete_next_phase.py \
  --split validation --policies-only
```

The validation cohort contained 6,050 patients and 234,037 hourly rows. The
frozen threshold was `0.9669101582337782`. Acquisition was restricted to rows
that abstained under the all-labs-hidden baseline and to historically observed
laboratory groups.

## Main results

| Policy | Coverage | Risk | Normalized PhysioNet utility | Eligible rows |
|---|---:|---:|---:|---:|
| All labs hidden | 0.90921 | 0.01009 | 0.00247 | — |
| Fixed order | 0.91782 | 0.01036 | 0.00221 | 20,745 |
| Random | 0.92112 | 0.01053 | 0.00312 | 20,745 |
| Patient-specific | 0.91850 | 0.01038 | 0.00506 | 20,745 |
| Oracle upper bound | 0.91782 | 0.01036 | 0.00808 | 20,745 |

The patient-specific policy used 49,362 deterministic training rows for its
selector fit, with a declared cap of 50,000 rows. Its paired patient-cluster
bootstrap difference versus all-labs-hidden was AUROC `+0.000768`
(`95% CI 0.000005 to 0.001553`), AUPRC `+0.002939` (`95% CI -0.004768 to
0.009303`), and Brier improvement `+0.0000465` (`95% CI -0.000138 to
0.000184`).

## Wait-and-re-check

Among 21,248 baseline-abstained rows, 20,652 had a strictly future next
observation. The median and mean delay were one hour. After that observation,
coverage was 0.11224 and 88.78% of cases remained unresolved at the frozen
confidence threshold.

## Interpretation and scope

These are retrospective validation findings, not clinical effectiveness
results. The all-labs-hidden baseline and acquisition policies use the same
state-calibrated XGBoost model. The oracle is an evaluation ceiling and is not
deployable. The risk-averse comparator is implemented as a tested CVaR-style
decision core, but was not substituted into this table because the current
validation runner evaluates the fixed, random, oracle, and trained
patient-specific group policies.

The next approval-gated step is a single locked held-out evaluation using the
frozen protocol. No held-out result is inferred from this validation report.
