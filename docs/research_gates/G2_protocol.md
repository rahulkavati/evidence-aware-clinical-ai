# G2 — Selective Prediction and Acquisition After Abstention

## Frozen validation protocol

- Use the G1 mask-aware XGBoost and its independent sigmoid calibrator.
- Select the confidence threshold on one patient-disjoint validation cohort.
- Evaluate on a separate patient-disjoint validation cohort.
- Never read the held-out test split during threshold or policy development.
- Candidate acquisition is limited to historically observed laboratory features
  available by the checkpoint; never-observed measurements remain unavailable.
- Acquisition must be evaluated against the no-acquisition abstention baseline.

## Primary metrics

- selective coverage and selective risk;
- abstention rate;
- selective Brier score;
- acquisition coverage and rescued-abstention count;
- paired patient-cluster confidence intervals where applicable.

## Interpretation rule

An acquisition policy is not considered beneficial merely because it rescues
some abstentions. It must improve the validated risk/coverage or utility tradeoff
against abstention under the same budget, cost, and delay assumptions.

Historical revelation is a retrospective information-reveal simulation, not a
clinical test-ordering study. Thresholds, costs, and delays are exploratory and
are not clinically validated.

## Test-set lock

The held-out test evaluation remains a separate milestone. It requires explicit
approval and must use this frozen protocol without threshold or policy retuning.
