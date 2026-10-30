# G1 — Mask-Aware Prediction and Independent Calibration

## Objective

Establish whether retrospective revelation of previously recorded clinical measurements provides sufficient predictive benefit to justify patient-specific acquisition research.

## Dataset

PhysioNet/Computing in Cardiology Challenge 2019, training sets A and B.

## Evaluation design

- Preserve the existing patient-level held-out test split.
- Use training-derived, patient-disjoint cohorts for predictor fitting, calibration, and acquisition-policy development.
- Reserve an independent development-evaluation cohort for the G1 acquisition-headroom experiment.
- Do not use the existing validation-calibrated model as the primary G1 model.
- Do not access held-out test outcomes during development.

## Predictor

Train XGBoost using naturally observed and synthetically masked feature states. Preserve the original feature ordering, checkpoint definitions, and temporal cutoffs.

## Calibration

Fit a sigmoid calibrator using predictions from patients excluded from predictor fitting. Select the calibration method without using the G1 evaluation cohort.

## Acquisition headroom

Compare the same patients under:

1. No acquisition — all six candidate laboratory feature families masked.
2. Full revelation — all historically available candidate laboratory feature families visible.

Report AUROC, AUPRC, Brier score, and paired patient-cluster confidence intervals. Record the actual number of positive checkpoint labels.

## Decision rule

If full revelation offers limited or uncertain improvement, prioritize selective prediction and abstention. Otherwise, proceed with patient-specific information-value estimation.

Do not select the research direction using the held-out test set.

## Limitations

Historical laboratory revelation is not equivalent to ordering a new clinical test. Measurement availability is retrospective and reflects historical clinical practice. Results do not establish prospective clinical utility.
