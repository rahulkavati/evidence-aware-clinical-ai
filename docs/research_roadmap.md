 # Research Roadmap

## Completed milestones

- G1 mask-aware XGBoost and independent calibration.
- Retrospective acquisition headroom evaluation with patient-cluster intervals.
- Validation-only selective prediction with patient-disjoint threshold selection.
- Selective-prediction robustness across risk targets, hospitals, and checkpoints.
- One-step acquisition after abstention using historical eligibility and a fixed order.
- Training-derived feature-level VOI policy, acquisition baselines, diagnostics, and validation cost sensitivity.

## Project completion status

All planned execution milestones are complete. The project has a frozen
validation protocol, validation cost/budget sensitivity, reproducibility
artifacts, and one locked held-out evaluation. The absence of prospective
turnaround timestamps is recorded as a limitation rather than filled with
synthetic clinical assumptions.

## Current decision

The validation and held-out evidence do not justify claiming that acquisition
improves selective prediction. VOI and fixed-order acquisition remain
retrospective baselines; in the locked evaluation, fixed-order acquisition
slightly increased coverage but worsened risk and Brier score relative to
abstention.
