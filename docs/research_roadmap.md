 # Research Roadmap

## Completed milestones

- G1 mask-aware XGBoost and independent calibration.
- Retrospective acquisition headroom evaluation with patient-cluster intervals.
- Validation-only selective prediction with patient-disjoint threshold selection.
- Selective-prediction robustness across risk targets, hospitals, and checkpoints.
- One-step acquisition after abstention using historical eligibility and a fixed order.
- Training-derived feature-level VOI policy, acquisition baselines, diagnostics, and validation cost sensitivity.

## Five remaining milestones

1. **Freeze the protocol.** Record the prediction state, threshold-selection rule,
   acquisition eligibility rule, primary metrics, and exclusion rules.
2. **Complete validation sensitivity.** Finish the pre-specified budget, simulated
   cost, and delay analyses without changing the held-out test split.
3. **Lock claims and reproducibility.** Consolidate aggregate reports, seeds,
   environment details, limitations, and manuscript-ready tables.
4. **Run the held-out evaluation.** Access the held-out test split exactly once,
   only after explicit approval, using the frozen protocol and no retuning.
5. **Finalize the project.** Audit outputs and privacy boundaries, complete the
   research narrative, and prepare the final repository/package.

## Current decision

The validation evidence does not yet justify claiming that acquisition improves
selective prediction. VOI and fixed-order acquisition remain retrospective
baselines; any positive result must be compared against abstention under the
same patient-disjoint protocol.
