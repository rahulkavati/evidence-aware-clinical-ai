# Research Questions

**Project:** Evidence-Aware Clinical AI  
**Status:** Provisional — to be refined after baseline experiments

## Primary Research Question

Can a clinical AI system assess the sufficiency of available patient information and regulate its next action between prediction, information acquisition, and abstention?

## Supporting Research Questions

### RQ1 — Evidence Sufficiency

Can calibrated predictive uncertainty help identify when available clinical information is insufficient for reliable early sepsis prediction?

**Evaluation:** Calibration, prediction errors, and selective risk versus coverage.

### RQ2 — Information Acquisition

How do measurement availability, timing, simulated acquisition delays, and costs affect the usefulness of active information acquisition?

**Evaluation:** Prediction performance, acquisition success rate, number of measurements, and cumulative cost.

### RQ3 — Decision Regulation

Can an evidence-aware policy make effective decisions between predicting, acquiring additional information, and abstaining?

**Evaluation:** Prediction utility, abstention rate, selective risk, and acquisition cost.

### RQ4 — Potential Methodological Improvement

Can a constraint-aware acquisition policy outperform appropriate existing methods under the same experimental conditions?

**Evaluation:** Comparison against random acquisition, fixed-order acquisition, information-gain methods, and an adapted published active-sensing baseline.

## Research Hypothesis

Explicitly evaluating the usefulness and feasibility of additional information may improve the prediction-utility-versus-cost tradeoff compared with acquisition methods that do not account for the same constraints.

This hypothesis has not yet been tested.

## Initial Experimental Setting

- **Dataset:** PhysioNet Challenge 2019
- **Task:** Early sepsis prediction
- **Prediction baselines:** Logistic Regression and XGBoost
- **Methodological reference:** SepsisLab, KDD 2024
- **Experimental approach:** Retrospective acquisition simulation
- **Long-term direction:** Metacognitive Healthcare AI

## Research Scope

The master's project focuses on structured clinical measurements. Full medical-report interpretation, multimodal reasoning, and clinical deployment are outside the initial scope.

The intended contribution is a measurable capability for evidence monitoring and decision regulation, not a claim of general metacognitive intelligence.