# Research Methodology and Experimental Design

**Project:** Evidence-Aware Clinical AI  
**Status:** Proposed methodology — implementation pending

## 1. Research Objective

This project investigates whether a clinical machine-learning system can evaluate the reliability of available patient information and decide whether to make a prediction, acquire additional information, or abstain.

The initial application is early sepsis prediction using retrospective ICU data.

Our main experimental interest is how measurement availability, timing, simulated acquisition delays, and costs affect these decisions.

## 2. Dataset

The primary dataset is the PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0.

Source: https://physionet.org/content/challenge-2019/1.0.0/

The dataset contains real, deidentified hourly ICU observations, missing clinical measurements, and sepsis labels.

We will initially follow the official challenge's early-prediction task and evaluation protocol.

## 3. Data Preparation

The data pipeline will:

- Load patient-level time-series files.
- Verify feature names, data types, and timestamps.
- Analyze missing measurements and label distributions.
- Create patient-level training, validation, and test splits.
- Apply leakage-safe preprocessing.
- Preserve original missingness masks and observation times.
- Save reproducible preprocessing configurations.

Imputation parameters, normalization statistics, and calibration methods will be fitted using the appropriate training or validation data only.

Future patient observations and outcome labels must not leak into the information available at an earlier prediction time.

## 4. Clinical Prediction Models

We will begin with two baseline models.

### Logistic Regression

A simple interpretable baseline for estimating sepsis risk.

### XGBoost

A nonlinear tabular machine-learning model suitable for structured clinical measurements.

A PyTorch GRU/LSTM may be added if needed for a closer comparison with published temporal active-sensing methods.

All models will be evaluated under a consistent patient-level split and prediction protocol.

## 5. Uncertainty Estimation

The prediction models will produce estimated sepsis risk.

We will investigate whether uncertainty and calibration methods help identify unreliable predictions.

Initial approaches include:

- prediction confidence;
- probability calibration;
- ensemble disagreement, if feasible.

Evaluation will include calibration, prediction errors, and selective risk versus coverage.

A model's confidence alone will not be treated as proof that clinical information is sufficient.

## 6. Active Information Acquisition

The acquisition simulator will represent which measurements are currently available and which historically observed measurements may be revealed.

Initial acquisition baselines:

1. No acquisition.
2. Random acquisition.
3. Fixed-order acquisition.
4. Information-gain-based acquisition.
5. Adapted SepsisLab active sensing, if a faithful implementation is feasible.

All acquisition policies will be evaluated using compatible prediction models and the same experimental conditions.

## 7. Evidence Evaluation and Decision Regulation

The proposed decision policy will consider:

- current predictive uncertainty;
- expected benefit of additional information;
- measurement availability;
- simulated acquisition cost;
- simulated acquisition delay;
- remaining acquisition budget.

It will choose between three actions:

**Predict:** Return a model prediction when the validated decision criteria are satisfied.

**Acquire:** Request an eligible measurement when its expected benefit justifies the acquisition.

**Abstain:** Withhold the automated prediction when the available evidence does not support a reliable decision and further acquisition is not justified or feasible.

These actions are simulated research decisions, not clinical recommendations.

## 8. Experimental Conditions

We will compare acquisition policies under three settings.

### Setting A — Idealized Acquisition

Eligible recorded measurements can be revealed immediately within the experimental acquisition budget.

### Setting B — Availability-Constrained Acquisition

Only measurements supported by the patient's historical record and the defined availability protocol may be revealed.

### Setting C — Availability and Delay Constraints

The acquisition simulator additionally applies explicitly defined measurement delays and decision deadlines.

Delay assumptions will be documented and tested through sensitivity analyses.

## 9. Evaluation Metrics

### Prediction Quality

- AUROC
- AUPRC
- Official PhysioNet Challenge utility, where applicable
- Calibration metrics

### Information Acquisition

- Number of measurements acquired
- Acquisition cost
- Acquisition success/failure rate
- Performance versus acquisition budget

### Decision Regulation

- Prediction coverage
- Abstention rate
- Selective risk
- Utility of the resulting decisions

## 10. Baseline Comparison

The proposed approach will be compared with existing acquisition methods under equivalent experimental conditions.

SepsisLab (KDD 2024) is the primary methodological reference.

However, SepsisLab's original datasets and prediction protocol differ from PhysioNet Challenge 2019. Any implementation on our dataset must be described as an adaptation unless its original experiment is reproduced exactly.

## 11. Research Contribution

The first intended contribution is a reproducible evaluation of evidence-aware clinical information acquisition under explicitly defined availability and temporal constraints.

If baseline experiments reveal meaningful limitations, we will investigate whether a constraint-aware decision policy can improve the prediction-utility-versus-cost tradeoff.

A novel algorithm is not assumed in advance.

## 12. Experimental Validity

We will use held-out patient evaluation, appropriate uncertainty intervals, and multiple random seeds where feasible.

Policy thresholds and hyperparameters will be selected without using the held-out test set.

We will distinguish between:

- historically observed measurements;
- artificially hidden measurements;
- unobserved measurements whose values are unknown;
- simulated acquisition costs and delays.

Retrospective acquisition experiments cannot establish how a system would perform if deployed in a real hospital.

## 13. Long-Term Research Direction

This project investigates a limited first-stage capability toward Metacognitive Healthcare AI.

It focuses on evidence monitoring, information seeking, and decision regulation using structured clinical measurements.

Future research may extend these capabilities to medical reports, clinical notes, multimodal patient evidence, and more complex reasoning under uncertainty.

For the high-level system and experimental architecture diagrams,
see [System Design](architecture/system_design.md).