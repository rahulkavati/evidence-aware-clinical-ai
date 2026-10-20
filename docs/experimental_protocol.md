# Experimental Protocol

**Project:** Evidence-Aware Clinical AI  
**Research Stage:** Stage 1 — Evidence Sufficiency and Active Information Acquisition  
**Version:** 1.0 — Initial Protocol  
**Date:** October 20, 2026  
**Status:** Proposed — implementation and validation pending

## 1. Research Objective

The objective of this project is to investigate whether a clinical machine-learning system can evaluate the reliability of its available information and decide whether to:

1. Make a prediction using the available evidence.
2. Acquire additional information that may improve the prediction.
3. Abstain when the available evidence is insufficient and further acquisition is not justified or feasible.

The initial application is early sepsis prediction using retrospective ICU observations.

This work investigates a limited capability toward the longer-term research direction of Metacognitive Healthcare AI.

## 2. Dataset

**Primary dataset:** PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0.

**Source:** https://physionet.org/content/challenge-2019/1.0.0/

The dataset contains real, deidentified ICU observations recorded hourly.

### Dataset Audit Results

| Metric                          |     Value |
| ------------------------------- | --------: |
| Patients                        |    40,336 |
| Hourly observations             | 1,552,210 |
| Sepsis-positive patients        |     2,932 |
| Sepsis-negative patients        |    37,404 |
| Patient-level sepsis prevalence |     7.27% |
| Median observation duration     |  38 hours |

The dataset includes 40 clinical variables and the `SepsisLabel` outcome.

The original patient records will remain outside Git version control.

## 3. Prediction Task

### Primary Benchmark

The primary prediction task will follow the official PhysioNet 2019 early-sepsis prediction protocol.

At each eligible ICU hour, the model will use information available up to that hour to produce a sepsis-risk estimate.

The official `SepsisLabel` values and challenge scoring rules will be preserved.

We will not redefine the first positive Challenge label as the verified clinical onset time.

### Secondary Checkpoint Analysis

Additional experiments will evaluate patients at ICU hours 6, 12, and 24.

These checkpoints are used for retrospective acquisition experiments and do not replace the official hourly prediction benchmark.

Patients will be eligible for a checkpoint analysis only when they have a recorded observation at that hour.

## 4. Patient-Level Data Splitting

Patients will be divided into:

| Split      | Planned proportion |
| ---------- | -----------------: |
| Training   |                70% |
| Validation |                15% |
| Test       |                15% |

Splitting will occur at the patient level, ensuring that observations from the same patient never appear in multiple splits.

Where feasible, the split will preserve hospital representation and patient-level sepsis-label proportions.

A fixed random seed will be used, and split identifiers will be generated reproducibly.

Patient identifiers and patient-level data will not be committed to GitHub.

An additional hospital-transfer experiment may train on Hospital A and evaluate on Hospital B, and vice versa.

## 5. Temporal Information Rules

At prediction time \(t\), the model may use only observations available at or before \(t\).

Future clinical measurements must not be used as prediction inputs.

The model must not access future `SepsisLabel` values.

Normalization, imputation, feature selection, and calibration must be fitted using the appropriate training or validation data.

For each patient, derived features such as last-observed value, observation mask, and time since last measurement must be calculated using only the permitted observation history.

## 6. Clinical Prediction Baselines

We will initially implement:

### Baseline 1 — Logistic Regression

An interpretable classification model used to establish a simple prediction baseline.

### Baseline 2 — XGBoost

A nonlinear gradient-boosted tree model used as a stronger structured-data baseline.

### Optional Temporal Baseline — GRU/LSTM

A PyTorch temporal model may be implemented if needed for a more faithful adaptation of published active-sensing methods.

The same patient-level splits and prediction protocol will be used for comparisons.

## 7. Uncertainty Estimation

The prediction models will produce estimated sepsis probabilities.

We will evaluate prediction reliability using:

- Probability calibration.
- Brier score.
- Expected Calibration Error.
- Reliability diagrams.
- Selective risk versus prediction coverage.

Ensemble disagreement may be investigated as an additional uncertainty measure.

Predictive confidence will not automatically be interpreted as evidence sufficiency.

A sufficiency criterion must be defined and calibrated using validation data.

## 8. Acquisition Unit

For the initial acquisition experiment, one acquisition action will reveal the latest eligible historically observed value of one selected clinical feature at the prediction checkpoint.

Candidate measurements must have been recorded at or before the checkpoint.

The simulator will not invent values for measurements that were never observed.

The selected value, its missingness indicator, and any derived information affected by masking must be handled consistently.

This is an information-reveal simulation, not a real clinical test-ordering system.

## 9. Retrospective Masking Protocol

To simulate incomplete information:

1. Start with a patient's history available through checkpoint \(t\).
2. Select eligible historically observed measurements to hide.
3. Construct the visible patient state without the hidden information.
4. Generate an initial prediction and uncertainty estimate.
5. Allow the acquisition policy to select an eligible hidden measurement.
6. Reveal the selected information.
7. Recompute the affected features
