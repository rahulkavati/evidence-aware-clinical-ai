# P003 — PhysioNet/CinC Challenge 2019

**Title:** Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019

**Authors:** Matthew Reyna and colleagues  
**Year:** 2019  
**Source:** PhysioNet / Critical Care Medicine  
**Review status:** Initial source-based review

**Dataset:** [https://physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/challenge-2019/1.0.0/)

**DOI:** [https://doi.org/10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857)

**Original publication:** [https://doi.org/10.1097/CCM.0000000000004145](https://doi.org/10.1097/CCM.0000000000004145)

## Research Problem

The challenge investigates early prediction of sepsis from routinely collected ICU clinical observations.

A central objective is to identify sepsis sufficiently early to support timely intervention while avoiding unnecessary false alarms.

## Dataset

The publicly released training data includes 40,336 patient records from two hospital systems.

The data contains hourly observations, including vital signs, laboratory measurements, demographic variables, and missing measurements.

A third hospital system was used for hidden challenge evaluation.

## Prediction Task

The challenge uses hourly sepsis labels and requires predictions based only on current and previously observed information.

The labels and scoring protocol are designed around early identification of sepsis, including a six-hour early-warning target.

For the initial project baseline, we will follow the official challenge task before considering alternative prediction horizons.

## Evaluation

The primary benchmark metric is the official challenge utility score, which rewards appropriately timed predictions and penalizes incorrect or late decisions.

We will also report:

- AUROC
- AUPRC
- Sensitivity and specificity where appropriate
- Calibration metrics
- Prediction performance across patient subgroups



## Relationship to Our Project

This benchmark provides the primary experimental setting for our Stage 1 research.

It allows us to investigate:

- Prediction under incomplete clinical information.
- Reliability of model predictions.
- Sequential acquisition of historically observed measurements.
- Measurement availability and timing.
- Simulated acquisition costs and delays.
- Predict, acquire, or abstain decisions.



## Important Limitations

The dataset does not establish the outcomes of tests that were never performed.

Historical measurement availability is not equivalent to the ability to request a new measurement in clinical practice.

Acquisition costs and turnaround delays must be treated as assumptions unless independently verified.

## Planned Implementation

1. Download and inspect the official training data.
2. Verify feature names and missingness patterns.
3. Establish patient-level train, validation, and test splits.
4. Implement the official challenge scoring function.
5. Train Logistic Regression and XGBoost baselines.
6. Add uncertainty and active acquisition experiments.



## Published Results

The challenge provides official results and participant submissions.

We have not yet selected a specific leaderboard entry for numerical reproduction.

Before recording benchmark scores, we will identify the exact submission, evaluation split, metric, and source.

## Citation

Reyna, M., Josef, C., Jeter, R., Shashikumar, S., Moody, B., Westover, M. B., Sharma, A., Nemati, S., and Clifford, G. D. (2019). Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. PhysioNet, version 1.0.0. [https://doi.org/10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857)