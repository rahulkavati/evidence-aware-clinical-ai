# PhysioNet 2019 — Dataset Audit

**Project:** Evidence-Aware Clinical AI  
**Dataset:** PhysioNet/Computing in Cardiology Challenge 2019, v1.0.0  
**Status:** Completed initial dataset audit  
**Audit date:** October 16, 2026

## Dataset Summary


| Metric                          | Result    |
| ------------------------------- | --------- |
| Total patient records           | 40,336    |
| Hospital A patients             | 20,336    |
| Hospital B patients             | 20,000    |
| Total hourly observations       | 1,552,210 |
| Sepsis-positive patients        | 2,932     |
| Sepsis-negative patients        | 37,404    |
| Patient-level sepsis prevalence | 7.27%     |
| Median observation duration     | 38 hours  |


A patient is counted as sepsis-positive if at least one hourly observation has `SepsisLabel = 1`.

## Most Missing Clinical Variables


| Variable         | Missingness |
| ---------------- | ----------- |
| Bilirubin_direct | 99.81%      |
| Fibrinogen       | 99.34%      |
| TroponinI        | 99.05%      |
| Bilirubin_total  | 98.51%      |
| Alkalinephos     | 98.39%      |
| AST              | 98.38%      |
| Lactate          | 97.33%      |
| PTT              | 97.06%      |
| SaO2             | 96.55%      |
| EtCO2            | 96.29%      |


Missingness is calculated across hourly observations, not as the percentage of patients who never had the measurement.

## Initial Findings

The dataset contains a substantial imbalance between sepsis-positive and sepsis-negative patients.

Several laboratory measurements are rarely observed at individual hourly time points.

These properties make the dataset relevant for investigating incomplete clinical information, but they also require careful treatment of missingness, temporal information, and measurement availability.

## Research Implications

The initial prediction models should be evaluated using metrics appropriate for imbalanced outcomes.

The acquisition framework must distinguish between:

- measurements already observed;
- historical measurements artificially hidden for simulation;
- measurements never recorded;
- measurements whose acquisition timing or cost is simulated.



## Next Steps

1. Analyze hourly sepsis-label distribution.
2. Inspect first positive-label timing.
3. Define patient-level training, validation, and test splits.
4. Establish leakage-safe preprocessing.
5. Implement the official challenge evaluation protocol.
6. Begin Logistic Regression and XGBoost baselines.



## Limitations

This audit is descriptive. It does not establish predictive performance, the clinical usefulness of any measurement, or the feasibility of requesting missing tests.

All statistics were generated from the locally downloaded public training dataset.