# Baseline Comparison and Research References

**Project:** Evidence-Aware Clinical AI  
**Research Stage:** Stage 1 — Evidence Sufficiency and Active Information Acquisition  
**Status:** Literature-based comparison; experimental validation pending

## 1. Purpose

This document identifies the existing research methods and baseline models that we plan to compare against.

Our goal is to investigate whether a clinical AI system can evaluate available patient information and decide whether to predict, acquire additional information, or abstain.

We will evaluate how measurement availability, timing, simulated acquisition delays, and costs affect these decisions.

## 2. Primary Research References


| Reference                  | Venue               | Main Contribution                                               | Role in Our Project                     |
| -------------------------- | ------------------- | --------------------------------------------------------------- | --------------------------------------- |
| SepsisLab                  | KDD 2024            | Sepsis prediction, uncertainty propagation, and active sensing  | Primary methodological reference        |
| Risk-Averse Active Sensing | NeurIPS 2023        | Learns what and when to measure under cost and risk constraints | Temporal acquisition reference          |
| PhysioNet Challenge 2019   | PhysioNet/CinC 2019 | Public ICU dataset and early-sepsis prediction benchmark        | Primary dataset and evaluation protocol |




### Reference Links

**SepsisLab — KDD 2024**

Paper: [https://pmc.ncbi.nlm.nih.gov/articles/PMC11470769/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11470769/)

Code: [https://github.com/yinchangchang/SepsisLab](https://github.com/yinchangchang/SepsisLab)

**Risk-Averse Active Sensing — NeurIPS 2023**

Paper: [https://proceedings.neurips.cc/paper_files/paper/2023/hash/1498a03a04f9bcd3a7d44058fc5dc639-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1498a03a04f9bcd3a7d44058fc5dc639-Abstract-Conference.html)

Code: [https://github.com/yvchao/cvar_sensing](https://github.com/yvchao/cvar_sensing)

**PhysioNet Challenge 2019**

Dataset: [https://physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/challenge-2019/1.0.0/)

## 3. Prediction Model Baselines

These models establish the initial early-sepsis prediction performance.


| Model               | Purpose                                       | Planned Framework |
| ------------------- | --------------------------------------------- | ----------------- |
| Logistic Regression | Simple, interpretable prediction baseline     | scikit-learn      |
| XGBoost             | Strong nonlinear tabular prediction baseline  | XGBoost           |
| GRU/LSTM            | Temporal neural prediction model, if required | PyTorch           |


The same dataset splits and compatible evaluation protocols will be used for fair comparisons.

## 4. Information Acquisition Baselines

These methods determine which additional measurements to acquire.


| Method                     | How It Works                                        | Role                                |
| -------------------------- | --------------------------------------------------- | ----------------------------------- |
| No Acquisition             | Predict using currently available measurements      | Reference baseline                  |
| Random Acquisition         | Select an eligible measurement randomly             | Simple acquisition baseline         |
| Fixed-Order Acquisition    | Acquire measurements in a predefined order          | Simple deterministic baseline       |
| Information Gain           | Select a measurement expected to reduce uncertainty | Greedy acquisition baseline         |
| SepsisLab RAS              | Uses propagated uncertainty to rank measurements    | Primary published-method adaptation |
| Risk-Averse Active Sensing | Optimizes measurement selection and timing          | Optional advanced comparator        |


SepsisLab RAS must not be claimed as reproduced unless its required uncertainty-propagation mechanism has been implemented faithfully.

## 5. Planned Research Comparison

We will evaluate acquisition methods under three experimental settings.

### Setting A — Idealized Acquisition

Eligible recorded measurements may be revealed immediately.

### Setting B — Availability-Constrained Acquisition

Only measurements allowed by the patient's historical availability mask may be revealed.

### Setting C — Availability and Delay Constraints

Acquisition additionally considers simulated turnaround delays, decision deadlines, and measurement costs.

These are retrospective simulations, not real hospital test-ordering experiments.

## 6. Evaluation Metrics



### Prediction Performance

- AUROC
- AUPRC
- Official PhysioNet Challenge utility, where applicable
- Calibration metrics



### Acquisition Performance

- Number of acquired measurements
- Acquisition cost
- Acquisition success/failure rate
- Prediction performance versus acquisition budget



### Decision Regulation

- Prediction coverage
- Abstention rate
- Selective risk
- Utility versus acquisition cost



## 7. Published Results Versus Our Results

Published results from the original papers must be recorded separately from our experiments.


| Reference                  | Published Results                                                | Our Results   |
| -------------------------- | ---------------------------------------------------------------- | ------------- |
| SepsisLab                  | See `published_results.md`                                       | Not available |
| Risk-Averse Active Sensing | Pending detailed verification                                    | Not available |
| PhysioNet 2019 benchmark   | Official challenge results; specific reference pending selection | Not available |
| Our proposed policy        | Not applicable                                                   | Not available |


Results obtained on different datasets or prediction protocols must not be directly compared as evidence of improvement.

## 8. Potential Research Difference

Existing research has already studied uncertainty-aware active sensing, cost-aware acquisition, temporal sensing, and acquisition stopping.

Our proposed investigation focuses on the interaction between:

- evidence sufficiency;
- patient-specific measurement availability;
- temporal constraints;
- simulated acquisition delays;
- measurement costs;
- predict, acquire, and abstain decisions.

These are proposed research dimensions, not confirmed novelty claims.

## 9. Reproduction Plan

1. Implement Logistic Regression and XGBoost.
2. Evaluate uncertainty and calibration.
3. Build the retrospective acquisition simulator.
4. Implement no-acquisition, random, fixed-order, and information-gain baselines.
5. Assess compatibility with SepsisLab RAS.
6. Implement a faithful adaptation if feasible.
7. Evaluate acquisition methods under shared experimental conditions.
8. Investigate an evidence-aware policy after baseline failure analysis.



## 10. Current Status

**Documentation:** Initial baseline comparison defined.

**Implementation:** Not started.

**Experimental results:** Not available.

**Novelty:** Not established.

The next step is to inspect PhysioNet Challenge 2019 and establish the reproducible prediction protocol.