
# Evidence-Aware Clinical AI

**Master's Research Project | Computer Science**

**Research direction:** Evidence sufficiency, uncertainty-aware information acquisition, and decision regulation in healthcare AI.

**Long-term vision:** Metacognitive Healthcare AI.

## Overview

Healthcare AI systems often make predictions using incomplete patient information. This project explores whether an AI system can evaluate the reliability of its available evidence and decide whether to make a prediction, request additional clinical information, or abstain when uncertainty remains high.

The initial research focuses on early sepsis prediction using structured ICU patient data.

## Research Objectives

- Evaluate prediction reliability under incomplete clinical information.
- Investigate which additional measurements may improve predictions.
- Study the impact of measurement availability, timing, simulated delays, and acquisition costs.
- Explore evidence-aware decisions between predicting, acquiring information, and abstaining.

## Dataset

**PhysioNet / Computing in Cardiology Challenge 2019**

Official dataset: https://physionet.org/content/challenge-2019/1.0.0/

The dataset contains real, deidentified, hourly ICU observations and early-sepsis labels. We use it for retrospective research experiments, not clinical deployment.

## Research Baselines

- **Clinical prediction:** Logistic Regression and XGBoost.
- **Active sensing reference:** SepsisLab (KDD 2024).
- **Temporal sensing reference:** Risk-Averse Active Sensing (NeurIPS 2023).

SepsisLab: https://pmc.ncbi.nlm.nih.gov/articles/PMC11470769/

Risk-Averse Active Sensing: https://proceedings.neurips.cc/paper_files/paper/2023/hash/1498a03a04f9bcd3a7d44058fc5dc639-Abstract-Conference.html

Published methods will be adapted or reproduced where feasible. Their original reported results are not directly comparable across different datasets or protocols.

## Proposed System

```text
Patient observations
        |
        v
Clinical prediction model
        |
        v
Uncertainty and evidence evaluation
        |
        v
Predict / Acquire information / Abstain
        |
        v
Re-evaluate after new information
```

## Research Questions

1. Can calibrated uncertainty help identify insufficient evidence for reliable early sepsis prediction?
2. How do availability, timing, delay, and cost affect information acquisition?
3. Can an evidence-aware policy regulate prediction, acquisition, and abstention decisions?
4. Can an improved policy outperform appropriate existing baselines under the same experimental constraints?

These questions are provisional; novelty has not yet been established.

## Planned Deliverables

- Reproducible clinical data pipeline
- Baseline prediction models
- Uncertainty evaluation
- Active information acquisition simulator
- Comparative experiments and ablations
- Interactive research demonstration
- Research documentation and manuscript draft

## Project Status

**Phase:** Completed retrospective validation study.

Completed validation milestones include G1 mask-aware headroom, independent
calibration, selective prediction, robustness analysis, training-derived VOI
acquisition, and acquisition-after-abstention analysis. The held-out test split
has not been accessed for these development decisions.

The frozen protocol, validation sensitivity analyses, reproducibility materials,
and single locked held-out evaluation are complete. The results remain
retrospective research findings and do not establish prospective clinical utility.

## Long-Term Direction

This project investigates an initial capability toward Metacognitive Healthcare AI: evaluating available evidence and regulating information-seeking decisions. Future work may extend the system to medical reports, multimodal clinical data, and more advanced evidence-aware reasoning.

## Research Use

This repository is intended for research and educational purposes only. It is not a medical device or a clinical decision-support system validated for patient care.
  
## Research Documentation

- [Baseline Comparison](docs/literature/baseline_comparison.md)
- [Research Methodology](docs/research_methodology.md)
- [System Architecture](docs/architecture/system_design.md)
