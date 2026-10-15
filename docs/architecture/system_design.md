# Stage 1 — System Architecture

**Project:** Evidence-Aware Clinical AI

**Goal:** Evaluate available patient information and determine whether to predict, acquire additional information, or abstain.

## High-Level Architecture

```mermaid
flowchart TD
    A["PhysioNet 2019 ICU Dataset"] --> B["Data Preprocessing"]
    B --> C["Clinical Prediction Model"]
    C --> D["Uncertainty Estimation"]
    D --> E["Evidence Evaluation Policy"]

    E --> F{"Next Action"}

    F -->|Sufficient evidence| G["Predict"]
    F -->|Useful information available| H["Acquire Measurement"]
    F -->|Insufficient evidence| I["Abstain"]

    H --> J["Acquisition Simulator"]
    J --> C

    G --> K["Research Evaluation"]
    I --> K
```

## Planned Components

### Data Processing
Load and preprocess hourly ICU patient observations.

### Clinical Prediction
Train Logistic Regression and XGBoost baselines.

### Uncertainty Estimation
Evaluate calibration and predictive uncertainty.

### Evidence Evaluation
Determine whether additional information may improve prediction reliability.

### Acquisition Simulator
Simulate measurement acquisition using historical observations.

### Decision Regulation
Choose between predicting, acquiring information, or abstaining.

### Evaluation
Compare prediction quality, acquisition cost, and decision reliability.

## Research Status

Architecture proposed. Implementation and experimental validation pending.