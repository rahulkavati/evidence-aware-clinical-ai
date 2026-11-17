
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

**Phase:** Completed retrospective study with a single frozen final evaluation.

### Final held-out results (6,051 patients, 232,215 hourly rows)

Thresholds were selected on validation data only and frozen in
[`configs/final_protocol.json`](configs/final_protocol.json). Full report:
[`experiments/results/final/final_test_evaluation.md`](experiments/results/final/final_test_evaluation.md).

| Finding | Result |
|---|---|
| Early-warning utility, labs as recorded (official PhysioNet 2019 score) | **0.419** (95% CI 0.380–0.457) |
| Early-warning utility, all 26 labs hidden | 0.380 (0.340–0.421) |
| Value of the recorded labs (utility difference) | +0.039 (0.015–0.065) |
| AUROC, labs as recorded / labs hidden | 0.840 / 0.806 |
| Selective prediction at 1% validation risk target | 90.8% coverage, 1.05% risk; escalated 9.2% of hours, which contained 47.5% of positive hours |
| One-step lab acquisition (fixed, random, patient-specific) vs labs hidden | No utility gain; best ΔAUROC +0.002 (fixed order) |
| Oracle acquisition ceiling (uses labels) | ΔAUPRC +0.148, utility +0.009: real but small headroom |

**Main conclusion:** recorded laboratory results add meaningful predictive value,
but none of the deployable one-step acquisition policies recovers it. The
oracle ceiling shows the headroom is concentrated in a small set of rows,
which defines the open problem for future acquisition policies.

### Methodological notes

- Every access to the held-out split, including four development-phase accesses
  before the final protocol, is disclosed in
  [`experiments/results/final/test_access_ledger.json`](experiments/results/final/test_access_ledger.json).
- Earlier next-phase reports are kept for history and marked superseded: they
  scored utility at a fixed 0.5 alert cut-off and their oracle reduced to the
  fixed order.
- Only the validation patients' raw PSV files are stored locally. All experiments
  run from `data/processed/*.parquet`; `make check-data` verifies them against
  the split manifests. `bash scripts/download_physionet2019.sh` restores the raw files.

## How to Run

Requires Python 3.11–3.14. On macOS:

```bash
make setup        # once: create .venv and install pinned dependencies
make check-data   # verify processed data matches the split manifests
make test         # full test suite (148 tests)
make validate     # final evaluation on validation (about 5 minutes)
```

Further targets:

```bash
make train        # retrain the all-26-lab model + calibrators (about 1 minute)
make freeze       # re-select thresholds on validation (refuses if already frozen)
make final-test   # locked held-out evaluation; appended to the access ledger
```

`experiments/run_final_evaluation.py --stage freeze|predict|evaluate|bootstrap|report`
runs individual stages; held-out access additionally requires `--locked-test`.

### Repository layout

| Path | Contents |
|---|---|
| `src/data` | Leakage-safe temporal features, patient-level splits, lab groups, masking |
| `src/models` | Logistic regression, XGBoost, mask-aware XGBoost, sequence baselines |
| `src/uncertainty` | Platt / isotonic calibration and robustness analyses |
| `src/acquisition` | Fixed, random, VOI, risk-averse and patient-specific acquisition policies |
| `src/evaluation` | Selective prediction, PhysioNet utility, early warning, paired bootstrap |
| `experiments/` | Runnable experiment scripts; `results/` holds every committed report |
| `configs/final_protocol.json` | Frozen thresholds and artifact hashes for the final evaluation |
| `docs/` | Research problem, protocol, audits, literature and gate documents |

## Long-Term Direction

This project investigates an initial capability toward Metacognitive Healthcare AI: evaluating available evidence and regulating information-seeking decisions. Future work may extend the system to medical reports, multimodal clinical data, and more advanced evidence-aware reasoning.

## Research Use

This repository is intended for research and educational purposes only. It is not a medical device or a clinical decision-support system validated for patient care.
  
## Research Documentation

- [Baseline Comparison](docs/literature/baseline_comparison.md)
- [Research Methodology](docs/research_methodology.md)
- [System Architecture](docs/architecture/system_design.md)
