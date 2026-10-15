# Research Problem

**Project:** Evidence-Aware Clinical AI  
**Status:** Initial research definition — subject to experimental validation

## Background

Healthcare AI systems often make predictions using incomplete patient information. In clinical settings, some measurements may be missing, delayed, or unavailable at the time a prediction is needed.

Although machine-learning models can still produce predictions, the reliability of those predictions may vary depending on the available information.

This project investigates whether an AI system can evaluate its current evidence and decide what to do next.

## Problem Statement

The main problem we want to study is:

**Can a clinical AI system determine whether its available information is sufficient for a reliable prediction, identify additional measurements that may improve the prediction, and decide when to predict, acquire more information, or abstain?**

We will initially investigate this problem using early sepsis prediction from structured ICU observations.

## Existing Research

Previous research has already investigated uncertainty quantification, missing-data modeling, active clinical sensing, and cost-aware information acquisition.

SepsisLab (KDD 2024) is a particularly relevant reference because it combines sepsis prediction, uncertainty estimation, and active sensing.

Our goal is not to claim that these individual capabilities are new. Instead, we want to investigate how evidence evaluation and information-acquisition decisions behave under explicit measurement availability, timing, delay, and cost constraints.

## Proposed Approach

The proposed system will:

1. Analyze currently available patient measurements.
2. Estimate sepsis risk and predictive uncertainty.
3. Evaluate whether additional information could improve the prediction.
4. Select an eligible measurement when acquisition is expected to be useful.
5. Re-evaluate the prediction after receiving the measurement.
6. Predict or abstain according to a validated decision policy.

The system will initially be evaluated through retrospective experiments rather than real clinical deployment.

## Dataset

**PhysioNet / Computing in Cardiology Challenge 2019, version 1.0.0**

The dataset contains real, deidentified ICU observations recorded over time, with missing measurements and early-sepsis prediction labels.

Official source: https://physionet.org/content/challenge-2019/1.0.0/

## Expected Research Contribution

The initial contribution will be a reproducible framework for evaluating evidence-aware acquisition and decision regulation.

We will investigate how acquisition methods perform under different measurement availability and timing assumptions.

If the experiments reveal a meaningful limitation in existing approaches, we will develop and evaluate a possible improvement.

Novelty has not yet been established.

## Long-Term Research Direction

This project is intended as an initial stage toward Metacognitive Healthcare AI.

The longer-term goal is to investigate AI systems that can monitor the reliability of their own assessments, identify missing or conflicting evidence, request useful additional information, and regulate their next actions.

Future research may extend the current structured-data approach to medical reports, clinical notes, and multimodal patient information.

## Limitations

The project uses retrospective clinical records. Historical measurements do not establish whether an unrecorded test could have been obtained on demand.

Measurement delays and costs may be simulated and must be reported as experimental assumptions.

The system is intended for research and is not validated for clinical decision-making.