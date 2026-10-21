# Prediction Baseline Results

**Project:** Evidence-Aware Clinical AI  
**Dataset:** PhysioNet Challenge 2019, v1.0.0  
**Evaluation:** Hourly validation set  
**Status:** Initial experimental results

## Dataset

- Training observations: 1,085,958
- Validation observations: 234,037
- Test observations: 232,215
- Patient-level splitting: 70/15/15
- Random seed: 42
- Validation positive-label prevalence: 1.7873%

## Model Comparison

| Metric      | Logistic Regression |    XGBoost |
| ----------- | ------------------: | ---------: |
| AUROC       |              0.7793 | **0.8427** |
| AUPRC       |              0.0812 | **0.1038** |
| Brier score |        **0.017219** |   0.017852 |

## Initial Findings

XGBoost achieves better discrimination than Logistic Regression on the validation set.

Logistic Regression has a lower Brier score, suggesting that probability reliability requires further investigation.

Neither model has yet been evaluated using the official PhysioNet Challenge utility score.

## Limitations

- Results are from the validation set only.
- Probability calibration has not been evaluated.
- No acquisition policy has been implemented.
- These results are not directly comparable with published SepsisLab results.
- The held-out test set has not been evaluated.

## Next Steps

1. Evaluate probability calibration.
2. Implement official challenge utility scoring.
3. Investigate selective prediction.
4. Build acquisition baselines.
