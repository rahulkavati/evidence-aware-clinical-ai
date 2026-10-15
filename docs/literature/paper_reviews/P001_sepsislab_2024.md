# P001 — SepsisLab (KDD 2024)

**Title:** SepsisLab: Early Sepsis Prediction with Uncertainty Quantification and Active Sensing

**Authors:** Changchang Yin, Pin-Yu Chen, Bingsheng Yao, Dakuo Wang, Jeffrey Caterino, Ping Zhang

**Venue:** ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024

**DOI:** 10.1145/3637528.3671586

**Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11470769/

**Code:** https://github.com/yinchangchang/SepsisLab

**Review status:** Initial source-based review

## 1. Research Problem

Sepsis prediction systems often operate with incomplete clinical measurements. Missing observations introduce uncertainty that can affect prediction reliability.

SepsisLab investigates how uncertainty from missing clinical information propagates into prediction outputs and how additional measurements can reduce that uncertainty.

## 2. Proposed Method

The system combines:

- A probabilistic clinical-data imputation model.
- A neural sepsis prediction model.
- Uncertainty propagation through the prediction model.
- Robust Active Sensing (RAS) for recommending informative measurements.
- An interactive clinical research interface.

## 3. Datasets

The authors evaluated the approach using:

- MIMIC-III
- AmsterdamUMCdb
- OSUWMC

The original experiments did not use our selected PhysioNet Challenge 2019 dataset.

## 4. Published Results

The following values are AUROC scores from Table 3 of the paper, using an 8% active-sensing ratio.

| Method | MIMIC-III | AmsterdamUMCdb | OSUWMC |
|---|---:|---:|---:|
| Random sensing | 0.785 | 0.793 | 0.811 |
| Monte Carlo | 0.812 | 0.817 | 0.886 |
| VAT | 0.822 | 0.823 | 0.916 |
| RAS | 0.835 | 0.840 | 0.929 |

**Source:** Original paper, Section 5.3.1, Table 3.

## 5. Main Findings

The authors reported that:

- Uncertainty propagated from missing observations was especially important early in patient admissions.
- Active sensing improved prediction performance compared with random sensing at the evaluated acquisition budgets.
- The proposed RAS method achieved the highest AUROC among the tested sensing methods in Table 3.
- Adversarial regularization improved the model's local smoothness and supported uncertainty propagation.

## 6. Relevance to Our Project

SepsisLab is our main methodological reference because it already combines uncertainty estimation, clinical prediction, and active sensing.

We should not claim that recommending additional clinical measurements is a novel contribution.

Our proposed work investigates evidence sufficiency and acquisition decisions under explicit availability, timing, simulated delay, and cost constraints.

## 7. Planned Comparison

We intend to:

1. Establish a prediction baseline on PhysioNet Challenge 2019.
2. Implement simple acquisition policies.
3. Assess whether SepsisLab's RAS method can be faithfully adapted.
4. Evaluate acquisition methods under common experimental conditions.
5. Investigate whether an evidence-aware predict/acquire/abstain policy offers measurable benefits.

## 8. Limitations of Direct Comparison

The original SepsisLab datasets, preprocessing, and prediction protocol differ from PhysioNet Challenge 2019.

Therefore, the published AUROC values cannot be treated as directly comparable benchmark scores for our dataset.

Any adapted implementation must be clearly identified as such.

## 9. Open Research Questions

- How does RAS perform under explicit measurement availability restrictions?
- How sensitive is acquisition performance to simulated turnaround delays?
- Can an explicit stopping or abstention policy improve selective prediction behavior?
- Does an evidence-aware policy outperform appropriately constraint-aware acquisition baselines?

These questions are provisional and do not establish novelty.

## 10. Citation

Yin, C., Chen, P.-Y., Yao, B., Wang, D., Caterino, J., and Zhang, P. (2024). SepsisLab: Early Sepsis Prediction with Uncertainty Quantification and Active Sensing. Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 6158–6168. https://doi.org/10.1145/3637528.3671586