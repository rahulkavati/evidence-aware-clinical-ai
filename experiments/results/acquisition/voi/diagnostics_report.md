# VOI estimator diagnostics and training-size evaluation

- No validation or held-out test data were used to fit these diagnostics.
- One independent predictor-fit cohort was held fixed and kept disjoint from every policy-fitting cohort.
- The estimator, `min_examples=30`, shrinkage=100, and cost weight=0 were not changed.
- Benefits are squared error before reveal minus squared error after reveal.
- Values are shown with sufficient precision to distinguish small nonzero values from true zero.

## Why the 100-patient development policy acquired zero measurements

The existing development command samples 100 training patients and then assigns 20% to the policy cohort, leaving only 20 policy-training patients. Feature-level counts can therefore remain below `min_examples=30`; suppressed estimates are set exactly to zero before policy selection. A nonzero estimate must also be positive after the configured simulated cost. This diagnostic separates those mechanisms.

### Actual 100-patient development reference

- Development sample: 100 patients
- Independent predictor-fit cohort: 80 patients
- Policy-training cohort: 20 patients
- Training outcome records: 213
- Patient-disjoint cohorts: yes

In this exact development reference, all feature-level empirical benefits are zero before rounding, so min-example suppression is not the only mechanism. The independent predictor produced no squared-error change between the masked and one-feature-revealed states for these sparse policy-fitting patients.

## Training-size sensitivity

| Policy patients | Predictor-fit patients | Outcome records | Patient-disjoint |
|---:|---:|---:|---|
| 20 | 22,588 | 207 | yes |
| 100 | 22,588 | 1,066 | yes |
| 500 | 22,588 | 5,021 | yes |
| 1,000 | 22,588 | 10,156 | yes |
| 3,000 | 22,588 | 30,683 | yes |

## Feature-level diagnostics

| Policy patients | Feature | Examples | Mean unshrunk benefit | SD | Median | Positive fraction | Shrunk benefit | Min-example suppression | Unshrunk sign | Shrinkage factor | Positive after cost | Unavailable |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| 20 | Lactate | 15 | -3.087280914e-99 | 1.195698756e-98 | -5.397265127e-156 | 0 | 0 | yes | negative | 0 | no | no |
| 20 | WBC | 36 | 4.270809531e-107 | 2.562485719e-106 | 1.389392649e-191 | 0.9722222222 | 1.130508405e-107 | no | positive | 0.2647058824 | yes | no |
| 20 | Creatinine | 38 | 4.076136608e-107 | 2.494109248e-106 | 2.283351158e-190 | 0.7631578947 | 1.122414428e-107 | no | positive | 0.2753623188 | yes | no |
| 20 | Platelets | 35 | 4.393648854e-107 | 2.599317716e-106 | 1.717323248e-203 | 0.7142857143 | 1.139094147e-107 | no | positive | 0.2592592593 | yes | no |
| 20 | BUN | 40 | -1.241178488e-105 | 7.85018413e-105 | -6.465766613e-218 | 0.45 | -3.546224251e-106 | no | negative | 0.2857142857 | no | no |
| 20 | Glucose | 43 | 3.602267517e-107 | 2.344743653e-106 | 2.648081482e-195 | 0.7209302326 | 1.083199323e-107 | no | positive | 0.3006993007 | yes | no |
| 100 | Lactate | 76 | -8.569377171e-94 | 7.470604437e-93 | -1.71877031e-161 | 0 | -3.700412869e-94 | no | negative | 0.4318181818 | no | no |
| 100 | WBC | 186 | 6.774464474e-105 | 9.227762125e-104 | 1.389392649e-191 | 0.9247311828 | 4.405770602e-105 | no | positive | 0.6503496503 | yes | no |
| 100 | Creatinine | 188 | 6.705994022e-105 | 9.183302262e-104 | 3.612983727e-201 | 0.75 | 4.377523876e-105 | no | positive | 0.6527777778 | yes | no |
| 100 | Platelets | 194 | -5.982368222e-103 | 8.10649382e-102 | 9.371751756e-207 | 0.7268041237 | -3.947549099e-103 | no | negative | 0.6598639456 | no | no |
| 100 | BUN | 191 | -2.796079412e-104 | 3.828265087e-103 | -4.116380554e-236 | 0.4607329843 | -1.83522738e-104 | no | negative | 0.6563573883 | no | no |
| 100 | Glucose | 231 | 5.457700492e-105 | 8.284627874e-104 | 1.621275947e-200 | 0.6883116883 | 3.80884838e-105 | no | positive | 0.6978851964 | yes | no |
| 500 | Lactate | 329 | -0.003013614592 | 0.05465848809 | -5.535902369e-160 | 0.003039513678 | -0.002311140328 | no | negative | 0.7668997669 | no | no |
| 500 | WBC | 888 | 1.093367969e-08 | 3.258206786e-07 | 9.723893538e-193 | 0.9177927928 | 9.827031945e-09 | no | positive | 0.8987854251 | yes | no |
| 500 | Creatinine | 895 | 1.08485495e-08 | 3.245524696e-07 | 1.162961863e-200 | 0.7418994413 | 9.75824302e-09 | no | positive | 0.8994974874 | yes | no |
| 500 | Platelets | 905 | -6.352732068e-09 | 1.911219756e-07 | 7.362034234e-201 | 0.7767955801 | -5.720619424e-09 | no | negative | 0.9004975124 | no | no |
| 500 | BUN | 912 | -0.001096157915 | 0.03310324655 | -3.649627288e-228 | 0.4396929825 | -0.0009878419153 | no | negative | 0.9011857708 | no | no |
| 500 | Glucose | 1,092 | -0.0009144718465 | 0.03021910968 | 1.836536719e-204 | 0.6639194139 | -0.0008377544097 | no | negative | 0.9161073826 | no | no |
| 1,000 | Lactate | 642 | -0.001544360126 | 0.03912803976 | -1.612827003e-162 | 0.001557632399 | -0.001336225338 | no | negative | 0.8652291105 | no | no |
| 1,000 | WBC | 1,781 | 5.451492174e-09 | 2.300663825e-07 | 1.276351549e-193 | 0.9129702414 | 5.161673345e-09 | no | positive | 0.9468367889 | yes | no |
| 1,000 | Creatinine | 1,825 | 5.320247564e-09 | 2.27281978e-07 | 1.588149383e-199 | 0.7457534247 | 5.043871068e-09 | no | positive | 0.9480519481 | yes | no |
| 1,000 | Platelets | 1,810 | -3.176366034e-09 | 1.351436405e-07 | 2.391048787e-200 | 0.7779005525 | -3.010064147e-09 | no | negative | 0.9476439791 | no | no |
| 1,000 | BUN | 1,868 | -0.0005351691747 | 0.02313020979 | -5.382467956e-260 | 0.4587794433 | -0.0005079756191 | no | negative | 0.9491869919 | no | no |
| 1,000 | Glucose | 2,230 | -0.0004478041508 | 0.0211466001 | 3.510685195e-204 | 0.669058296 | -0.0004285850886 | no | negative | 0.9570815451 | no | no |
| 3,000 | Lactate | 1,929 | -0.00051398643 | 0.02257301241 | -7.194552137e-163 | 0.001036806636 | -0.0004886544226 | no | negative | 0.9507146378 | no | no |
| 3,000 | WBC | 5,436 | 6.598819557e-05 | 0.004865127428 | 1.805800012e-193 | 0.9111479029 | 6.479621227e-05 | no | positive | 0.9819364162 | yes | no |
| 3,000 | Creatinine | 5,523 | 2.672414756e-06 | 0.0001979928976 | 2.912043181e-199 | 0.7416259279 | 2.624888262e-06 | no | positive | 0.982215899 | yes | no |
| 3,000 | Platelets | 5,472 | -1.707315872e-06 | 0.0001262175118 | 1.867950571e-200 | 0.7748538012 | -1.676674883e-06 | no | negative | 0.9820531228 | no | no |
| 3,000 | BUN | 5,685 | -0.0001774939331 | 0.01325931478 | -6.164552263e-234 | 0.4525945471 | -0.000174425758 | no | negative | 0.9827139153 | no | no |
| 3,000 | Glucose | 6,638 | -0.0001479025352 | 0.01225827378 | 1.987877004e-203 | 0.6758059657 | -0.0001457074842 | no | negative | 0.9851588008 | no | no |

## Actual development-reference feature diagnostics

| Feature | Examples | Mean unshrunk benefit | Shrunk benefit | Min-example suppression | Unshrunk sign | Positive after cost |
|---|---:|---:|---:|---|---|---|
| Lactate | 19 | 0 | 0 | yes | zero | no |
| WBC | 35 | 0 | 0 | no | zero | no |
| Creatinine | 38 | 0 | 0 | no | zero | no |
| Platelets | 35 | 0 | 0 | no | zero | no |
| BUN | 40 | 0 | 0 | no | zero | no |
| Glucose | 46 | 0 | 0 | no | zero | no |

## Interpretation

- `min_examples_suppressed=yes` means the estimator deliberately returns exactly zero, regardless of the raw empirical mean.
- `unshrunk_sign=negative` means the empirical mean benefit is negative before shrinkage and rounding.
- `positive after cost=no` is evaluated using the configured simulated cost; this is not an observed clinical cost.
- Larger policy cohorts can reduce suppression and shrinkage, but any change is a training-size sensitivity result, not a validated improvement.
- Historical feature availability remains an oracle retrospective assumption, and the independent predictor remains a policy-fitting model rather than the deployed validation predictor.