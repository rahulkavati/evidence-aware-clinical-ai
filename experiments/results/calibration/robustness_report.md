# Calibration robustness evaluation

Validation-only evaluation on the patient-disjoint selection subset. No test-set records were loaded.

- Calibration patients: 3,025
- Selection patients: 3,025
- Selection hourly observations: 116,968
- Patient-cluster bootstrap: 200 resamples (seed 42); 95% percentile intervals.
- Existing calibration metrics consistency check: passed (maximum absolute difference 1.35e-07).
- Small-sample flag: fewer than 100 rows or 30 patients in a high-risk range.

## Overall and hospital metrics

| Scope | Method | N | Positives | AUROC | AUPRC | Brier | ECE (10 bins) | Brier 95% CI | ECE 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | raw | 116,968 | 2,090 | 0.837221 | 0.099164 | 0.017957 | 0.004582 | [0.015917, 0.020023] | [0.003183, 0.008281] |
| overall | sigmoid | 116,968 | 2,090 | 0.837221 | 0.099164 | 0.016985 | 0.001608 | [0.014997, 0.018910] | [0.000978, 0.004077] |
| overall | isotonic | 116,968 | 2,090 | 0.835939 | 0.093366 | 0.016769 | 0.001345 | [0.014855, 0.018533] | [0.000622, 0.003917] |
| hospital_A | raw | 58,182 | 1,258 | 0.844268 | 0.114070 | 0.021120 | 0.005505 | [0.017784, 0.024729] | [0.002747, 0.009779] |
| hospital_A | sigmoid | 58,182 | 1,258 | 0.844268 | 0.114070 | 0.020374 | 0.003376 | [0.017345, 0.023606] | [0.001319, 0.006851] |
| hospital_A | isotonic | 58,182 | 1,258 | 0.842924 | 0.110360 | 0.019958 | 0.001831 | [0.017094, 0.023086] | [0.001116, 0.005332] |
| hospital_B | raw | 58,786 | 832 | 0.820461 | 0.087749 | 0.014827 | 0.005937 | [0.011576, 0.018525] | [0.002824, 0.012143] |
| hospital_B | sigmoid | 58,786 | 832 | 0.820461 | 0.087749 | 0.013631 | 0.002859 | [0.010825, 0.016586] | [0.000907, 0.006819] |
| hospital_B | isotonic | 58,786 | 832 | 0.819122 | 0.075611 | 0.013613 | 0.003150 | [0.011046, 0.016282] | [0.001693, 0.007187] |

## High-risk probability diagnostics

Ranges are [0.1, 0.2), [0.2, 0.5), [0.5, 0.8), and [0.8, 1.0]. Empty ranges are reported with unavailable means/frequencies. A flagged row indicates a small sample under the thresholds above; bin-level rates should not be treated as precise.

| Scope | Method | Probability range | Count | Mean probability | Observed positive frequency | Unique patients | Small sample? |
|---|---|---|---:|---:|---:|---:|---|
| overall | raw | [0.1, 0.2) | 2,881 | 0.141120 | 0.139882 | 292 | No |
| overall | raw | [0.2, 0.5) | 1,852 | 0.302704 | 0.127430 | 109 | No |
| overall | raw | [0.5, 0.8) | 332 | 0.609531 | 0.189759 | 31 | No |
| overall | raw | [0.8, 1.0] | 56 | 0.866795 | 0.107143 | 9 | Yes |
| overall | sigmoid | [0.1, 0.2) | 2,335 | 0.137735 | 0.133191 | 164 | No |
| overall | sigmoid | [0.2, 0.5) | 851 | 0.285898 | 0.170388 | 58 | No |
| overall | sigmoid | [0.5, 0.8) | 75 | 0.611768 | 0.160000 | 11 | Yes |
| overall | sigmoid | [0.8, 1.0] | 0 | — | — | 0 | Yes |
| overall | isotonic | [0.1, 0.2) | 5,159 | 0.130569 | 0.119209 | 369 | No |
| overall | isotonic | [0.2, 0.5) | 1,184 | 0.231513 | 0.158784 | 71 | No |
| overall | isotonic | [0.5, 0.8) | 0 | — | — | 0 | Yes |
| overall | isotonic | [0.8, 1.0] | 0 | — | — | 0 | Yes |
| hospital_A | raw | [0.1, 0.2) | 1,576 | 0.141712 | 0.177665 | 153 | No |
| hospital_A | raw | [0.2, 0.5) | 949 | 0.300432 | 0.147524 | 52 | No |
| hospital_A | raw | [0.5, 0.8) | 118 | 0.612511 | 0.228814 | 12 | Yes |
| hospital_A | raw | [0.8, 1.0] | 44 | 0.871562 | 0.022727 | 3 | Yes |
| hospital_A | sigmoid | [0.1, 0.2) | 1,256 | 0.136139 | 0.168790 | 82 | No |
| hospital_A | sigmoid | [0.2, 0.5) | 383 | 0.276014 | 0.185379 | 27 | Yes |
| hospital_A | sigmoid | [0.5, 0.8) | 51 | 0.631986 | 0.078431 | 4 | Yes |
| hospital_A | sigmoid | [0.8, 1.0] | 0 | — | — | 0 | Yes |
| hospital_A | isotonic | [0.1, 0.2) | 2,751 | 0.130275 | 0.153035 | 199 | No |
| hospital_A | isotonic | [0.2, 0.5) | 563 | 0.231518 | 0.161634 | 32 | No |
| hospital_A | isotonic | [0.5, 0.8) | 0 | — | — | 0 | Yes |
| hospital_A | isotonic | [0.8, 1.0] | 0 | — | — | 0 | Yes |
| hospital_B | raw | [0.1, 0.2) | 1,305 | 0.140406 | 0.094253 | 139 | No |
| hospital_B | raw | [0.2, 0.5) | 903 | 0.305092 | 0.106312 | 57 | No |
| hospital_B | raw | [0.5, 0.8) | 214 | 0.607888 | 0.168224 | 19 | Yes |
| hospital_B | raw | [0.8, 1.0] | 12 | 0.849316 | 0.416667 | 6 | Yes |
| hospital_B | sigmoid | [0.1, 0.2) | 1,079 | 0.139594 | 0.091752 | 82 | No |
| hospital_B | sigmoid | [0.2, 0.5) | 468 | 0.293987 | 0.158120 | 31 | No |
| hospital_B | sigmoid | [0.5, 0.8) | 24 | 0.568804 | 0.333333 | 7 | Yes |
| hospital_B | sigmoid | [0.8, 1.0] | 0 | — | — | 0 | Yes |
| hospital_B | isotonic | [0.1, 0.2) | 2,408 | 0.130906 | 0.080565 | 170 | No |
| hospital_B | isotonic | [0.2, 0.5) | 621 | 0.231509 | 0.156200 | 39 | No |
| hospital_B | isotonic | [0.5, 0.8) | 0 | — | — | 0 | Yes |
| hospital_B | isotonic | [0.8, 1.0] | 0 | — | — | 0 | Yes |

## Interpretation and limitations

Calibration should not be compared using overall ECE alone. Review the Brier score, discrimination, hospital-specific results, high-risk ranges, sample counts, and cluster-bootstrap intervals together. The calibration subset and selection subset contain different patients; these results do not establish performance on an external cohort. Bootstrap intervals resample patients, while all hourly observations within a selected patient remain together.
