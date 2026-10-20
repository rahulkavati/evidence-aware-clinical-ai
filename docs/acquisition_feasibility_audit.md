# Measurement-Acquisition Feasibility Audit

**Dataset:** PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0<br>
**Source:** Official PhysioNet Challenge 2019 training sets A and B<br>
**Patients audited:** 40,336<br>
**Full-record sepsis-positive patients:** 2,932

## Time Convention and Definitions

Each row is an hourly observation indexed by ICULOS, with the first hour numbered 1. A checkpoint t includes rows with ICULOS <= t, inclusive. Eligibility requires a row at ICULOS == t, determined from the truncated rows only.

The 8 vital-sign features are analyzed separately from the 26 laboratory/test features. At each checkpoint, feature availability is evaluated only using the patient's rows at or before that checkpoint. A patient without a row exactly at the checkpoint is not in that checkpoint's eligible cohort; a later row cannot confer eligibility.

A historically observed measurement is a finite, non-null value in an eligible patient's history through the checkpoint. Each such laboratory cell is retrospectively maskable in a simulation. A feature with no such value is unobserved and has an unknown value; this audit does not assume it could have been ordered or obtained on demand. Full-record sepsis status is used only to describe retrospective positive-case retention, not to determine eligibility or acquisition state.

## Checkpoint Results

### Overall cohort

| ICU hour | Eligible patients | 1+ labs, n (%) | 3+ labs, n (%) | Positive cases retained, n (%) |
|---:|---:|---:|---:|---:|
| 6 | 39,794 | 32,227 (80.98%) | 25,243 (63.43%) | 2,886/2,932 (98.43%) |
| 12 | 39,386 | 36,948 (93.81%) | 33,506 (85.07%) | 2,436/2,932 (83.08%) |
| 24 | 31,101 | 30,383 (97.69%) | 29,832 (95.92%) | 1,854/2,932 (63.23%) |

### By hospital

| ICU hour | Hospital | Eligible patients | 1+ labs, n (%) | 3+ labs, n (%) | Positive cases retained, n (%) |
|---:|:---|---:|---:|---:|---:|
| 6 | Hospital A | 19,853 | 16,406 (82.64%) | 14,223 (71.64%) | 1,749/1,790 (97.71%) |
| 6 | Hospital B | 19,941 | 15,821 (79.34%) | 11,020 (55.26%) | 1,137/1,142 (99.56%) |
| 12 | Hospital A | 19,918 | 19,023 (95.51%) | 18,149 (91.12%) | 1,546/1,790 (86.37%) |
| 12 | Hospital B | 19,468 | 17,925 (92.07%) | 15,357 (78.88%) | 890/1,142 (77.93%) |
| 24 | Hospital A | 16,240 | 16,147 (99.43%) | 16,053 (98.85%) | 1,150/1,790 (64.25%) |
| 24 | Hospital B | 14,861 | 14,236 (95.79%) | 13,779 (92.72%) | 704/1,142 (61.65%) |

### Laboratory feature availability

Patient-level availability among eligible patients, pooled across hospitals:

| Laboratory feature | ICU hour 6 | ICU hour 12 | ICU hour 24 |
|:---|---:|---:|---:|
| BaseExcess | 9,536/39,794 (23.96%) | 11,165/39,386 (28.35%) | 10,603/31,101 (34.09%) |
| HCO3 | 10,388/39,794 (26.10%) | 15,832/39,386 (40.20%) | 16,042/31,101 (51.58%) |
| FiO2 | 13,534/39,794 (34.01%) | 15,037/39,386 (38.18%) | 14,019/31,101 (45.08%) |
| pH | 13,441/39,794 (33.78%) | 15,757/39,386 (40.01%) | 14,892/31,101 (47.88%) |
| PaCO2 | 12,915/39,794 (32.45%) | 15,302/39,386 (38.85%) | 14,482/31,101 (46.56%) |
| SaO2 | 7,766/39,794 (19.52%) | 9,940/39,386 (25.24%) | 9,946/31,101 (31.98%) |
| AST | 6,234/39,794 (15.67%) | 9,574/39,386 (24.31%) | 10,129/31,101 (32.57%) |
| BUN | 18,020/39,794 (45.28%) | 28,966/39,386 (73.54%) | 29,315/31,101 (94.26%) |
| Alkalinephos | 6,189/39,794 (15.55%) | 9,471/39,386 (24.05%) | 10,004/31,101 (32.17%) |
| Calcium | 15,813/39,794 (39.74%) | 25,319/39,386 (64.28%) | 25,631/31,101 (82.41%) |
| Chloride | 11,284/39,794 (28.36%) | 16,845/39,386 (42.77%) | 16,982/31,101 (54.60%) |
| Creatinine | 16,939/39,794 (42.57%) | 28,196/39,386 (71.59%) | 29,258/31,101 (94.07%) |
| Bilirubin_direct | 776/39,794 (1.95%) | 1,145/39,386 (2.91%) | 1,345/31,101 (4.32%) |
| Glucose | 25,728/39,794 (64.65%) | 33,712/39,386 (85.59%) | 29,859/31,101 (96.01%) |
| Lactate | 7,340/39,794 (18.45%) | 9,350/39,386 (23.74%) | 9,569/31,101 (30.77%) |
| Magnesium | 14,272/39,794 (35.86%) | 25,075/39,386 (63.66%) | 26,485/31,101 (85.16%) |
| Phosphate | 10,549/39,794 (26.51%) | 18,782/39,386 (47.69%) | 20,300/31,101 (65.27%) |
| Potassium | 20,988/39,794 (52.74%) | 31,001/39,386 (78.71%) | 29,443/31,101 (94.67%) |
| Bilirubin_total | 6,059/39,794 (15.23%) | 9,402/39,386 (23.87%) | 9,991/31,101 (32.12%) |
| TroponinI | 4,458/39,794 (11.20%) | 5,803/39,386 (14.73%) | 5,082/31,101 (16.34%) |
| Hct | 19,997/39,794 (50.25%) | 30,304/39,386 (76.94%) | 29,085/31,101 (93.52%) |
| Hgb | 18,301/39,794 (45.99%) | 28,953/39,386 (73.51%) | 28,867/31,101 (92.82%) |
| PTT | 10,416/39,794 (26.17%) | 15,368/39,386 (39.02%) | 15,517/31,101 (49.89%) |
| WBC | 16,744/39,794 (42.08%) | 27,477/39,386 (69.76%) | 28,728/31,101 (92.37%) |
| Fibrinogen | 2,148/39,794 (5.40%) | 3,116/39,386 (7.91%) | 3,415/31,101 (10.98%) |
| Platelets | 16,743/39,794 (42.07%) | 27,749/39,386 (70.45%) | 28,750/31,101 (92.44%) |

### Distribution of distinct observed laboratory features

Eligible patient counts by the number of distinct laboratory/test features seen by each checkpoint:

| Distinct laboratory features | ICU hour 6 | ICU hour 12 | ICU hour 24 |
|---:|---:|---:|---:|
| 0 | 7,567 | 2,438 | 718 |
| 1 | 5,804 | 2,585 | 389 |
| 2 | 1,180 | 857 | 162 |
| 3 | 575 | 439 | 55 |
| 4 | 1,136 | 592 | 90 |
| 5 | 1,294 | 762 | 163 |
| 6 | 900 | 663 | 180 |
| 7 | 1,041 | 729 | 209 |
| 8 | 1,839 | 1,044 | 138 |
| 9 | 1,237 | 1,261 | 633 |
| 10 | 1,215 | 1,647 | 1,153 |
| 11 | 1,716 | 2,940 | 2,472 |
| 12 | 1,077 | 1,480 | 1,179 |
| 13 | 1,637 | 2,333 | 1,595 |
| 14 | 2,968 | 4,507 | 3,795 |
| 15 | 1,942 | 2,998 | 2,959 |
| 16 | 1,559 | 2,309 | 2,349 |
| 17 | 1,449 | 2,653 | 2,878 |
| 18 | 1,225 | 2,287 | 2,824 |
| 19 | 920 | 1,691 | 2,338 |
| 20 | 543 | 1,100 | 1,604 |
| 21 | 418 | 795 | 1,171 |
| 22 | 306 | 608 | 896 |
| 23 | 164 | 414 | 689 |
| 24 | 69 | 186 | 324 |
| 25 | 13 | 67 | 136 |
| 26 | 0 | 1 | 2 |

## Limitations

- This is a retrospective audit of the PhysioNet 2019 Challenge training sets, not a prospective or deployment study.
- The hourly files do not establish whether a non-null value represents a newly performed test, a carried-forward value, or a clinically orderable measurement at that time. Cell counts therefore describe maskable recorded entries, not verified laboratory orders or independent assays.
- Missing values are unknown. The audit cannot estimate the value, cost, delay, or feasibility of acquiring an unrecorded test, and it does not treat unobserved tests as available acquisition candidates.
- Recorded availability reflects historical clinical practice and selection, which may differ by patient, hospital, and severity; it is not a randomized measure of test utility.
- Sepsis-positive status uses any positive label in a patient's full record. Its use here is limited to retrospective cohort characterization and positive-case retention; future labels must remain excluded from policy state.
- This feasibility audit does not train a model or evaluate predictive benefit from acquiring measurements.

## Interpretation and Recommendation

The proportion with at least one or three historically observed laboratory features quantifies how often a retrospective masking experiment can hide an existing lab entry at each checkpoint. It does not quantify how often a new measurement could have been acquired. Availability should be treated as patient- and time-dependent, and conclusions should not be generalized to on-demand clinical testing.

Next, run controlled retrospective masking experiments only on measurements actually present by the checkpoint. Compare no acquisition with policies that reveal eligible hidden historical entries under explicit budgets; keep never-observed tests unavailable or unknown. Use patient-level train/validation/test splits, prevent full-record labels from entering policy state, and report performance separately by hospital and checkpoint. Any simulated delay, cost, or acquisition success assumption should be clearly labeled and tested in sensitivity analyses.

Aggregate machine-readable outputs are in `outputs/acquisition_feasibility/`. No patient-level output is produced.
