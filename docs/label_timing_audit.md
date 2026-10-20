# First-Positive Label Timing Audit

**Dataset:** PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0<br>
**Patients audited:** 40,336<br>
**Hospitals:** Hospital A and Hospital B

## Definitions

Rows represent hourly observations indexed by ICULOS, with hour 1 as the first row. A checkpoint t is eligible only when a row exists exactly at ICULOS=t. Label timing is compared to t inclusively: at or before means <=t; after means >t.

The first positive label is the minimum `ICULOS` whose `SepsisLabel` equals 1 anywhere in that patient's complete record. Patients are eligible at a checkpoint only if their record contains a row exactly at that hour. Within eligible patients, the three reported groups are mutually exclusive and exhaustive: first positive label at or before the checkpoint, first positive label after it, and no positive label in the full record.

The ‘pre-positive-label eligible’ group means eligible patients whose first positive Challenge label occurs strictly after the checkpoint. It is a retrospective label-timing characterization, not evidence that the patient had no clinical sepsis at that checkpoint.

**Important outcome caveat:** First positive SepsisLabel timing is the first positive Challenge label, not verified true clinical onset. Full-record labels are used only for retrospective cohort analysis and are not model input or acquisition-policy information.

## Overall Results

Counts are among checkpoint-eligible patients; percentages use eligible patients as the denominator.

| ICU hour | Eligible | First positive at/before, n (%) | First positive after, n (%) | No positive label, n (%) | Pre-positive-label eligible, n (%) |
|---:|---:|---:|---:|---:|---:|
| 6 | 39,794 | 706 (1.77%) | 2,180 (5.48%) | 36,908 (92.75%) | 2,180 (5.48%) |
| 12 | 39,386 | 483 (1.23%) | 1,953 (4.96%) | 36,950 (93.82%) | 1,953 (4.96%) |
| 24 | 31,101 | 274 (0.88%) | 1,580 (5.08%) | 29,247 (94.04%) | 1,580 (5.08%) |

## Hospital Results

| ICU hour | Hospital | Eligible | First positive at/before, n (%) | First positive after, n (%) | No positive label, n (%) | Pre-positive-label eligible, n (%) |
|---:|:---|---:|---:|---:|---:|---:|
| 6 | Hospital A | 19,853 | 380 (1.91%) | 1,369 (6.90%) | 18,104 (91.19%) | 1,369 (6.90%) |
| 6 | Hospital B | 19,941 | 326 (1.63%) | 811 (4.07%) | 18,804 (94.30%) | 811 (4.07%) |
| 12 | Hospital A | 19,918 | 336 (1.69%) | 1,210 (6.07%) | 18,372 (92.24%) | 1,210 (6.07%) |
| 12 | Hospital B | 19,468 | 147 (0.76%) | 743 (3.82%) | 18,578 (95.43%) | 743 (3.82%) |
| 24 | Hospital A | 16,240 | 185 (1.14%) | 965 (5.94%) | 15,090 (92.92%) | 965 (5.94%) |
| 24 | Hospital B | 14,861 | 89 (0.60%) | 615 (4.14%) | 14,157 (95.26%) | 615 (4.14%) |

## Cross-Audit Validation

Eligible patient counts and the number of full-record positive-labeled eligible patients match `outputs/acquisition_feasibility/summary.json` for the overall cohort and both hospitals at all three checkpoints.

## Limitations and Interpretation

- `SepsisLabel` timing is a dataset label convention. The first positive hourly label is not necessarily the true biological or clinical onset time.
- Labels later in the patient record are intentionally used here only to characterize historical timing and pre-positive-label cohort size. They must never be exposed as model features, patient state, or acquisition-policy information at an earlier checkpoint.
- Patients without a row exactly at a checkpoint are excluded from that checkpoint's eligible denominator, even if records exist both before and after it.
- The report is descriptive and does not train a prediction model or establish prospective performance.

A future prediction experiment may use this audit to describe how much eligible data precedes the first positive Challenge label. It must preserve the distinction between that label and clinical onset and must enforce temporal separation in every feature and policy input.

Aggregate output: `outputs/label_timing/summary.json`. No patient-level output is written.
