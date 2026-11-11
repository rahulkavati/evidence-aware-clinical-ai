# Acquisition after abstention

Validation-only one-step retrospective reveal for cases that fail the strict selective-prediction threshold.

- Selected threshold: `0.962646`
- Held-out test accessed: `False`
- Policy: reveal the first historically eligible lab in fixed order, then re-apply the same threshold.

| Scope | Baseline coverage | Assisted coverage | Baseline risk | Assisted risk | Eligible abstentions | Rescued |
|---|---:|---:|---:|---:|---:|---:|
| pooled_all | 0.91713 | 0.91957 | 0.01038 | 0.01045 | 9422 | 285 |
| hospital_A | 0.91092 | 0.91288 | 0.01206 | 0.01214 | 4942 | 114 |
| hospital_B | 0.92328 | 0.92619 | 0.00873 | 0.00880 | 4480 | 171 |
| hour_6 | 0.95166 | 0.95334 | 0.01446 | 0.01444 | 116 | 5 |
| hour_12 | 0.96608 | 0.96710 | 0.01124 | 0.01157 | 95 | 3 |
| hour_24 | 0.97546 | 0.97718 | 0.00662 | 0.00661 | 57 | 4 |

This is a retrospective historical-information reveal, not a clinical test-ordering study. The fixed-order policy does not inspect hidden measurement values.
