# Locked held-out evaluation

Single final evaluation using the frozen validation-selected threshold and fixed one-step policy.

- Held-out test accessed: `True`
- Frozen threshold: `0.9626458569`
- No threshold or policy retuning was performed on test data.

| Scope | Baseline coverage | Assisted coverage | Baseline risk | Assisted risk | Rescued |
|---|---:|---:|---:|---:|---:|
| pooled_all | 0.92401 | 0.92666 | 0.01067 | 0.01084 | 616 |
| hospital_A | 0.90892 | 0.91089 | 0.01310 | 0.01324 | 234 |
| hospital_B | 0.93984 | 0.94321 | 0.00820 | 0.00841 | 382 |
| hour_6 | 0.95245 | 0.95614 | 0.01336 | 0.01366 | 22 |
| hour_12 | 0.96697 | 0.96951 | 0.01033 | 0.01048 | 15 |
| hour_24 | 0.97334 | 0.97462 | 0.00767 | 0.00788 | 6 |

This retrospective evaluation does not establish prospective clinical utility.
