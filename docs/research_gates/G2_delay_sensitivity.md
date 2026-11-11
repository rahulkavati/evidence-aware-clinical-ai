# G2 Delay and Cost Sensitivity

## Cost sensitivity

The training-derived VOI validation analysis evaluated dimensionless simulated
cost weights of `0.0000`, `0.0001`, `0.0005`, and `0.0010`. Mean acquisitions and
coverage decreased as the simulated cost weight increased. These values are
policy-analysis assumptions, not clinical costs.

## Delay limitation

The PhysioNet retrospective records do not provide test-order timestamps,
result-availability timestamps, or turnaround-time distributions. Therefore a
prospective acquisition delay cannot be empirically estimated without inventing
clinical timing data or changing the task definition.

The project does not fabricate delay results. The acquisition findings are
reported as instantaneous historical-information revelation under an explicit
oracle-availability assumption. Prospective delay sensitivity is a required
follow-up experiment with a dataset that contains order and result timing.
