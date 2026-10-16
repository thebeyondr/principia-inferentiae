# Roofline analysis

## Mental model

Roofline compares a **measured kernel or named phase** with empirical compute and HBM
bandwidth ceilings for one precision. It does not classify an application from generic
GPU utilization, and it does not answer whether model state fits.

## Theory

`arithmetic_intensity [FLOP/B] = measured_FLOPs / measured_HBM_bytes`

`machine_balance [FLOP/B] = empirical_compute [FLOP/s] / empirical_HBM [B/s]`

`roofline [FLOP/s] = min(empirical_compute, empirical_HBM × arithmetic_intensity)`

Principia classifies a sufficiently filled scope below the ridge as
`hbm_bandwidth_bound`, above it as `compute_bound`, and near it as `mixed`. Efficiency
below the declared threshold is `underfilled`. Missing or incompatible measurements are
`insufficient_evidence`.

## What must be measured

- same-scope FLOPs, HBM bytes, and duration;
- precision-specific empirical compute ceiling;
- empirical HBM ceiling under compatible clocks/power/thermal state;
- raw profiler artifact digest and named scope;
- capacity pressure separately, if relevant.

The equations are theory. Counter definitions, overlap, scope selection, and ceilings are
measurement. If achieved throughput exceeds the supplied roofline by more than tolerance,
the evidence conflicts and the result is inconclusive.

## Failure modes

- assigning the whole request one class;
- using logical tensor bytes instead of measured HBM traffic;
- treating low roofline efficiency as HBM-bound;
- using vendor peak compute with an empirical bandwidth ceiling;
- treating high HBM allocation as high HBM bandwidth.

## Discriminating experiments

Hold precision and hardware state fixed, sweep shape/batch to move arithmetic intensity,
and re-profile the same scope. Use end-to-end metrics only to establish whether the scope
matters to service behavior.

## Retrieval checks

1. What is the ridge point and what are its units?
2. Why does an underfilled kernel not support compute- versus bandwidth-bound language?
3. What evidence ties a kernel classification to an E2E regression?

## Sources

- [Roofline paper](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)

