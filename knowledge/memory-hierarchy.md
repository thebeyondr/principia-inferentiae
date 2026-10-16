# Memory hierarchy

## Mental model

Capacity asks whether state can reside somewhere. Bandwidth asks how quickly bytes cross
a boundary. Locality and overlap decide which boundary matters. These are separate
questions even when both involve HBM.

## Theory

Transfer time lower bound:

`time [s] ≥ bytes_moved [B] / bandwidth [B/s]`

Operational intensity:

`arithmetic_intensity [FLOP/B] = FLOPs / bytes_moved_from_the_named_level`

The denominator must name the level: HBM traffic is not host-to-device traffic, and
cache-line traffic is not logical tensor size.

## What must be measured

- allocated and reserved capacity per rank;
- achieved HBM, interconnect, and host-device bytes/s over an aligned scope;
- cache hit behavior or profiler counters where available;
- transfers, collectives, stalls, overlap, GPU clocks, power, and thermal state.

Datasheet bandwidth is a theoretical bound. Principia roofline inputs require empirical
ceilings for the declared hardware state and precision.

## Failure modes

- claiming “memory-bound” because HBM is nearly full;
- dividing tensor bytes by time while ignoring rereads and cache behavior;
- substituting generic GPU utilization for compute throughput;
- comparing empirical kernel traffic with a vendor peak from a different state;
- ignoring interconnect or host bottlenecks in multi-rank service.

## Discriminating experiments

Measure the same named scope while varying arithmetic intensity without changing
precision. Separately vary model/KV capacity at fixed kernel shape. The first probes
bandwidth/compute; the second probes fit and allocator behavior.

## Retrieval checks

1. Can 95% HBM capacity imply HBM-bandwidth-bound execution?
2. Which byte counter belongs in a claimed HBM roofline?
3. Why must clocks/power/thermal state accompany an empirical ceiling?

## Sources

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)

