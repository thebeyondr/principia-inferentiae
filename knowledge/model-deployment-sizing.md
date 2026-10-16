# Model and deployment sizing

## Mental model

Sizing has two independent gates:

1. select a model/precision that passes quality evaluation;
2. size a serving configuration from per-rank fit and matched measured SLO-goodput.

A capacity estimate can reject a configuration, but it cannot establish production GPU
count.

## Theory

Weight bytes:

`parameters × bits_per_weight / 8 + quantization_metadata_bytes`

Per-rank required bytes include sharded/replicated weights, KV state, fixed runtime,
activations, graph/workspace reserve, fragmentation, and safety margin. Replica GPUs are:

`tensor_parallel_size × pipeline_parallel_size`

After a matched service curve:

`active_replicas = ceil(required_peak_goodput / max_SLO_goodput_per_replica)`

Then add failure and rollout replicas before multiplying by GPUs per replica.

## What must be measured

- actual per-rank allocated/reserved HBM and imbalance;
- service curve under exact workload hash, config hash, metric scope, topology, runtime,
  precision, flags, warmup, and repeated trials;
- quality artifact for the same model revision/precision;
- placement/interconnect behavior and failure-domain assumptions.

Theory yields a fit lower bound. Only the matched service curve yields a recommendation.

## Failure modes

- equating aggregate capacity division with maximum-rank fit;
- omitting quantization scales/zero points or runtime reserve;
- reusing throughput from a different prompt/output distribution;
- sizing from peak TPS without SLO-goodput;
- forgetting rollout/failure reserve or treating reserve as steady-state capacity.

## Discriminating experiments

Run the generated load matrix through the SLO knee with pinned identities. If one rank
dominates memory or collectives, change one sharding/topology variable and rerun the
equivalent workload before claiming improvement.

## Retrieval checks

1. Why is quality not a term in the replica formula?
2. What three identities must a service curve match?
3. Why can the memory lower bound be smaller than recommended GPU count?

## Sources

- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
- [vLLM bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/)
- [DistServe paper](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)

