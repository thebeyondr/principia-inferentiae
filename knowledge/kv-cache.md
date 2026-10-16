# KV cache

## Mental model

KV cache is persistent attention state for active tokens. Its capacity depends on the
number of transformer layers, **KV heads** rather than query heads, head dimension, KV
dtype, active tokens, and sharding/replication semantics.

## Theory

Logical KV bytes per token:

`2 × layers × kv_heads × head_dim × kv_dtype_bytes`

The factor `2` stores keys and values. For MHA, `kv_heads = attention_heads`; for GQA,
`1 < kv_heads < attention_heads`; for MQA, `kv_heads = 1`. Per-rank bytes then depend on
whether pipeline parallelism splits layers and tensor parallelism splits KV heads.
Tensor parallelism does not divide the layer count. `shard_layers` keeps KV heads replicated.

Configured logical demand:

`active_tokens [token] = max_sequences × max_sequence_tokens`

This is a conservative capacity input, not expected utilization.

## What must be measured

- runtime allocation granularity, block occupancy, fragmentation, prefix-cache hits;
- active sequences and tokens over time;
- per-rank allocated/reserved HBM and preemption/eviction counters;
- exact TP/PP layout and runtime KV-sharding behavior.

The formula is theory. Allocator behavior, prefix reuse, and maximum-rank imbalance are
measurement. Static bytes do not prove allocatability.

## Failure modes

- using query-head count for GQA/MQA KV;
- dividing KV by TP without declaring runtime sharding;
- omitting both K and V;
- treating context-window maximum as observed active tokens or vice versa;
- confusing KV capacity pressure with HBM bandwidth saturation.

## Discriminating experiments

Sweep concurrent sequences at fixed prompt/output lengths and record per-rank KV
occupancy, allocation failures, preemption, and latency. Repeat with prefix caching
disabled/enabled under the same prefix-reuse trace.

## Retrieval checks

1. What changes in the formula from MHA to MQA?
2. Why can aggregate KV bytes fit while one rank OOMs?
3. Which experiment separates prefix reuse from allocator fragmentation?

## Sources

- [Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
- [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)

