# Prefill and decode

## Mental model

Prefill processes the prompt and creates KV state; decode repeatedly consumes existing KV
state to produce new tokens. They share weights and hardware but have different tensor
shapes, batching opportunities, latency objectives, and likely bottlenecks.

## Theory

For dense attention, prefill attention work grows roughly with prompt-length squared,
while one decode step attends over the current sequence length. This asymptotic view
helps choose experiments; it does not predict runtime latency.

Approximate request generation time:

`generation [ms] ≈ first decode step + Σ subsequent inter-token intervals [ms]`

TPOT is an average over output tokens; ITL is a distribution over token gaps. They are
related but not interchangeable.

## What must be measured

- separately scoped prefill and decode durations;
- prompt/output token distributions and active sequence counts;
- scheduler batch composition and preemption;
- kernel/phase FLOPs, HBM bytes, duration, and precision for roofline claims;
- TTFT, ITL, TPOT, E2E, per-user TPS, aggregate TPS, and goodput.

Phase theory cannot establish that production prefill is compute-bound or decode is
bandwidth-bound. Only compatible measured scope and empirical ceilings can.

## Failure modes

- labeling every long-prompt regression “prefill compute”;
- treating high aggregate TPS as good per-user decode cadence;
- comparing decode with different output-length caps;
- using average TPOT to hide ITL tail spikes;
- assigning one roofline class to the entire server.

## Discriminating experiments

Hold model, precision, output length, and offered load fixed while sweeping prompt length;
then hold prompt length fixed while sweeping active decode sequences. Profile each named
phase and correlate it with request metrics.

## Retrieval checks

1. Why can prefill and decode require separate service objectives?
2. What evidence permits a phase roofline conclusion?
3. How can aggregate TPS rise while per-user TPS falls?

## Sources

- [Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [DistServe paper](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)

