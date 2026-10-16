# Request lifecycle

## Mental model

A serving request crosses client/network, admission, queue, scheduler, model execution,
streaming, and final-response boundaries. A latency claim is uninterpretable until its
start/end boundary is named.

## Theory

For one request:

`E2E [ms] = client/network + queue + preprocessing + prefill + decode/stream + postprocessing`

This is a conceptual decomposition, not an additive percentile identity: component p99
values generally do **not** sum to E2E p99. Continuous batching also couples one request’s
phase timing to other active requests.

## What must be measured

- timestamp-aligned client send, server receive, admission, first scheduled step, first token, each subsequent token, and final byte;
- queue depth/scheduler wait, tokenization time, prefill/decode phase windows;
- runtime, model revision, precision, batching flags, workload hash, and metric scope.

Theory identifies boundaries. Measurement assigns latency to a concrete run. Generic
request duration cannot distinguish network delay from engine delay.

## Failure modes

- calling server-side execution “E2E”;
- summing separately aggregated p99 components;
- comparing traces with different prompt/output distributions;
- using unsynchronized clocks for causal ordering;
- attributing TTFT growth to prefill without queue timestamps.

## Discriminating experiments

Replay the same canonical trace twice: once through loopback and once through the real
network path. Preserve runtime/config identity and compare timestamped component deltas.

## Retrieval checks

1. Why can component p99 values not be added?
2. Which timestamps separate queue delay from prefill?
3. Is first-token latency inference-only or client-observed in the current run?

## Sources

- [Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
- [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/)
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)

