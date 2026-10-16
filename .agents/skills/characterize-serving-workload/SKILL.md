---
name: characterize-serving-workload
description: Convert request-trace JSONL and an explicit serving SLO into an order-invariant token, traffic, burst, prefix-reuse, concurrency, and SLO profile. Use when characterizing an inference workload or traffic trace.
license: Apache-2.0
compatibility: Requires Python 3.11+ and the principia CLI.
---

# Characterize serving workload

Use before memory analysis, benchmarking, sizing, or incident comparison.

Run:

`principia characterize-workload --run <run.yaml> --trace <trace.jsonl> --slo <slo.yaml> --output <workload.json>`

Check:

- prompt and output distributions remain separate;
- arrival rate, burst window, tenant count, and prefix reuse are present;
- TTFT, ITL, TPOT, E2E, per-user TPS, aggregate TPS, and goodput are not conflated;
- missing completion timestamps remain explicit.

Pass the emitted workload hash unchanged to benchmark and sizing commands. Do not summarize “20K tokens” without saying whether they are input or generated tokens.

For a matched vLLM timed-trace replay, materialize deterministic prefix chunk hashes:

`principia prepare-vllm-trace --trace <trace.jsonl> --chunk-hash-size 512 --output <vllm-trace.jsonl>`

