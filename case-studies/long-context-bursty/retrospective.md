# Static-analysis retrospective

## What ran

The R1 deterministic pipeline consumed the typed trace, SLO, synthetic model/hardware,
serving config, planned roofline record, quality-contract fixture, and sizing policy. No
GPU command was executed.

## Results

- 12 requests; prompt p99 **20,000 input tokens**.
- Output p99 **512 generated tokens**—not 20,000 output tokens.
- Peak one-second arrival rate: 6 requests/s; observed peak concurrency: 12.
- Shared-prefix p99: 2,048 tokens; deterministic 512-token chunk hashes were emitted for timed-trace replay.
- Logical KV: 131,072 bytes/token; configured active tokens: 164,096.
- Estimated maximum-rank requirement: 29,963,660,452 bytes.
- Declared usable synthetic capacity: 77,309,411,328 bytes/rank.
- Static aggregate capacity lower bound: 1 GPU.
- Configured topology still uses 2 GPUs/replica; the aggregate lower bound is not a topology recommendation.
- Roofline: `insufficient_evidence`; no FLOPs/HBM-byte/duration artifact exists.
- Deployment: `insufficient_evidence`; `recommended_gpu_count: null`.
  Observed peak concurrency is 12 and the configured maximum is 8, so the
  sequence/concurrency gate fails closed before any GPU count is recommended.

## Prediction versus result

The contract-level prediction passed: token directions stayed separate, static fit was
computed, roofline failed closed, and sizing refused a recommendation. The learner-owned
runtime predictions remain intentionally blank until the pinned GPU run; there is no
measured result to compare.

## What this does not establish

The synthetic model/hardware values do not establish real allocatability, TTFT, ITL,
TPOT, E2E, throughput, goodput, cost, or a bottleneck. The quality gate only exercises
the independent-gate contract. It is not a real quality result.

## Next discriminating experiment

Execute `benchmark-plan.yaml` after recording immutable runtime, model, hardware, and
environment artifacts. Ingest a service curve only if workload/config hashes and metric
scope match. Profile named prefill and decode phases before making roofline claims.

