# Engineering analysis

## Run contract

- Run ID: `long-context-bursty-static-r1`
- Input time: `2026-10-16T15:12:00-05:00`
- Code revision: `unpublished-first-pass`
- Reproducibility tier: `R1`

## Workload

- Workload hash: `66025a84cb8ee1536330fa6fe1f74a77b4d3eb97cf641a476a1dfc37c7af170a`
- Requests: 12
- Prompt tokens p50/p95/p99: 18500 / 20000 / 20000
- Output tokens p50/p95/p99: 256 / 512 / 512
- Peak one-second arrival rate: 6.000 requests/s

## Capacity and performance

- Per-rank memory fit: `True`
- Required / usable bytes per rank: 29963660452 / 77309411328
- Roofline class: `insufficient_evidence`
- Memory-fit lower bound: 1 GPU(s)
- GPUs per configured replica: 2
- Recommended GPU count: null

## Claims

### `deployment.recommendation`

- Type / status / confidence: `deployment_sizing` / `insufficient_evidence` / `high`
- Conclusion: No deployment is recommended because the configured sequence or concurrency limit is below the observed workload.
- Method: `deployment-sizing/1.0.0`
- Evidence: `sizing.workload`, `sizing.memory`, `sizing.config`, `sizing.quality_gate`
- Assumptions: Each service-curve point represents one replica at the declared TP/PP topology.; Replica capacity scales linearly only after identical replicas are measured.; Failure and rollout reserves are additive and unavailable to steady-state load.
- Limitations: A serving configuration that rejects the observed tokens or concurrency cannot be sized.
- Missing telemetry: configured max sequence and concurrency cover the observed p99 and peak concurrency
- Next experiment: Execute the generated benchmark plan and ingest the complete measured service curve.

### `memory.per_rank_fit`

- Type / status / confidence: `memory_capacity_fit` / `supported` / `low`
- Conclusion: Configured replica fits per rank: 29963660452 required bytes versus 77309411328 usable bytes; aggregate capacity lower bound is 1 GPU(s).
- Method: `memory-capacity/1.0.0`
- Evidence: `memory.model`, `memory.hardware`, `memory.serving_config`, `memory.estimate`
- Assumptions: Capacity uses configured maximum concurrent sequences and sequence length.; The result is a maximum-rank estimate; aggregate capacity is only a lower bound.; HBM capacity fit is separate from HBM bandwidth and compute throughput.; Pipeline stages are approximated as equal partitions plus declared imbalance.
- Limitations: This static estimate does not establish allocatability or SLO-goodput.; Runtime reserve accuracy depends on the supplied measured or declared reserve.
- Missing telemetry: none
- Next experiment: Run the pinned serving benchmark and record peak allocated HBM per rank.

### `roofline.classification`

- Type / status / confidence: `kernel_or_phase_roofline` / `insufficient_evidence` / `none`
- Conclusion: No roofline bottleneck classification is supported by the supplied evidence.
- Method: `empirical-roofline/1.0.0`
- Evidence: `roofline.measurement`, `roofline.hardware`
- Assumptions: Roofline classification is valid only for a measured kernel or defined phase.; Generic GPU utilization is not evidence of compute saturation.
- Limitations: HBM capacity pressure, if known, is reported separately and cannot select a roofline class.
- Missing telemetry: measured FLOPs; measured HBM bytes; measured duration; raw artifact
- Next experiment: Profile the named kernel or phase and record FLOPs, HBM bytes moved, duration, precision, and the raw-artifact digest.

### `workload.characterization`

- Type / status / confidence: `workload_characterization` / `supported` / `high`
- Conclusion: Observed 12 requests; prompt p99=20000 tokens, output p99=512 tokens, and peak one-second arrival rate=6.000 requests/s.
- Method: `workload-characterization/1.0.0`
- Evidence: `workload.trace`, `workload.slo`
- Assumptions: Nearest-rank percentiles are computed over the supplied observation window.; One-second arrival buckets are aligned to the first observed arrival.
- Limitations: The trace describes the supplied window, not future demand.
- Missing telemetry: none
- Next experiment: none

## Benchmark plan

- Status: `required`
- Workload hash: `66025a84cb8ee1536330fa6fe1f74a77b4d3eb97cf641a476a1dfc37c7af170a`
- Config hash: `064c645e2eb16c2b8eef0d307d07f12d65d80900775272d02f481d5bf53121f3`
- Load points: 3.0, 4.5, 6.0, 7.5 requests/s
- Command template: `vllm bench serve --backend vllm --model ${PINNED_MODEL_PATH} --dataset-name timed_trace --dataset-path ${VLLM_TIMED_TRACE_PATH} --num-prompts 12 --self-timed --timed-trace-sec-multiplier ${TIME_MULTIPLIER} --timed-trace-chunk-hash-size 512 --max-concurrency 8 --seed 20261016`
