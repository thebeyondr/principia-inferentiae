# Inference decision brief

- Run: `long-context-bursty-static-r1`
- Evidence tier: `R1`
- Method revision: `principia-methods/1.0.0`

## Decisions
- **insufficient_evidence — deployment_sizing:** No deployment is recommended because the configured sequence or concurrency limit is below the observed workload.
  Confidence: `high`.
  Missing: configured max sequence and concurrency cover the observed p99 and peak concurrency.
  Next: Execute the generated benchmark plan and ingest the complete measured service curve.
- **supported — memory_capacity_fit:** Configured replica fits per rank: 29963660452 required bytes versus 77309411328 usable bytes; aggregate capacity lower bound is 1 GPU(s).
  Confidence: `low`.
  Next: Run the pinned serving benchmark and record peak allocated HBM per rank.
- **insufficient_evidence — kernel_or_phase_roofline:** No roofline bottleneck classification is supported by the supplied evidence.
  Confidence: `none`.
  Missing: measured FLOPs; measured HBM bytes; measured duration; raw artifact.
  Next: Profile the named kernel or phase and record FLOPs, HBM bytes moved, duration, precision, and the raw-artifact digest.
- **supported — workload_characterization:** Observed 12 requests; prompt p99=20000 tokens, output p99=512 tokens, and peak one-second arrival rate=6.000 requests/s.
  Confidence: `high`.

## Guardrail

Null recommendations and inconclusive findings are intentional fail-closed outcomes.
