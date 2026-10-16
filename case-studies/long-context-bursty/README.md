# Synthetic long-context burst

This case exercises the first vertical slice without pretending to be a GPU benchmark.

## Scope

- prompt p99: 20,000 **input** tokens;
- output maximum: 512 **generated** tokens;
- two bursts plus sparse arrivals;
- repeated and unique prefixes across three tenants;
- synthetic model and hardware contracts;
- planned, not measured, roofline input;
- no service curve.

A workload with 20,000 generated tokens is explicitly out of scope and would receive a
different trace and workload hash.

## Reproduce

From the project root after installation:

```bash
make case
```

Expected gates:

- workload characterization: supported;
- static per-rank capacity: supported or rejected from typed inputs;
- roofline: `insufficient_evidence`;
- deployment sizing: memory-fit lower bound only,
  `recommended_gpu_count: null` because observed peak concurrency (12) exceeds
  `max_concurrent_sequences` (8);
- real execution: follow `benchmark-plan.yaml`, then create a new real model/hardware
  contract and matched service curve.

`prediction.md` remains learner-owned. `retrospective.md` records only generated static
outcomes until the GPU plan is executed.

