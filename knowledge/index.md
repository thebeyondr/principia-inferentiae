# Knowledge index

This wiki is a retrieval surface, not a second implementation. Equations explain the
methods; executable definitions live in `src/principia_inferentiae/`.

## Serving path

1. [Request lifecycle](request-lifecycle.md): boundaries from client arrival to final token.
2. [Prefill and decode](prefill-decode.md): distinct phase shapes and batching constraints.
3. [KV cache](kv-cache.md): MHA/GQA/MQA state growth, sharding, and capacity.
4. [Memory hierarchy](memory-hierarchy.md): capacity, bandwidth, locality, and transfers.

## Decisions

5. [Serving metrics](serving-metrics.md): TTFT, ITL, TPOT, E2E, per-user and aggregate TPS, goodput.
6. [Workload characterization](workload-characterization.md): token distributions, arrivals, bursts, concurrency, reuse, and tenants.
7. [Model and deployment sizing](model-deployment-sizing.md): quality gate, per-rank fit, service curves, topology, and reserves.
8. [Roofline analysis](roofline-analysis.md): measured scope, arithmetic intensity, empirical ceilings, and fail-closed classes.

## Evidence path

`sources/catalog.yaml → knowledge synthesis → typed inputs → principia CLI → claims → reports`

All performance conclusions need units, scope, provenance, assumptions, limitations, and
a falsifying experiment. Static estimates are capacity evidence, not benchmark evidence.

## Global retrieval checks

1. Which layer computes claims?
2. Why can a model fit in HBM yet miss an ITL SLO?
3. Which identity fields must a service curve match?
4. Why is GPU utilization insufficient for roofline classification?
5. What changes a ranked incident hypothesis into a confirmed root cause?

