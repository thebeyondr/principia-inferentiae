---
name: diagnose-serving-regression
description: Rank timestamp-aligned serving-regression hypotheses and permit root-cause language only after a controlled baseline, intervention, equivalent rerun, and measured delta. Use when diagnosing an inference latency, throughput, or memory regression.
license: Apache-2.0
compatibility: Requires Python 3.11+ and the principia CLI.
---

# Diagnose serving regression

Prepare an `IncidentSpec` covering queue/scheduler, prefill, decode, KV capacity, HBM bandwidth, compute, interconnect, host/tokenization, and network as applicable.

Run:

`principia diagnose-regression --run <run.yaml> --incident <incident.yaml> --output <diagnosis.json>`

Interpretation:

- correlation-only evidence may rank a hypothesis but never confirms root cause;
- supporting and contrary evidence stays `inconclusive`;
- confirmation requires baseline → explicit hypothesis → one changed variable → equivalent rerun → measured delta;
- an OOM mechanism additionally requires a direct OOM event and preceding causal timeline evidence.

Execute the emitted next discriminating experiment when the gate does not pass.

