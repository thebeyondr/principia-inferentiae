---
name: size-model-deployment
description: Compute model, quantization metadata, KV, reserve, and per-rank memory fit, then size replicas only from a workload- and config-matched measured SLO-goodput curve. Use when estimating GPU count or model deployment size.
license: Apache-2.0
compatibility: Requires Python 3.11+ and the principia CLI.
---

# Size model deployment

First compute capacity:

`principia estimate-memory --run <run.yaml> --model <model.yaml> --hardware <hardware.yaml> --config <config.yaml> --output <memory.json>`

Then size:

`principia size-deployment --run <run.yaml> --workload <workload.json> --memory <memory.json> --model <model.yaml> --hardware <hardware.yaml> --config <config.yaml> --quality-gate <quality.json> --policy <policy.yaml> [--service-curve <curve.yaml>] --output <sizing.json>`

Rules:

- model/precision quality is an independent input gate;
- use maximum-rank fit, topology, active replicas, failure reserve, and rollout reserve;
- a curve must match workload hash, config hash, and metric scope;
- without that curve, report the memory lower bound with `recommended_gpu_count: null`;
- execute the emitted benchmark plan rather than extrapolating from capacity.

