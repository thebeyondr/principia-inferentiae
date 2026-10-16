---
name: analyze-memory-vs-compute
description: Classify one measured kernel or phase against empirical precision-specific compute and HBM ceilings while keeping HBM capacity separate. Use when deciding whether a kernel or phase is memory-bandwidth-bound or compute-bound.
license: Apache-2.0
compatibility: Requires Python 3.11+ and the principia CLI.
---

# Analyze memory versus compute

Run:

`principia analyze-roofline --run <run.yaml> --measurement <measurement.yaml> --hardware <hardware.yaml> --output <roofline.json>`

Accept only a named kernel or phase with measured FLOPs, HBM bytes, duration, precision, and raw-artifact digest. Interpret these outcomes exactly:

- `compute_bound`
- `hbm_bandwidth_bound`
- `underfilled`
- `mixed`
- `insufficient_evidence`

Capacity pressure is reported separately. Generic GPU utilization never proves compute saturation. If measurement scope or compatible empirical ceilings are absent, preserve the fail-closed result and request the experiment named in the claim.

