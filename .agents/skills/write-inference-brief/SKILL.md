---
name: write-inference-brief
description: Render deterministic engineering or stakeholder Markdown from validated analysis JSON without inventing findings, suppressing nulls, or hiding uncertainty. Use when writing an inference report or stakeholder brief.
license: Apache-2.0
compatibility: Requires Python 3.11+ and the principia CLI.
---

# Write inference brief

Run one of:

`principia write-brief --analysis <analysis.json> --audience engineering --output <report.md>`

`principia write-brief --analysis <analysis.json> --audience stakeholder --output <brief.md>`

The JSON analysis remains authoritative. Preserve claim status, confidence, evidence IDs, assumptions, limitations, missing telemetry, and next experiment. Do not:

- turn a memory-fit lower bound into a deployment recommendation;
- translate correlation into causality;
- hide contrary evidence or `null`;
- merge TTFT, ITL, TPOT, E2E, per-user TPS, aggregate TPS, or goodput.

