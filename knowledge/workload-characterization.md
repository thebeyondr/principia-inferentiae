# Workload characterization

## Mental model

A workload is a distribution over request shapes and time, not a single context length.
Prompt tokens and generated tokens drive different work and must remain separate.

## Theory

Characterize at minimum:

- prompt/output/total token p50, p95, and p99 [token/request];
- arrival rate [request/s], concurrency [request], and explicit burst window;
- prefix reuse [fraction], tenant mix, and SLOs;
- observation window and arrival process.

Principia uses nearest-rank percentiles and one-second buckets aligned to the first
arrival. The workload hash covers canonicalized requests plus the SLO.

## What must be measured

Request arrival/completion timestamps, prompt/generated token counts, tenant identifier,
prefix key where permitted, failures, and client-observed SLO metrics. Use a trace window
long enough to expose bursts and tails.

The profile describes observed input. It does not forecast traffic or prove that a
synthetic random-token dataset reproduces semantic prefix/cache behavior.

## Failure modes

- saying “20K tokens” without input/output direction;
- substituting mean length for a tail-capacity decision;
- computing concurrency without completion timestamps;
- reordering or filtering failures without recording policy;
- matching a benchmark by averages while hashes differ.

## Discriminating experiments

Replay the canonical trace, then independently perturb prompt length, output length,
arrival burstiness, and prefix reuse. A changed workload receives a new identity and
service curve.

## Retrieval checks

1. Why do 20K input and 20K output imply different service behavior?
2. Which missing field prevents observed concurrency measurement?
3. Why does a matching mean not satisfy workload identity?

## Sources

- [vLLM bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/)
- [DistServe paper](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)
- [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/)

