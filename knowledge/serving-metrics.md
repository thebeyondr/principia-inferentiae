# Serving metrics

## Mental model

Latency, token cadence, throughput, and SLO attainment answer different questions. Keep
their scopes and populations explicit.

## Theory and units

- **TTFT [ms/request]:** request boundary to first token.
- **ITL [ms/token-gap]:** distribution of gaps between emitted tokens.
- **TPOT [ms/output-token]:** generation duration normalized by generated tokens under a declared convention.
- **E2E [ms/request]:** request boundary to completion.
- **Per-user TPS [token/s/user]:** perceived generation rate for one active request.
- **Aggregate TPS [token/s/system]:** total token production across requests.
- **Goodput [good requests/s]:** completed requests satisfying every declared SLO per unit time.

`per_user_tps ≈ 1000 / TPOT_ms` only under a compatible steady-generation convention.
It does not replace measured ITL tails.

## What must be measured

- p50/p90/p95/p99 with sample count and aggregation method;
- offered load, achieved load, arrival process, concurrency, warmup, and failures;
- inference-only or end-to-end boundary, including queue/network scope;
- prompt/output lengths and good/bad request classification.

Equations define metrics. A service curve measures them under a fixed workload/config
identity. A mean latency cannot certify a p99 SLO.

## Failure modes

- reporting total TPS as user-visible speed;
- dropping failed requests from latency and goodput;
- mixing ITL samples across requests without declaring weighting;
- comparing open-loop and closed-loop loads as equivalent;
- calling requests/s throughput “goodput” without SLO classification.

## Discriminating experiments

Sweep offered load through and beyond the knee. At each point record all latency tails,
per-user/aggregate TPS, failures, and goodput fraction across repeated trials.

## Retrieval checks

1. Which metric captures first-token responsiveness?
2. Can aggregate TPS improve while goodput degrades?
3. What boundary must accompany every E2E number?

## Sources

- [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/)
- [vLLM bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/)
- [DistServe paper](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)

