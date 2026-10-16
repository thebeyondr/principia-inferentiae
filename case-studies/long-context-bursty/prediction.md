# Pre-execution prediction

**Owner:** learner  
**Status:** learner entry required before the real GPU run  
**Locked benchmark-plan revision:** `1.0.0`

## Workload distinction

This case has prompt p99 of **20,000 input tokens** and at most **512 generated output
tokens**. It is not a 20,000-output-token workload. A 20K-output case would require a new
trace/workload hash, a much longer decode horizon, different active-KV residency, and a
new service curve.

## Learner predictions

Fill these before executing the GPU plan; do not edit after seeing results.

- Expected TTFT p99 at each load point: _pending_
- Expected ITL and TPOT p99 at each load point: _pending_
- Expected goodput knee [requests/s]: _pending_
- Expected maximum-rank HBM [bytes]: _pending_
- Expected prefill roofline class and why: _pending_
- Expected decode roofline class and why: _pending_
- First failure mode as load rises: _pending_
- One observation that would falsify the bottleneck prediction: _pending_

## Static-analysis prediction

The deterministic tool should separate prompt/output distributions, compute a per-rank
capacity estimate, refuse a roofline class without profile counters, and return a null
GPU recommendation without a matching measured service curve. Those are contract
predictions, not runtime-performance predictions.

