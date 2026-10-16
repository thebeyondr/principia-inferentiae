# Principia Inferentiae

An executable, evidence-first knowledge base for LLM inference engineering.

Markdown explains. Versioned YAML/JSON records facts. Only
`src/principia_inferentiae/` computes conclusions. Portable Agent Skills in
`.agents/skills/` call tested CLI commands and contain no hidden formulas.

## What it answers

- What workload was actually observed, including separate prompt and generated-token distributions?
- Does the declared model/runtime configuration fit per rank, including quantization metadata, KV state, runtime reserve, fragmentation, and safety margin?
- Is a measured kernel or phase compute-bound, HBM-bandwidth-bound, underfilled, mixed, or unsupported by evidence?
- Does a workload- and config-matched measured service curve justify a deployment count?
- Which incident hypotheses rank highest, and has the causal gate actually passed?
- How can the same validated claims be rendered for engineers and stakeholders?

Without a matched measured service curve, sizing deliberately emits a memory-fit lower
bound with `recommended_gpu_count: null` and `status: insufficient_evidence`.

## Reproducible quick start

Requires Python 3.11+.

The committed `uv.lock` is the fully resolved path:

```bash
uv sync --extra dev --frozen
uv run make verify
```

The equivalent standard-library virtual-environment path is:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src tests
pytest
```

Run the complete synthetic long-context case:

```bash
principia prepare-vllm-trace \
  --trace case-studies/long-context-bursty/inputs/trace.jsonl \
  --chunk-hash-size 512 \
  --output case-studies/long-context-bursty/outputs/vllm-timed-trace.jsonl

principia analyze \
  --run case-studies/long-context-bursty/inputs/run.yaml \
  --trace case-studies/long-context-bursty/inputs/trace.jsonl \
  --slo case-studies/long-context-bursty/inputs/slo.yaml \
  --model case-studies/long-context-bursty/inputs/model.yaml \
  --hardware case-studies/long-context-bursty/inputs/hardware.yaml \
  --config case-studies/long-context-bursty/inputs/serving-config.yaml \
  --measurement case-studies/long-context-bursty/inputs/roofline-measurement.yaml \
  --quality-gate case-studies/long-context-bursty/inputs/quality-gate.yaml \
  --policy case-studies/long-context-bursty/inputs/sizing-policy.yaml \
  --output case-studies/long-context-bursty/outputs/analysis.json

principia write-brief \
  --analysis case-studies/long-context-bursty/outputs/analysis.json \
  --audience engineering \
  --output case-studies/long-context-bursty/report.md

principia write-brief \
  --analysis case-studies/long-context-bursty/outputs/analysis.json \
  --audience stakeholder \
  --output case-studies/long-context-bursty/outputs/stakeholder-brief.md
```

The case says **20K input tokens and 512 output tokens**. It does not relabel a
20K-generation workload as a 20K-prompt workload. Its hardware and model facts are
synthetic teaching fixtures, and it contains no fabricated benchmark result. The
expected production recommendation is therefore `null`; the emitted benchmark plan is
the next evidence-producing step.

Export the current JSON Schemas:

```bash
principia export-schemas --output-directory schemas
```

## Evidence contract

Every conclusion is an immutable validated `Claim` with:

- `claim_type`, `evidence_ids`, and `method_id`;
- explicit units, assumptions, limitations, and confidence;
- supporting and contrary evidence IDs;
- missing telemetry and the next discriminating experiment.

Conflicting evidence must be `inconclusive`. Correlation-only telemetry may rank a
hypothesis but cannot confirm root cause. Causal language requires:

`baseline → explicit hypothesis → one intervention → equivalent rerun → measured delta`

Run metadata, timestamps, revisions, provenance, and digests are inputs. The tools never
read wall-clock time, Git state, machine state, or the network.

## Repository map

- `knowledge/`: original concise synthesis, retrieval checks, and citations.
- `sources/catalog.yaml`: source locators and explicit digest semantics.
- `data/catalogs/`: versioned synthetic model, hardware, and runtime facts.
- `src/principia_inferentiae/`: sole computational authority.
- `schemas/`: generated interface schemas.
- `.agents/skills/`: thin open-standard CLI orchestration.
- `tests/`: unit, invariant, acceptance, traceability, deterministic-output, and skill-contract tests.
- `case-studies/`: learner prediction, typed inputs, generated outputs, and retrospective.

## Non-goals

This slice does not choose a model on quality, predict unmeasured throughput, infer a
root cause from utilization, implement an autoscaler, copy third-party articles, or
claim that a synthetic fixture represents a vendor benchmark.

## Sources

Start at [`sources/catalog.yaml`](sources/catalog.yaml) and
[`knowledge/index.md`](knowledge/index.md). Core references include the Roofline paper,
CUDA Programming Guide, JAX Scaling Book, vLLM internals/metrics/benchmark docs,
DistServe’s SLO-goodput framing, and the Agent Skills specification.

