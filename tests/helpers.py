"""Shared practice-test helpers. Import as ``helpers`` (tests/ on pytest pythonpath)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from principia_inferentiae.io import load_document
from principia_inferentiae.practice.dag import ConceptDAG

REPO = Path(__file__).resolve().parents[1]
CONCEPTS = REPO / "data" / "concepts"
NOW = datetime(2026, 6, 1, tzinfo=UTC)

EXPECTED_IDS = {
    "units.bytes_prefixes",
    "units.us_ms_s",
    "units.tflops",
    "units.gbps",
    "units.tokens_per_s_inversion",
    "model.params_bytes",
    "model.quant_metadata",
    "model.group_size_overhead",
    "model.weight_shards_tp",
    "model.per_rank_weight_bytes",
    "kv.bytes_per_token",
    "kv.mha_gqa_mqa",
    "kv.tp_pp_split",
    "kv.shard_layers_replicate",
    "kv.active_tokens",
    "kv.per_rank_fit",
    "kv.rank_oom",
    "phases.prefill_ms",
    "phases.decode_gap",
    "phases.ttft_queue_prefill",
    "phases.itl_vs_tpot",
    "phases.chunked_prefill",
    "phases.disaggregation",
    "metrics.boundaries",
    "metrics.per_user_vs_aggregate",
    "metrics.nearest_rank_percentile",
    "metrics.p50_vs_p99",
    "metrics.goodput",
    "metrics.inference_vs_e2e",
    "metrics.littles_law",
    "workload.prompt_vs_output",
    "workload.peak_concurrency",
    "workload.burst_vs_steady",
    "workload.prefix_reuse",
    "workload.tenant_mix",
    "roofline.arithmetic_intensity",
    "roofline.ridge_point",
    "roofline.classify",
    "roofline.capacity_vs_bandwidth",
    "roofline.utilization_not_saturation",
    "sizing.service_curve_identity",
    "sizing.memory_fit_lower_bound",
    "sizing.null_gpu_count",
    "sizing.benchmark_plan",
    "diagnosis.hypothesis_rank",
    "diagnosis.causal_gate",
    "diagnosis.correlation_vs_cause",
    "diagnosis.missing_telemetry",
    "policy.admission",
    "policy.queue",
    "policy.chunked_prefill",
    "policy.retry_storm",
    "policy.goodput",
    "policy.prefix",
    "policy.priority",
    "policy.sticky",
    "policy.hog",
    "policy.frag",
    "policy.spec",
    "policy.evict",
    "policy.lora",
    "policy.scale",
    "policy.batch",
}


def concept_documents() -> list[object]:
    documents: list[object] = []
    for path in sorted(CONCEPTS.glob("*.yaml")):
        loaded = load_document(path)
        assert isinstance(loaded, list)
        documents.extend(loaded)
    return documents


def load_concept_dag() -> ConceptDAG:
    return ConceptDAG.from_documents(concept_documents())
