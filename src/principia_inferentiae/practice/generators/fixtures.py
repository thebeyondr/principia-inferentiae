"""Shared builders. Generators pass knobs in; tests rebuild the same objects."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from principia_inferentiae.contracts import (
    ComputeCeiling,
    DiagnosisAnalysis,
    Digest,
    EvidenceKind,
    EvidenceRecord,
    HardwareSpec,
    HypothesisSpec,
    IncidentEvidence,
    IncidentSpec,
    InterventionResult,
    ModelSpec,
    QualityGate,
    QuantizationSpec,
    RooflineAnalysis,
    RooflineMeasurement,
    RunMetadata,
    RuntimeReserve,
    ServiceCurve,
    ServiceCurvePoint,
    ServingConfig,
    SizingAnalysis,
    SizingPolicy,
    SLOSpec,
    SourceProvenance,
    TraceRequest,
    WorkloadProfile,
)
from principia_inferentiae.diagnosis import diagnose_incident
from principia_inferentiae.memory import estimate_memory
from principia_inferentiae.roofline import analyze_roofline
from principia_inferentiae.sizing import serving_config_hash, size_deployment
from principia_inferentiae.workload import characterize_workload

FIXED = datetime(2026, 10, 16, tzinfo=timezone.utc)

Attention = Literal["MHA", "GQA", "MQA"]
WeightSharding = Literal["tensor_and_pipeline", "pipeline_only", "replicated"]
KvSharding = Literal["replicated", "shard_kv_heads", "shard_layers"]
EvidenceClass = Literal["authoritative", "measured", "synthetic", "declared"]
MetricScope = Literal["inference_only", "end_to_end"]


def digest(character: str = "a") -> Digest:
    return Digest(algorithm="sha256", value=character * 64)


def provenance(evidence_class: EvidenceClass = "synthetic", note: str = "Synthetic practice fixture.") -> SourceProvenance:
    return SourceProvenance(
        source_uri="fixture://principia/practice",
        revision="fixture-1",
        digest=digest("1"),
        evidence_class=evidence_class,
        retrieved_at_utc=FIXED,
        usage_note=note,
    )


def practice_run() -> RunMetadata:
    return RunMetadata(
        schema_version="1.0.0",
        run_id="practice-run",
        created_at_utc=FIXED,
        package_version="0.1.0",
        code_revision="practice-fixture",
        reproducibility_tier="R1",
        method_revision="practice-1",
        random_seed=0,
        input_digests=(digest("2"),),
    )


class Knobs:
    def __init__(self, raw: dict[str, object]) -> None:
        preset_name = raw.get("preset", "fit_small")
        if not isinstance(preset_name, str):
            raise TypeError("preset must be a string")
        if preset_name not in PRESETS:
            raise KeyError(preset_name)
        merged: dict[str, object] = dict(PRESETS[preset_name])
        for key, value in raw.items():
            if key != "preset":
                merged[key] = value
        self.raw = merged

    def integer(self, key: str, default: int) -> int:
        value = self.raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(key)
        return value

    def number(self, key: str, default: float) -> float:
        value = self.raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(key)
        return float(value)

    def text(self, key: str, default: str) -> str:
        value = self.raw.get(key, default)
        if not isinstance(value, str):
            raise TypeError(key)
        return value

    def optional_int(self, key: str) -> int | None:
        if key not in self.raw or self.raw[key] is None:
            return None
        value = self.raw[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(key)
        return value


def _attention(heads: int, kv_heads: int) -> Attention:
    if kv_heads == heads:
        return "MHA"
    if kv_heads == 1:
        return "MQA"
    return "GQA"


def memory_case(
    raw: dict[str, object] | None = None,
) -> tuple[RunMetadata, ModelSpec, HardwareSpec, ServingConfig]:
    knobs = Knobs({} if raw is None else raw)
    heads = knobs.integer("attention_heads", 8)
    kv_heads = knobs.integer("kv_heads", 2)
    group_size: int | None
    if "scale_bytes" in knobs.raw:
        group_size = knobs.integer("group_size", 64)
        metadata = knobs.number("scale_bytes", 8.0) / group_size
    else:
        group_size = knobs.optional_int("group_size")
        metadata = knobs.number("metadata", 0.0)
    bits = knobs.number("bits", 16.0)
    weight = QuantizationSpec(
        name="practice-weights",
        bits_per_weight=bits,
        metadata_bytes_per_parameter=metadata,
        group_size=group_size,
        metadata_semantics="Practice fixture. Metadata is scale bytes divided by group size when set.",
    )
    source = provenance()
    model = ModelSpec(
        model_id=knobs.text("model_id", "synthetic-practice"),
        revision="fixture-1",
        parameter_count=knobs.integer("parameter_count", 1_000_000),
        layer_count=knobs.integer("layer_count", 4),
        hidden_size=knobs.integer("hidden_size", 256),
        attention_heads=heads,
        kv_heads=kv_heads,
        head_dim=knobs.integer("head_dim", 64),
        attention_type=_attention(heads, kv_heads),
        weight_format=weight,
        kv_dtype_bytes=knobs.number("kv_dtype_bytes", 2.0),
        provenance=source,
        kv_projection_parameter_count=knobs.optional_int("kv_projection_parameter_count"),
    )
    tp = knobs.integer("tensor_parallel_size", 1)
    hardware = HardwareSpec(
        hardware_id="synthetic-80gb",
        revision="fixture-1",
        device_memory_bytes=knobs.integer("device_memory_bytes", 80 * 1024**3),
        empirical_hbm_bandwidth_gbps=knobs.number("bandwidth_gbps", 1_000.0),
        hbm_method_id="practice-ceiling",
        compute_ceilings=(
            ComputeCeiling(
                precision="bf16",
                empirical_tflops=knobs.number("empirical_tflops", 100.0),
                method_id="practice-ceiling",
                provenance=source,
            ),
        ),
        supported_tensor_parallel_sizes=(1, 2, 4, 8),
        devices_per_node=8,
        interconnect="synthetic full mesh",
        provenance=source,
    )
    sharding = knobs.text("weight_sharding", "tensor_and_pipeline")
    kv_sharding = knobs.text("kv_sharding", "shard_kv_heads")
    config = ServingConfig(
        config_id="practice-config",
        model_id=model.model_id,
        model_revision=model.revision,
        runtime_name="vllm",
        runtime_revision="0.11.0",
        precision="bf16",
        hardware_id=hardware.hardware_id,
        tensor_parallel_size=tp,
        pipeline_parallel_size=knobs.integer("pipeline_parallel_size", 1),
        max_concurrent_sequences=knobs.integer("max_concurrent_sequences", 4),
        max_sequence_tokens=knobs.integer("max_sequence_tokens", 4096),
        max_batched_tokens=knobs.integer("max_sequence_tokens", 4096),
        weight_sharding=_weight_sharding(sharding),
        kv_sharding=_kv_sharding(kv_sharding),
        sharding_imbalance_fraction=knobs.number("imbalance", 0.0),
        runtime_reserve=RuntimeReserve(
            runtime_fixed_bytes_per_rank=knobs.integer("reserve_bytes", 1_048_576),
            activation_bytes_per_rank=0,
            graph_bytes_per_rank=0,
            workspace_bytes_per_rank=0,
            fragmentation_fraction=knobs.number("fragmentation", 0.05),
            safety_margin_fraction=knobs.number("safety_margin", 0.10),
            provenance=source,
        ),
        material_runtime_flags=("--disable-log-requests",),
    )
    return practice_run(), model, hardware, config


def _weight_sharding(value: str) -> WeightSharding:
    if value == "tensor_and_pipeline":
        return "tensor_and_pipeline"
    if value == "pipeline_only":
        return "pipeline_only"
    if value == "replicated":
        return "replicated"
    raise ValueError(value)


def _kv_sharding(value: str) -> KvSharding:
    if value == "replicated":
        return "replicated"
    if value == "shard_kv_heads":
        return "shard_kv_heads"
    if value == "shard_layers":
        return "shard_layers"
    raise ValueError(value)


PRESETS: dict[str, dict[str, object]] = {
    "fit_small": {},
    "gqa": {"attention_heads": 32, "kv_heads": 8, "layer_count": 32, "head_dim": 128},
    "mha": {"attention_heads": 8, "kv_heads": 8, "layer_count": 4, "head_dim": 64},
    "mqa": {"attention_heads": 16, "kv_heads": 1, "layer_count": 4, "head_dim": 64},
    "tp4": {
        "attention_heads": 32,
        "kv_heads": 8,
        "layer_count": 32,
        "head_dim": 128,
        "tensor_parallel_size": 4,
        "weight_sharding": "tensor_and_pipeline",
        "kv_sharding": "shard_kv_heads",
    },
    "pp2": {
        "layer_count": 8,
        "pipeline_parallel_size": 2,
        "tensor_parallel_size": 1,
        "weight_sharding": "pipeline_only",
    },
    "replicated": {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "weight_sharding": "replicated",
    },
    "shard_layers": {
        "attention_heads": 32,
        "kv_heads": 8,
        "layer_count": 32,
        "head_dim": 128,
        "tensor_parallel_size": 4,
        "kv_sharding": "shard_layers",
        "weight_sharding": "tensor_and_pipeline",
    },
    "quant_group": {
        "parameter_count": 1_000_000,
        "bits": 4.0,
        "scale_bytes": 8.0,
        "group_size": 64,
    },
    "imbalance": {"tensor_parallel_size": 2, "imbalance": 0.1, "attention_heads": 8, "kv_heads": 2},
    "oom": {"parameter_count": 8_000_000_000, "bits": 16.0, "device_memory_bytes": 1_048_576},
    "lower_bound": {
        "parameter_count": 40_000_000_000,
        "bits": 16.0,
        "attention_heads": 32,
        "kv_heads": 8,
        "layer_count": 32,
        "head_dim": 128,
        "tensor_parallel_size": 4,
        "weight_sharding": "tensor_and_pipeline",
        "kv_sharding": "shard_kv_heads",
        "max_concurrent_sequences": 2,
        "max_sequence_tokens": 1024,
        "device_memory_bytes": 80 * 1024**3,
    },
}


def default_slo() -> SLOSpec:
    return SLOSpec(
        max_ttft_ms_p99=1_500,
        max_itl_ms_p99=80,
        max_e2e_ms_p99=10_000,
        min_goodput_fraction=0.95,
        metric_scope="end_to_end",
        queue_boundary="client send to final byte",
    )


def _required_int(item: dict[str, object], key: str) -> int:
    value = item[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(key)
    return value


def traces_from(raw: list[object]) -> tuple[TraceRequest, ...]:
    records: list[TraceRequest] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("trace request must be an object")
        prefix = item.get("prefix_key")
        shared = item.get("shared_prefix_tokens", 0)
        completed = item.get("completed_time_ms")
        if isinstance(shared, bool) or not isinstance(shared, int):
            raise TypeError("shared_prefix_tokens")
        if completed is not None and (isinstance(completed, bool) or not isinstance(completed, int)):
            raise TypeError("completed_time_ms")
        tenant = item.get("tenant_id", "a")
        if not isinstance(tenant, str):
            raise TypeError("tenant_id")
        records.append(
            TraceRequest(
                request_id=str(item["request_id"]),
                arrival_time_ms=_required_int(item, "arrival_time_ms"),
                completed_time_ms=completed,
                prompt_tokens=_required_int(item, "prompt_tokens"),
                output_tokens=_required_int(item, "output_tokens"),
                tenant_id=tenant,
                prefix_key=None if prefix is None else str(prefix),
                shared_prefix_tokens=shared,
            )
        )
    return tuple(records)


def default_trace() -> tuple[TraceRequest, ...]:
    return traces_from(
        [
            {
                "request_id": "r1",
                "arrival_time_ms": 0,
                "completed_time_ms": 400,
                "prompt_tokens": 128,
                "output_tokens": 32,
                "tenant_id": "a",
                "prefix_key": "sys",
                "shared_prefix_tokens": 64,
            },
            {
                "request_id": "r2",
                "arrival_time_ms": 50,
                "completed_time_ms": 500,
                "prompt_tokens": 256,
                "output_tokens": 64,
                "tenant_id": "a",
                "prefix_key": "sys",
                "shared_prefix_tokens": 64,
            },
            {
                "request_id": "r3",
                "arrival_time_ms": 1_200,
                "completed_time_ms": 1_800,
                "prompt_tokens": 512,
                "output_tokens": 16,
                "tenant_id": "b",
                "prefix_key": None,
                "shared_prefix_tokens": 0,
            },
        ]
    )


def workload_profile(
    requests: tuple[TraceRequest, ...],
    slo: SLOSpec | None = None,
) -> WorkloadProfile:
    return characterize_workload(practice_run(), requests, default_slo() if slo is None else slo).profile


def roofline_case(raw: dict[str, object]) -> tuple[RunMetadata, RooflineMeasurement, HardwareSpec]:
    knobs = Knobs({**{"preset": "fit_small"}, **raw})
    measured = provenance("measured", "Measured practice roofline fixture.")
    hardware = HardwareSpec(
        hardware_id="synthetic-80gb",
        revision="fixture-1",
        device_memory_bytes=knobs.integer("device_memory_bytes", 80 * 1024**3),
        empirical_hbm_bandwidth_gbps=knobs.number("bandwidth_gbps", 1_000.0),
        hbm_method_id="practice-ceiling",
        compute_ceilings=(
            ComputeCeiling(
                precision="bf16",
                empirical_tflops=knobs.number("empirical_tflops", 100.0),
                method_id="practice-ceiling",
                provenance=measured,
            ),
        ),
        supported_tensor_parallel_sizes=(1, 2, 4, 8),
        devices_per_node=8,
        interconnect="synthetic full mesh",
        provenance=measured,
    )
    status = knobs.text("measurement_status", "measured")
    measurement = RooflineMeasurement(
        measurement_id="practice-roofline",
        measurement_status="measured" if status == "measured" else "planned",
        scope_kind="kernel",
        scope_name=knobs.text("scope_name", "decode_gemm"),
        hardware_id=hardware.hardware_id,
        hardware_revision=hardware.revision,
        precision="bf16",
        flops=None if status != "measured" else knobs.number("flops", 2e11),
        hbm_bytes_moved=None if status != "measured" else knobs.number("hbm_bytes", 1e9),
        duration_ms=None if status != "measured" else knobs.number("duration_ms", 2.5),
        capacity_used_bytes=knobs.optional_int("capacity_used_bytes"),
        generic_gpu_utilization_fraction=(
            None
            if "utilization" not in knobs.raw
            else knobs.number("utilization", 0.0)
        ),
        raw_artifact_digest=None if status != "measured" else digest("e"),
        provenance=measured if status == "measured" else provenance(),
    )
    return practice_run(), measurement, hardware


def sizing_analysis(raw: dict[str, object]) -> SizingAnalysis:
    run, model, hardware, config = memory_case(raw)
    profile = characterize_workload(run, default_trace(), default_slo()).profile
    memory = estimate_memory(run, model, hardware, config).estimate
    gate = QualityGate(
        evaluation_id="practice-quality",
        model_id=config.model_id,
        model_revision=config.model_revision,
        precision=config.precision,
        passed=True,
        metric_summary="Synthetic practice quality gate.",
        artifact_digest=digest("b"),
    )
    policy = SizingPolicy(
        target_headroom_fraction=0.2,
        failure_reserve_replicas=1,
        rollout_reserve_replicas=1,
    )
    mode = raw.get("curve", "no_curve")
    if not isinstance(mode, str):
        raise TypeError("curve")
    curve: ServiceCurve | None
    if mode == "no_curve":
        curve = None
    else:
        measured = provenance("measured", "Measured practice service curve.")
        workload_hash = profile.workload_hash if mode != "wrong_workload" else "0" * 64
        config_hash = serving_config_hash(config) if mode != "wrong_config" else "0" * 64
        scope: MetricScope = "inference_only" if mode == "wrong_scope" else "end_to_end"
        meets = mode != "misses_slo"
        curve = ServiceCurve(
            curve_id="practice-curve",
            workload_hash=workload_hash,
            config_hash=config_hash,
            measured=True,
            metric_scope=scope,
            points=(
                ServiceCurvePoint(
                    offered_rps=4,
                    concurrency=2,
                    achieved_goodput_rps=4 if meets else 0,
                    ttft_ms_p99=1_000 if meets else 2_000,
                    itl_ms_p99=50 if meets else 200,
                    tpot_ms_p99=50 if meets else 200,
                    e2e_ms_p99=8_000 if meets else 20_000,
                    per_user_tps=20 if meets else 5,
                    aggregate_tps=200 if meets else 20,
                    goodput_fraction=0.99 if meets else 0,
                    trial_count=3,
                    dispersion_method="IQR",
                ),
            ),
            raw_artifact_digest=digest("f"),
            benchmark_method_id="practice-benchmark",
            provenance=measured,
        )
    return size_deployment(run, profile, memory, model, hardware, config, gate, policy, curve)


def _link(
    evidence_id: str,
    kind_name: str,
    description: str,
    timestamp_ms: int,
    value: str,
    hypothesis_id: str,
    stance: Literal["supporting", "contrary"] = "supporting",
) -> IncidentEvidence:
    digest_char = evidence_id[0] if evidence_id[:1] in {"a", "b", "c", "d", "e", "f"} else "a"
    return IncidentEvidence(
        evidence=EvidenceRecord(
            evidence_id=evidence_id,
            kind=EvidenceKind(kind_name),
            description=description,
            observed_value=value,
            unit="count",
            source_digest=digest(digest_char),
            timestamp_ms=timestamp_ms,
        ),
        hypothesis_id=hypothesis_id,
        stance=stance,
    )


def rank_incident() -> IncidentSpec:
    direct = "h-direct"
    metric = "h-metric"
    corr = "h-corr"
    hypotheses = (
        HypothesisSpec(
            hypothesis_id=direct,
            category="queue_scheduler",
            statement="Queue delay.",
            expected_observation="Queue depth leads TTFT.",
            expected_metric_direction="decrease",
        ),
        HypothesisSpec(
            hypothesis_id=metric,
            category="prefill",
            statement="Prefill cost.",
            expected_observation="Prefill ms rose.",
            expected_metric_direction="decrease",
        ),
        HypothesisSpec(
            hypothesis_id=corr,
            category="host_tokenization",
            statement="Tokenizer correlation.",
            expected_observation="Host time correlates.",
            expected_metric_direction="decrease",
        ),
    )
    return IncidentSpec(
        incident_id="practice-rank",
        baseline_run_id="baseline-rank",
        hypotheses=hypotheses,
        evidence=(
            _link("d-direct", "direct_event", "queue_depth", 10, "1", direct),
            _link("m-metric", "measured_metric", "prefill_ms", 20, "40", metric),
            _link("c-corr", "correlation", "host_cpu", 30, "0.4", corr),
        ),
        interventions=(),
        required_telemetry=("queue_depth", "prefill_ms", "host_cpu"),
    )


def causal_incident() -> IncidentSpec:
    passed = "h-pass"
    missed = "h-miss"
    hypotheses = (
        HypothesisSpec(
            hypothesis_id=passed,
            category="prefill",
            statement="Shorter prompts reduce TTFT.",
            expected_observation="TTFT decreases after the prompt cap.",
            expected_metric_direction="decrease",
        ),
        HypothesisSpec(
            hypothesis_id=missed,
            category="decode",
            statement="Decode correlation only.",
            expected_observation="ITL moved with batch.",
            expected_metric_direction="decrease",
        ),
    )
    return IncidentSpec(
        incident_id="practice-causal",
        baseline_run_id="baseline-causal",
        hypotheses=hypotheses,
        evidence=(
            _link("p-base", "measured_metric", "ttft_ms", 100, "40", passed),
            _link("p-rerun", "intervention", "ttft_ms_rerun", 200, "10", passed),
            _link("m-corr", "correlation", "itl_ms", 100, "0.8", missed),
        ),
        interventions=(
            InterventionResult(
                intervention_id="i-pass",
                hypothesis_id=passed,
                explicit_hypothesis="Shorter prompts reduce TTFT.",
                changed_variable="max_prompt_tokens: 8192 -> 1024",
                baseline_run_id="baseline-causal",
                rerun_id="rerun-causal",
                baseline_workload_hash="a" * 64,
                rerun_workload_hash="a" * 64,
                baseline_config_hash="b" * 64,
                rerun_config_hash="c" * 64,
                metric_name="ttft_ms",
                dispersion=1.0,
                unit="ms",
                baseline_evidence_id="p-base",
                rerun_evidence_id="p-rerun",
            ),
        ),
        required_telemetry=("ttft_ms", "ttft_ms_rerun", "itl_ms"),
    )


def correlation_incident() -> IncidentSpec:
    hypothesis = HypothesisSpec(
        hypothesis_id="h-queue",
        category="queue_scheduler",
        statement="Queueing tracks TTFT.",
        expected_observation="Queue depth rises before TTFT.",
        expected_metric_direction="increase",
    )
    return IncidentSpec(
        incident_id="practice-correlation",
        baseline_run_id="baseline-correlation",
        hypotheses=(hypothesis,),
        evidence=(_link("q-corr", "correlation", "queue_depth", 50, "0.9", hypothesis.hypothesis_id),),
        interventions=(),
        required_telemetry=("queue_depth",),
    )


def missing_incident() -> IncidentSpec:
    hypothesis = HypothesisSpec(
        hypothesis_id="h-kv",
        category="kv_capacity",
        statement="KV pressure.",
        expected_observation="KV occupancy rises.",
        expected_metric_direction="decrease",
    )
    return IncidentSpec(
        incident_id="practice-missing",
        baseline_run_id="baseline-missing",
        hypotheses=(hypothesis,),
        evidence=(_link("k-metric", "measured_metric", "request latency", 10, "80", hypothesis.hypothesis_id),),
        interventions=(),
        required_telemetry=("request latency", "kv cache occupancy"),
    )


def diagnosis_of(incident: IncidentSpec) -> DiagnosisAnalysis:
    return diagnose_incident(practice_run(), incident)


def roofline_of(raw: dict[str, object]) -> RooflineAnalysis:
    run, measurement, hardware = roofline_case(raw)
    return analyze_roofline(run, measurement, hardware)
