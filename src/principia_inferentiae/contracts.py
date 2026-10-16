"""Immutable, validated contracts shared by every analysis path."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    """Strict base model; tuples keep collection fields immutable as well."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AnalysisStatus(StrEnum):
    SUPPORTED = "supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INCONCLUSIVE = "inconclusive"
    MEMORY_NOT_FIT = "memory_not_fit"
    SLO_NOT_MET = "slo_not_met"
    QUALITY_GATE_FAILED = "quality_gate_failed"


class Confidence(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BoundClassification(StrEnum):
    COMPUTE_BOUND = "compute_bound"
    HBM_BANDWIDTH_BOUND = "hbm_bandwidth_bound"
    UNDERFILLED = "underfilled"
    MIXED = "mixed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvidenceKind(StrEnum):
    INPUT = "input"
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    DIRECT_EVENT = "direct_event"
    MEASURED_METRIC = "measured_metric"
    CORRELATION = "correlation"
    CONFIGURATION = "configuration"
    INTERVENTION = "intervention"
    ARTIFACT = "artifact"
    ESTIMATE = "estimate"


class Digest(FrozenModel):
    algorithm: Literal["sha256"]
    value: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceProvenance(FrozenModel):
    source_uri: str
    revision: str
    digest: Digest
    evidence_class: Literal["authoritative", "measured", "synthetic", "declared"]
    retrieved_at_utc: datetime
    usage_note: str

    @model_validator(mode="after")
    def require_timezone(self) -> Self:
        if self.retrieved_at_utc.tzinfo is None:
            raise ValueError("retrieved_at_utc must include a UTC offset")
        return self


class RunMetadata(FrozenModel):
    schema_version: Literal["1.0.0"]
    run_id: str = Field(min_length=1)
    created_at_utc: datetime
    package_version: str = Field(min_length=1)
    code_revision: str = Field(min_length=1)
    reproducibility_tier: Literal["R0", "R1", "R2", "R3"]
    method_revision: str = Field(min_length=1)
    random_seed: int
    input_digests: tuple[Digest, ...]
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_timezone(self) -> Self:
        if self.created_at_utc.tzinfo is None:
            raise ValueError("created_at_utc must include a UTC offset")
        return self


class UnitRecord(FrozenModel):
    field: str
    unit: str


class EvidenceRecord(FrozenModel):
    evidence_id: str = Field(min_length=1)
    kind: EvidenceKind
    description: str = Field(min_length=1)
    observed_value: str
    unit: str
    source_digest: Digest
    timestamp_ms: int | None = Field(default=None, ge=0)
    event_type: Literal["oom", "preemption"] | None = None


class Claim(FrozenModel):
    claim_id: str = Field(min_length=1)
    claim_type: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    status: AnalysisStatus
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    supporting_evidence_ids: tuple[str, ...] = ()
    contrary_evidence_ids: tuple[str, ...] = ()
    method_id: str = Field(min_length=1)
    units: tuple[UnitRecord, ...]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    confidence: Confidence
    missing_telemetry: tuple[str, ...]
    next_discriminating_experiment: str | None

    @model_validator(mode="after")
    def enforce_evidence_logic(self) -> Self:
        evidence = set(self.evidence_ids)
        if not set(self.supporting_evidence_ids).issubset(evidence):
            raise ValueError("supporting evidence must appear in evidence_ids")
        if not set(self.contrary_evidence_ids).issubset(evidence):
            raise ValueError("contrary evidence must appear in evidence_ids")
        if (
            self.supporting_evidence_ids
            and self.contrary_evidence_ids
            and self.status is not AnalysisStatus.INCONCLUSIVE
        ):
            raise ValueError("conflicting evidence requires inconclusive status")
        if self.status is AnalysisStatus.SUPPORTED and not self.supporting_evidence_ids:
            raise ValueError("supported claims require supporting evidence")
        if self.status is AnalysisStatus.INSUFFICIENT_EVIDENCE and not (
            self.missing_telemetry or self.next_discriminating_experiment
        ):
            raise ValueError("insufficient-evidence claims must name the evidence gap")
        return self


class EvidenceEnvelope(FrozenModel):
    run: RunMetadata
    evidence: tuple[EvidenceRecord, ...]
    claims: tuple[Claim, ...]

    @model_validator(mode="after")
    def validate_traceability(self) -> Self:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        claim_ids = [item.claim_id for item in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim IDs must be unique")
        known_evidence = set(evidence_ids)
        for claim in self.claims:
            unknown = set(claim.evidence_ids) - known_evidence
            if unknown:
                raise ValueError(f"claim {claim.claim_id} references unknown evidence: {unknown}")
        return self


class QuantizationSpec(FrozenModel):
    name: str
    bits_per_weight: float = Field(gt=0, le=32)
    metadata_bytes_per_parameter: float = Field(ge=0)
    group_size: int | None = Field(default=None, gt=0)
    metadata_semantics: str

    @model_validator(mode="after")
    def require_group_for_metadata(self) -> Self:
        if self.metadata_bytes_per_parameter > 0 and self.group_size is None:
            raise ValueError("quantization metadata requires an explicit group_size")
        return self


class ModelSpec(FrozenModel):
    model_id: str
    revision: str
    parameter_count: int = Field(gt=0)
    layer_count: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    attention_heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    attention_type: Literal["MHA", "GQA", "MQA"]
    weight_format: QuantizationSpec
    kv_dtype_bytes: float = Field(gt=0, le=8)
    provenance: SourceProvenance

    kv_projection_parameter_count: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_attention_geometry(self) -> Self:
        if (
            self.kv_projection_parameter_count is not None
            and self.kv_projection_parameter_count > self.parameter_count
        ):
            raise ValueError("KV projection parameter count cannot exceed parameter_count")
        if self.attention_type == "MHA" and self.kv_heads != self.attention_heads:
            raise ValueError("MHA requires kv_heads == attention_heads")
        if self.attention_type == "MQA" and self.kv_heads != 1:
            raise ValueError("MQA requires kv_heads == 1")
        if self.attention_type == "GQA":
            if not 1 < self.kv_heads < self.attention_heads:
                raise ValueError("GQA requires 1 < kv_heads < attention_heads")
            if self.attention_heads % self.kv_heads:
                raise ValueError("GQA requires attention_heads divisible by kv_heads")
        return self


class ComputeCeiling(FrozenModel):
    precision: str
    empirical_tflops: float = Field(gt=0)
    method_id: str
    provenance: SourceProvenance


class HardwareSpec(FrozenModel):
    hardware_id: str
    revision: str
    device_memory_bytes: int = Field(gt=0)
    empirical_hbm_bandwidth_gbps: float = Field(gt=0)
    hbm_method_id: str
    compute_ceilings: tuple[ComputeCeiling, ...]
    supported_tensor_parallel_sizes: tuple[int, ...]
    devices_per_node: int = Field(gt=0)
    interconnect: str
    provenance: SourceProvenance

    @model_validator(mode="after")
    def validate_unique_precisions(self) -> Self:
        precisions = [item.precision for item in self.compute_ceilings]
        if len(precisions) != len(set(precisions)):
            raise ValueError("compute ceiling precision entries must be unique")
        if any(size <= 0 for size in self.supported_tensor_parallel_sizes):
            raise ValueError("supported tensor-parallel sizes must be positive")
        return self


class ModelCatalog(FrozenModel):
    schema_version: Literal["1.0.0"]
    catalog_revision: str
    catalog_digest_semantics: str
    entries: tuple[ModelSpec, ...]


class HardwareCatalog(FrozenModel):
    schema_version: Literal["1.0.0"]
    catalog_revision: str
    catalog_digest_semantics: str
    entries: tuple[HardwareSpec, ...]


class RuntimeSource(FrozenModel):
    url: str
    revision: str
    locator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeCatalogEntry(FrozenModel):
    runtime_id: str
    revision: str
    artifact_digest: Digest | None
    artifact_digest_semantics: str
    source: RuntimeSource
    supported_evidence_use: str
    required_run_fields: tuple[str, ...]


class RuntimeCatalog(FrozenModel):
    schema_version: Literal["1.0.0"]
    catalog_revision: str
    catalog_digest_semantics: str
    entries: tuple[RuntimeCatalogEntry, ...]


class SLOSpec(FrozenModel):
    max_ttft_ms_p99: float | None = Field(default=None, gt=0)
    max_itl_ms_p99: float | None = Field(default=None, gt=0)
    max_tpot_ms_p99: float | None = Field(default=None, gt=0)
    max_e2e_ms_p99: float | None = Field(default=None, gt=0)
    min_per_user_tps: float | None = Field(default=None, gt=0)
    min_aggregate_tps: float | None = Field(default=None, gt=0)
    min_goodput_fraction: float = Field(gt=0, le=1)
    metric_scope: Literal["inference_only", "end_to_end"]
    queue_boundary: str

    @model_validator(mode="after")
    def reject_impossible_latency_budget(self) -> Self:
        if (
            self.max_e2e_ms_p99 is not None
            and self.max_ttft_ms_p99 is not None
            and self.max_e2e_ms_p99 <= self.max_ttft_ms_p99
        ):
            raise ValueError("E2E p99 must exceed TTFT p99")
        return self


class TraceRequest(FrozenModel):
    request_id: str
    arrival_time_ms: int = Field(ge=0)
    completed_time_ms: int | None = Field(default=None, ge=0)
    prompt_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    tenant_id: str
    prefix_key: str | None = None
    shared_prefix_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        if self.completed_time_ms is not None and self.completed_time_ms <= self.arrival_time_ms:
            raise ValueError("completed_time_ms must follow arrival_time_ms")
        if self.shared_prefix_tokens > self.prompt_tokens:
            raise ValueError("shared_prefix_tokens cannot exceed prompt_tokens")
        if self.prefix_key is None and self.shared_prefix_tokens:
            raise ValueError("shared_prefix_tokens requires prefix_key")
        return self


class VllmTimedTraceRecord(FrozenModel):
    timestamp: int = Field(ge=0)
    input_length: int = Field(gt=0)
    output_length: int = Field(gt=0)
    hash_ids: tuple[int, ...] = Field(min_length=1)


class Percentiles(FrozenModel):
    p50: float = Field(ge=0)
    p95: float = Field(ge=0)
    p99: float = Field(ge=0)


class WorkloadProfile(FrozenModel):
    workload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_digest: Digest
    request_count: int = Field(gt=0)
    observation_window_ms: int = Field(gt=0)
    prompt_tokens: Percentiles
    output_tokens: Percentiles
    total_tokens: Percentiles
    mean_arrival_rps: float = Field(ge=0)
    peak_one_second_arrival_rps: float = Field(ge=0)
    burstiness_peak_to_mean: float = Field(ge=1)
    peak_observed_concurrency: int | None = Field(default=None, gt=0)
    prefix_reuse_fraction: float = Field(ge=0, le=1)
    shared_prefix_tokens: Percentiles
    tenant_count: int = Field(gt=0)
    slo: SLOSpec
    missing_telemetry: tuple[str, ...]


class RuntimeReserve(FrozenModel):
    runtime_fixed_bytes_per_rank: int = Field(ge=0)
    activation_bytes_per_rank: int = Field(ge=0)
    graph_bytes_per_rank: int = Field(ge=0)
    workspace_bytes_per_rank: int = Field(ge=0)
    fragmentation_fraction: float = Field(ge=0, lt=1)
    safety_margin_fraction: float = Field(ge=0, lt=1)
    provenance: SourceProvenance


class ServingConfig(FrozenModel):
    config_id: str
    model_id: str
    model_revision: str
    runtime_name: str
    runtime_revision: str
    precision: str
    hardware_id: str
    tensor_parallel_size: int = Field(gt=0)
    pipeline_parallel_size: int = Field(gt=0)
    max_concurrent_sequences: int = Field(gt=0)
    max_sequence_tokens: int = Field(gt=0)
    max_batched_tokens: int = Field(gt=0)
    weight_sharding: Literal["tensor_and_pipeline", "pipeline_only", "replicated"]
    kv_sharding: Literal["replicated", "shard_kv_heads", "shard_layers"]
    sharding_imbalance_fraction: float = Field(ge=0, lt=1)
    runtime_reserve: RuntimeReserve
    material_runtime_flags: tuple[str, ...]


class MemoryEstimate(FrozenModel):
    status: AnalysisStatus
    weights_total_bytes: int
    quantization_metadata_total_bytes: int
    weights_per_rank_bytes: int
    kv_bytes_per_token_logical: int
    kv_bytes_per_token_per_rank: int
    active_tokens_configured: int
    kv_cache_per_rank_bytes: int
    runtime_reserve_per_rank_bytes: int
    fragmentation_reserve_per_rank_bytes: int
    required_per_rank_bytes: int
    usable_device_bytes: int
    headroom_per_rank_bytes: int
    fits_per_rank: bool
    aggregate_capacity_lower_bound_gpus: int | None
    model_digest: Digest
    hardware_digest: Digest
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    assumptions: tuple[str, ...]


class RooflineMeasurement(FrozenModel):
    measurement_id: str
    measurement_status: Literal["measured", "planned"]
    scope_kind: Literal["kernel", "phase"]
    scope_name: str
    phase: Literal["prefill", "decode_step"] | None = None
    hardware_id: str
    hardware_revision: str
    precision: str
    flops: float | None = Field(default=None, gt=0)
    hbm_bytes_moved: float | None = Field(default=None, gt=0)
    duration_ms: float | None = Field(default=None, gt=0)
    capacity_used_bytes: int | None = Field(default=None, ge=0)
    generic_gpu_utilization_fraction: float | None = Field(default=None, ge=0, le=1)
    raw_artifact_digest: Digest | None = None
    provenance: SourceProvenance

    @model_validator(mode="after")
    def validate_measured_fields(self) -> Self:
        measured = (self.flops, self.hbm_bytes_moved, self.duration_ms)
        if self.measurement_status == "measured" and any(value is None for value in measured):
            raise ValueError("measured roofline input requires FLOPs, HBM bytes, and duration")
        if self.measurement_status == "measured" and self.raw_artifact_digest is None:
            raise ValueError("measured roofline input requires a raw artifact digest")
        if self.scope_kind == "phase" and self.phase is None:
            raise ValueError("phase measurements require phase=prefill or phase=decode_step")
        if self.scope_kind == "kernel" and self.phase is not None:
            raise ValueError("kernel measurements cannot also set a serving phase")
        banned = ("entire serving", "end-to-end", "end to end", "whole system")
        lowered = self.scope_name.lower()
        if any(phrase in lowered for phrase in banned):
            raise ValueError("roofline scope must be one kernel or phase, not the serving system")
        return self


class RooflineMetrics(FrozenModel):
    status: AnalysisStatus
    classification: BoundClassification
    arithmetic_intensity_flops_per_byte: float | None
    machine_balance_flops_per_byte: float | None
    achieved_tflops: float | None
    roofline_ceiling_tflops: float | None
    roofline_efficiency_fraction: float | None
    achieved_hbm_bandwidth_gbps: float | None
    capacity_pressure_fraction: float | None
    assumptions: tuple[str, ...]


class QualityGate(FrozenModel):
    evaluation_id: str
    model_id: str
    model_revision: str
    precision: str
    passed: bool
    metric_summary: str
    artifact_digest: Digest


class ServiceCurvePoint(FrozenModel):
    offered_rps: float = Field(gt=0)
    concurrency: int = Field(gt=0)
    achieved_goodput_rps: float = Field(ge=0)
    ttft_ms_p99: float = Field(gt=0)
    itl_ms_p99: float = Field(gt=0)
    tpot_ms_p99: float = Field(gt=0)
    e2e_ms_p99: float = Field(gt=0)
    per_user_tps: float = Field(ge=0)
    aggregate_tps: float = Field(ge=0)
    goodput_fraction: float = Field(ge=0, le=1)
    trial_count: int = Field(ge=3)
    dispersion_method: str = Field(min_length=1)


class ServiceCurve(FrozenModel):
    curve_id: str
    workload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    measured: bool
    metric_scope: Literal["inference_only", "end_to_end"]
    points: tuple[ServiceCurvePoint, ...] = Field(min_length=1)
    raw_artifact_digest: Digest
    benchmark_method_id: str
    provenance: SourceProvenance

    @model_validator(mode="after")
    def validate_measurement_integrity(self) -> Self:
        for point in self.points:
            if point.achieved_goodput_rps > point.offered_rps * (1 + 1e-6):
                raise ValueError("achieved goodput cannot exceed offered load")
        if self.measured and self.provenance.evidence_class != "measured":
            raise ValueError("a measured service curve requires measured provenance")
        return self


class SizingPolicy(FrozenModel):
    target_headroom_fraction: float = Field(ge=0, lt=1)
    failure_reserve_replicas: int = Field(ge=0)
    rollout_reserve_replicas: int = Field(ge=0)


class BenchmarkPlan(FrozenModel):
    plan_id: str
    status: Literal["required", "not_required"]
    required_workload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_metrics: tuple[str, ...]
    load_points_rps: tuple[float, ...]
    command_template: str
    equivalence_requirements: tuple[str, ...]


class DeploymentRecommendation(FrozenModel):
    status: AnalysisStatus
    memory_fit_lower_bound_gpus: int | None
    recommended_gpu_count: int | None
    active_replicas: int | None
    total_replicas_with_reserve: int | None
    gpus_per_replica: int
    matched_service_curve_id: str | None
    max_slo_goodput_rps_per_replica: float | None
    required_peak_rps_with_headroom: float
    quality_gate_evaluation_id: str
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]


class HypothesisSpec(FrozenModel):
    hypothesis_id: str
    category: Literal[
        "queue_scheduler",
        "prefill",
        "decode",
        "kv_capacity",
        "hbm_bandwidth",
        "compute",
        "interconnect",
        "host_tokenization",
        "network",
    ]
    statement: str
    expected_observation: str
    expected_metric_direction: Literal["increase", "decrease"]


class IncidentEvidence(FrozenModel):
    evidence: EvidenceRecord
    hypothesis_id: str
    stance: Literal["supporting", "contrary"]


class InterventionResult(FrozenModel):
    intervention_id: str
    hypothesis_id: str
    explicit_hypothesis: str
    changed_variable: str
    baseline_run_id: str
    rerun_id: str
    baseline_workload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rerun_workload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rerun_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_name: str
    dispersion: float = Field(gt=0)
    unit: str
    baseline_evidence_id: str
    rerun_evidence_id: str


class IncidentSpec(FrozenModel):
    incident_id: str
    baseline_run_id: str
    hypotheses: tuple[HypothesisSpec, ...] = Field(min_length=1)
    evidence: tuple[IncidentEvidence, ...] = Field(min_length=1)
    interventions: tuple[InterventionResult, ...]
    required_telemetry: tuple[str, ...]

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        hypotheses = {item.hypothesis_id for item in self.hypotheses}
        if any(item.hypothesis_id not in hypotheses for item in self.evidence):
            raise ValueError("incident evidence references unknown hypothesis")
        if any(item.hypothesis_id not in hypotheses for item in self.interventions):
            raise ValueError("intervention references unknown hypothesis")
        evidence_ids = {item.evidence.evidence_id for item in self.evidence}
        for intervention in self.interventions:
            if intervention.baseline_evidence_id not in evidence_ids:
                raise ValueError("intervention baseline evidence is missing")
            if intervention.rerun_evidence_id not in evidence_ids:
                raise ValueError("intervention rerun evidence is missing")
        return self


class RankedHypothesis(FrozenModel):
    hypothesis_id: str
    category: str
    score: int
    status: Literal["confirmed", "correlated", "inconclusive", "unsupported"]
    supporting_evidence_ids: tuple[str, ...]
    contrary_evidence_ids: tuple[str, ...]
    causal_gate_passed: bool


class Diagnosis(FrozenModel):
    status: AnalysisStatus
    root_cause_hypothesis_id: str | None
    ranked_hypotheses: tuple[RankedHypothesis, ...]
    missing_telemetry: tuple[str, ...]
    next_discriminating_experiment: str | None


class WorkloadAnalysis(EvidenceEnvelope):
    profile: WorkloadProfile


class MemoryAnalysis(EvidenceEnvelope):
    estimate: MemoryEstimate


class RooflineAnalysis(EvidenceEnvelope):
    metrics: RooflineMetrics


class SizingAnalysis(EvidenceEnvelope):
    recommendation: DeploymentRecommendation
    benchmark_plan: BenchmarkPlan


class DiagnosisAnalysis(EvidenceEnvelope):
    diagnosis: Diagnosis


class EngineeringAnalysis(EvidenceEnvelope):
    profile: WorkloadProfile
    memory: MemoryEstimate
    roofline: RooflineMetrics
    sizing: DeploymentRecommendation
    benchmark_plan: BenchmarkPlan
    diagnosis: Diagnosis | None = None
