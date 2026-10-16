from __future__ import annotations

import pytest
from pydantic import ValidationError

from principia_inferentiae.contracts import (
    AnalysisStatus,
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
    RunMetadata,
    ServiceCurve,
    ServiceCurvePoint,
    ServingConfig,
    SizingPolicy,
    SLOSpec,
    SourceProvenance,
    TraceRequest,
)
from principia_inferentiae.diagnosis import diagnose_incident
from principia_inferentiae.memory import estimate_memory
from principia_inferentiae.sizing import serving_config_hash, size_deployment
from principia_inferentiae.workload import characterize_workload


def _curve(
    workload_hash: str,
    config_hash: str,
    provenance: SourceProvenance,
    *,
    meets_slo: bool = True,
) -> ServiceCurve:
    return ServiceCurve(
        curve_id="curve-1",
        workload_hash=workload_hash,
        config_hash=config_hash,
        measured=provenance.evidence_class == "measured",
        metric_scope="end_to_end",
        points=(
            ServiceCurvePoint(
                offered_rps=4,
                concurrency=8,
                achieved_goodput_rps=4 if meets_slo else 0,
                ttft_ms_p99=1_000 if meets_slo else 2_000,
                itl_ms_p99=50 if meets_slo else 100,
                tpot_ms_p99=50 if meets_slo else 100,
                e2e_ms_p99=8_000 if meets_slo else 12_000,
                per_user_tps=20 if meets_slo else 10,
                aggregate_tps=200 if meets_slo else 80,
                goodput_fraction=0.99 if meets_slo else 0,
                trial_count=3,
                dispersion_method="IQR",
            ),
        ),
        raw_artifact_digest=Digest(algorithm="sha256", value="f" * 64),
        benchmark_method_id="fixture-benchmark",
        provenance=provenance,
    )


def test_no_service_curve_returns_only_memory_lower_bound(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    quality_gate: QualityGate,
    policy: SizingPolicy,
) -> None:
    fitting = _fitting(config)
    workload = characterize_workload(run, trace, slo)
    memory = estimate_memory(run, model, hardware, fitting)
    result = size_deployment(
        run,
        workload.profile,
        memory.estimate,
        model,
        hardware,
        fitting,
        quality_gate,
        policy,
        None,
    )
    assert result.recommendation.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    lower_bound = result.recommendation.memory_fit_lower_bound_gpus
    assert lower_bound is not None
    assert lower_bound >= 1
    assert result.recommendation.recommended_gpu_count is None
    assert result.benchmark_plan.status == "required"


def test_mismatched_workload_or_config_hash_is_rejected(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    quality_gate: QualityGate,
    policy: SizingPolicy,
) -> None:
    fitting = _fitting(config)
    workload = characterize_workload(run, trace, slo)
    memory = estimate_memory(run, model, hardware, fitting)
    curves = (
        _curve("0" * 64, serving_config_hash(fitting), hardware.provenance),
        _curve(workload.profile.workload_hash, "0" * 64, hardware.provenance),
    )
    for curve in curves:
        result = size_deployment(
            run,
            workload.profile,
            memory.estimate,
            model,
            hardware,
            fitting,
            quality_gate,
            policy,
            curve,
        )
        assert result.recommendation.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
        assert result.recommendation.matched_service_curve_id is None
        assert result.recommendation.recommended_gpu_count is None


def _fitting(config: ServingConfig) -> ServingConfig:
    return config.model_copy(update={"max_sequence_tokens": 16_384, "max_batched_tokens": 16_384})


def test_matched_slo_goodput_sizes_replicas_and_reserve(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    quality_gate: QualityGate,
    policy: SizingPolicy,
) -> None:
    fitting = _fitting(config)
    workload = characterize_workload(run, trace, slo)
    memory = estimate_memory(run, model, hardware, fitting)
    measured = hardware.provenance.model_copy(
        update={"evidence_class": "measured", "usage_note": "Measured service-curve fixture."}
    )
    curve = _curve(
        workload.profile.workload_hash,
        serving_config_hash(fitting),
        measured,
    )
    result = size_deployment(
        run,
        workload.profile,
        memory.estimate,
        model,
        hardware,
        fitting,
        quality_gate,
        policy,
        curve,
    )
    assert result.recommendation.status is AnalysisStatus.SUPPORTED
    assert result.recommendation.active_replicas == 1
    assert result.recommendation.total_replicas_with_reserve == 3
    assert result.recommendation.recommended_gpu_count == 6


def test_measured_curve_that_misses_slo_does_not_extrapolate(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    quality_gate: QualityGate,
    policy: SizingPolicy,
) -> None:
    fitting = _fitting(config)
    workload = characterize_workload(run, trace, slo)
    memory = estimate_memory(run, model, hardware, fitting)
    measured = hardware.provenance.model_copy(
        update={"evidence_class": "measured", "usage_note": "Measured service-curve fixture."}
    )
    curve = _curve(
        workload.profile.workload_hash,
        serving_config_hash(fitting),
        measured,
        meets_slo=False,
    )
    result = size_deployment(
        run,
        workload.profile,
        memory.estimate,
        model,
        hardware,
        fitting,
        quality_gate,
        policy,
        curve,
    )
    assert result.recommendation.status is AnalysisStatus.SLO_NOT_MET
    assert result.recommendation.recommended_gpu_count is None


def _evidence(
    evidence_id: str,
    kind: EvidenceKind,
    description: str,
    timestamp_ms: int,
    value: str,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        kind=kind,
        description=description,
        observed_value=value,
        unit="count",
        source_digest=Digest(algorithm="sha256", value=evidence_id[0] * 64),
        timestamp_ms=timestamp_ms,
        event_type="oom" if evidence_id == "c-oom" else None,
    )


def test_correlation_only_incident_cannot_claim_root_cause(run: RunMetadata) -> None:
    hypothesis = HypothesisSpec(
        hypothesis_id="h-queue",
        category="queue_scheduler",
        statement="Queueing caused the TTFT regression.",
        expected_observation="Queue depth rises before TTFT.",
        expected_metric_direction="increase",
    )
    correlation = _evidence(
        "a-correlation",
        EvidenceKind.CORRELATION,
        "queue_depth",
        100,
        "0.91 correlation with TTFT",
    )
    incident = IncidentSpec(
        incident_id="incident-correlation",
        baseline_run_id="baseline-1",
        hypotheses=(hypothesis,),
        evidence=(
            IncidentEvidence(
                evidence=correlation,
                hypothesis_id=hypothesis.hypothesis_id,
                stance="supporting",
            ),
        ),
        interventions=(),
        required_telemetry=("queue_depth", "scheduler_wait_ms"),
    )
    result = diagnose_incident(run, incident)
    assert result.diagnosis.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert result.diagnosis.root_cause_hypothesis_id is None
    assert result.diagnosis.ranked_hypotheses[0].status == "correlated"


def test_confirmed_oom_requires_direct_timeline_and_causal_rerun(
    run: RunMetadata,
) -> None:
    hypothesis = HypothesisSpec(
        hypothesis_id="h-kv",
        category="kv_capacity",
        statement="KV capacity exhaustion caused CUDA OOM.",
        expected_observation="KV occupancy precedes OOM; reducing concurrency removes OOM.",
        expected_metric_direction="decrease",
    )
    occupancy = _evidence(
        "b-occupancy",
        EvidenceKind.MEASURED_METRIC,
        "gpu_kv_cache_usage",
        100,
        "0.99",
    )
    oom = _evidence(
        "c-oom",
        EvidenceKind.DIRECT_EVENT,
        "CUDA OOM event",
        120,
        "1",
    )
    rerun = _evidence(
        "d-rerun",
        EvidenceKind.INTERVENTION,
        "rerun_oom_count",
        200,
        "0",
    )
    links = tuple(
        IncidentEvidence(
            evidence=item,
            hypothesis_id=hypothesis.hypothesis_id,
            stance="supporting",
        )
        for item in (occupancy, oom, rerun)
    )
    incident = IncidentSpec(
        incident_id="incident-oom",
        baseline_run_id="baseline-oom",
        hypotheses=(hypothesis,),
        evidence=links,
        interventions=(
            InterventionResult(
                intervention_id="i-1",
                hypothesis_id=hypothesis.hypothesis_id,
                explicit_hypothesis=hypothesis.statement,
                changed_variable="max_concurrent_sequences: 16 -> 8",
                baseline_run_id="baseline-oom",
                rerun_id="rerun-oom",
                baseline_workload_hash="a" * 64,
                rerun_workload_hash="a" * 64,
                baseline_config_hash="b" * 64,
                rerun_config_hash="c" * 64,
                metric_name="oom_count",
                dispersion=0.1,
                unit="count",
                baseline_evidence_id=oom.evidence_id,
                rerun_evidence_id=rerun.evidence_id,
            ),
        ),
        required_telemetry=(
            "gpu_kv_cache_usage",
            "CUDA OOM event",
            "rerun_oom_count",
        ),
    )
    result = diagnose_incident(run, incident)
    assert result.diagnosis.status is AnalysisStatus.SUPPORTED
    assert result.diagnosis.root_cause_hypothesis_id == hypothesis.hypothesis_id
    assert result.diagnosis.ranked_hypotheses[0].causal_gate_passed


def test_conflicting_evidence_remains_inconclusive(run: RunMetadata) -> None:
    hypothesis = HypothesisSpec(
        hypothesis_id="h-hbm",
        category="hbm_bandwidth",
        statement="HBM bandwidth saturation caused decode regression.",
        expected_observation="Measured bandwidth reaches the empirical ceiling.",
        expected_metric_direction="increase",
    )
    support = _evidence(
        "e-support",
        EvidenceKind.MEASURED_METRIC,
        "decode_hbm_bandwidth",
        100,
        "0.95 ceiling",
    )
    contrary = _evidence(
        "f-contrary",
        EvidenceKind.MEASURED_METRIC,
        "decode_hbm_bandwidth_control",
        100,
        "0.35 ceiling",
    )
    incident = IncidentSpec(
        incident_id="incident-conflict",
        baseline_run_id="baseline-conflict",
        hypotheses=(hypothesis,),
        evidence=(
            IncidentEvidence(
                evidence=support,
                hypothesis_id=hypothesis.hypothesis_id,
                stance="supporting",
            ),
            IncidentEvidence(
                evidence=contrary,
                hypothesis_id=hypothesis.hypothesis_id,
                stance="contrary",
            ),
        ),
        interventions=(),
        required_telemetry=("decode_hbm_bandwidth",),
    )
    result = diagnose_incident(run, incident)
    assert result.diagnosis.status is AnalysisStatus.INCONCLUSIVE
    assert result.diagnosis.root_cause_hypothesis_id is None
    assert result.claims[0].supporting_evidence_ids
    assert result.claims[0].contrary_evidence_ids


def test_synthetic_curve_cannot_recommend_gpus(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    quality_gate: QualityGate,
    policy: SizingPolicy,
) -> None:
    fitting = _fitting(config)
    workload = characterize_workload(run, trace, slo)
    memory = estimate_memory(run, model, hardware, fitting)
    curve = _curve(
        workload.profile.workload_hash,
        serving_config_hash(fitting),
        hardware.provenance,
    )
    result = size_deployment(
        run,
        workload.profile,
        memory.estimate,
        model,
        hardware,
        fitting,
        quality_gate,
        policy,
        curve,
    )
    assert result.recommendation.recommended_gpu_count is None
    assert result.recommendation.status is AnalysisStatus.INSUFFICIENT_EVIDENCE


def test_goodput_above_offered_load_is_rejected(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
) -> None:
    workload = characterize_workload(run, trace, slo)
    curve = _curve(
        workload.profile.workload_hash,
        serving_config_hash(config),
        hardware.provenance,
    )
    payload = curve.model_dump()
    payload["points"][0]["achieved_goodput_rps"] = 5
    with pytest.raises(ValidationError, match="achieved goodput cannot exceed offered load"):
        ServiceCurve.model_validate(payload)


def test_sequence_limit_below_workload_does_not_size(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    quality_gate: QualityGate,
    policy: SizingPolicy,
) -> None:
    workload = characterize_workload(run, trace, slo)
    memory = estimate_memory(run, model, hardware, config)
    measured = hardware.provenance.model_copy(
        update={"evidence_class": "measured", "usage_note": "Measured service-curve fixture."}
    )
    curve = _curve(
        workload.profile.workload_hash,
        serving_config_hash(config),
        measured,
    )
    result = size_deployment(
        run,
        workload.profile,
        memory.estimate,
        model,
        hardware,
        config,
        quality_gate,
        policy,
        curve,
    )
    assert result.recommendation.recommended_gpu_count is None
    assert "sequence or concurrency" in result.claims[0].statement


def test_memory_identity_mismatch_raises(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    quality_gate: QualityGate,
    policy: SizingPolicy,
) -> None:
    workload = characterize_workload(run, trace, slo)
    memory = estimate_memory(run, model, hardware, config)
    with pytest.raises(ValueError, match="memory estimate identity"):
        size_deployment(
            run,
            workload.profile,
            memory.estimate,
            model,
            hardware,
            _fitting(config),
            quality_gate,
            policy,
            None,
        )


def test_tiny_wrong_direction_delta_is_not_a_root_cause(run: RunMetadata) -> None:
    hypothesis = HypothesisSpec(
        hypothesis_id="h-kv",
        category="kv_capacity",
        statement="KV capacity exhaustion caused CUDA OOM.",
        expected_observation="Reducing concurrency removes OOM.",
        expected_metric_direction="decrease",
    )
    baseline = _evidence("c-oom", EvidenceKind.DIRECT_EVENT, "CUDA OOM event", 120, "1")
    rerun = _evidence("d-rerun", EvidenceKind.INTERVENTION, "rerun_oom_count", 200, "1.0001")
    occupancy = _evidence(
        "b-occupancy",
        EvidenceKind.MEASURED_METRIC,
        "gpu_kv_cache_usage",
        100,
        "0.99",
    )
    incident = IncidentSpec(
        incident_id="incident-tiny",
        baseline_run_id="baseline-oom",
        hypotheses=(hypothesis,),
        evidence=tuple(
            IncidentEvidence(
                evidence=item,
                hypothesis_id=hypothesis.hypothesis_id,
                stance="supporting",
            )
            for item in (occupancy, baseline, rerun)
        ),
        interventions=(
            InterventionResult(
                intervention_id="i-1",
                hypothesis_id=hypothesis.hypothesis_id,
                explicit_hypothesis=hypothesis.statement,
                changed_variable="max_concurrent_sequences: 16 -> 8",
                baseline_run_id="baseline-oom",
                rerun_id="rerun-oom",
                baseline_workload_hash="a" * 64,
                rerun_workload_hash="a" * 64,
                baseline_config_hash="b" * 64,
                rerun_config_hash="c" * 64,
                metric_name="oom_count",
                dispersion=0.1,
                unit="count",
                baseline_evidence_id=baseline.evidence_id,
                rerun_evidence_id=rerun.evidence_id,
            ),
        ),
        required_telemetry=("gpu_kv_cache_usage", "CUDA OOM event", "rerun_oom_count"),
    )
    result = diagnose_incident(run, incident)
    assert result.diagnosis.root_cause_hypothesis_id is None
    assert result.diagnosis.ranked_hypotheses[0].causal_gate_passed is False
