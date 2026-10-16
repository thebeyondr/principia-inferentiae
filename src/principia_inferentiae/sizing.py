"""Fail-closed deployment sizing from memory fit and matched SLO-goodput."""

from __future__ import annotations

import math

from principia_inferentiae.contracts import (
    AnalysisStatus,
    BenchmarkPlan,
    Claim,
    Confidence,
    DeploymentRecommendation,
    EvidenceKind,
    EvidenceRecord,
    HardwareSpec,
    MemoryEstimate,
    ModelSpec,
    QualityGate,
    RunMetadata,
    ServiceCurve,
    ServiceCurvePoint,
    ServingConfig,
    SizingAnalysis,
    SizingPolicy,
    UnitRecord,
    WorkloadProfile,
)
from principia_inferentiae.io import digest_value
from principia_inferentiae.workload import assert_workload_hash

METHOD_ID = "deployment-sizing/1.0.0"


def serving_config_hash(config: ServingConfig) -> str:
    return digest_value(config).value


def _point_meets_request_slo(point: ServiceCurvePoint, profile: WorkloadProfile) -> bool:
    slo = profile.slo
    checks = [point.goodput_fraction >= slo.min_goodput_fraction]
    if slo.max_ttft_ms_p99 is not None:
        checks.append(point.ttft_ms_p99 <= slo.max_ttft_ms_p99)
    if slo.max_itl_ms_p99 is not None:
        checks.append(point.itl_ms_p99 <= slo.max_itl_ms_p99)
    if slo.max_tpot_ms_p99 is not None:
        checks.append(point.tpot_ms_p99 <= slo.max_tpot_ms_p99)
    if slo.max_e2e_ms_p99 is not None:
        checks.append(point.e2e_ms_p99 <= slo.max_e2e_ms_p99)
    if slo.min_per_user_tps is not None:
        checks.append(point.per_user_tps >= slo.min_per_user_tps)
    return all(checks)


def _benchmark_plan(
    run: RunMetadata,
    profile: WorkloadProfile,
    config: ServingConfig,
    required: bool,
) -> BenchmarkPlan:
    peak = profile.peak_one_second_arrival_rps
    load_points = tuple(
        sorted({round(max(0.001, peak * multiplier), 6) for multiplier in (0.50, 0.75, 1.00, 1.25)})
    )
    command = (
        "vllm bench serve --backend vllm --model ${PINNED_MODEL_PATH} "
        "--dataset-name timed_trace --dataset-path ${VLLM_TIMED_TRACE_PATH} "
        f"--num-prompts {profile.request_count} --self-timed "
        "--timed-trace-sec-multiplier ${TIME_MULTIPLIER} "
        "--timed-trace-chunk-hash-size 512 "
        f"--max-concurrency {config.max_concurrent_sequences} "
        f"--seed {run.random_seed}"
    )
    return BenchmarkPlan(
        plan_id=f"benchmark.{profile.workload_hash[:12]}.{serving_config_hash(config)[:12]}",
        status="required" if required else "not_required",
        required_workload_hash=profile.workload_hash,
        required_config_hash=serving_config_hash(config),
        required_metrics=(
            "ttft_ms_p99",
            "itl_ms_p99",
            "tpot_ms_p99",
            "e2e_ms_p99",
            "per_user_tps",
            "aggregate_tps",
            "achieved_goodput_rps",
            "goodput_fraction",
        ),
        load_points_rps=load_points,
        command_template=command,
        equivalence_requirements=(
            "Pin model artifact digest, runtime revision, precision, flags, and topology.",
            "Replay the canonical workload distribution and arrival shape.",
            "Scale offered load only through a recorded timed-trace multiplier.",
            "Record warmup, trial count, dispersion, failures, and raw artifact digest.",
            "State inference-only versus end-to-end metric boundaries.",
        ),
    )


def size_deployment(
    run: RunMetadata,
    profile: WorkloadProfile,
    memory: MemoryEstimate,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    quality_gate: QualityGate,
    policy: SizingPolicy,
    service_curve: ServiceCurve | None,
) -> SizingAnalysis:
    """Recommend capacity only when every identity and evidence gate passes."""

    assert_workload_hash(profile)
    config_hash = serving_config_hash(config)
    if (
        memory.config_hash != config_hash
        or memory.model_digest != digest_value(model)
        or memory.hardware_digest != digest_value(hardware)
    ):
        raise ValueError("memory estimate identity does not match model, hardware, and config")
    required_peak = profile.peak_one_second_arrival_rps / (1 - policy.target_headroom_fraction)
    gpus_per_replica = config.tensor_parallel_size * config.pipeline_parallel_size
    evidence_items = [
        EvidenceRecord(
            evidence_id="sizing.workload",
            kind=EvidenceKind.INPUT,
            description="Canonical workload profile and SLO",
            observed_value=profile.workload_hash,
            unit="sha256",
            source_digest=digest_value(profile),
        ),
        EvidenceRecord(
            evidence_id="sizing.memory",
            kind=EvidenceKind.ESTIMATE,
            description="Per-rank memory-fit estimate",
            observed_value=str(memory.required_per_rank_bytes),
            unit="bytes/rank",
            source_digest=digest_value(memory),
        ),
        EvidenceRecord(
            evidence_id="sizing.config",
            kind=EvidenceKind.CONFIGURATION,
            description="Serving configuration identity",
            observed_value=config_hash,
            unit="sha256",
            source_digest=digest_value(config),
        ),
        EvidenceRecord(
            evidence_id="sizing.quality_gate",
            kind=EvidenceKind.ARTIFACT,
            description="Independent model/precision quality-selection result",
            observed_value=str(quality_gate.passed).lower(),
            unit="boolean",
            source_digest=quality_gate.artifact_digest,
        ),
    ]
    if service_curve is not None:
        evidence_items.append(
            EvidenceRecord(
                evidence_id="sizing.service_curve",
                kind=EvidenceKind.MEASURED_METRIC,
                description="Candidate measured service curve",
                observed_value=service_curve.curve_id,
                unit="curve_id",
                source_digest=service_curve.raw_artifact_digest,
            )
        )
    evidence = tuple(evidence_items)

    missing: tuple[str, ...] = ()
    limitations: tuple[str, ...]
    matched_curve_id: str | None = None
    max_goodput: float | None = None
    active_replicas: int | None = None
    total_replicas: int | None = None
    recommended_gpu_count: int | None = None

    quality_identity_matches = (
        quality_gate.model_id == config.model_id
        and quality_gate.model_revision == config.model_revision
        and quality_gate.precision == config.precision
    )
    workload_fits_config = profile.total_tokens.p99 <= config.max_sequence_tokens and (
        profile.peak_observed_concurrency is None
        or profile.peak_observed_concurrency <= config.max_concurrent_sequences
    )
    curve_identity_matches = (
        service_curve is not None
        and service_curve.provenance.evidence_class == "measured"
        and service_curve.workload_hash == profile.workload_hash
        and service_curve.config_hash == config_hash
        and service_curve.metric_scope == profile.slo.metric_scope
    )

    if not memory.fits_per_rank:
        status = AnalysisStatus.MEMORY_NOT_FIT
        limitations = ("The configured model state does not fit per rank.",)
        statement = (
            "No deployment is recommended because the configured per-rank state does not fit."
        )
        confidence = Confidence.HIGH
    elif not workload_fits_config:
        status = AnalysisStatus.INSUFFICIENT_EVIDENCE
        missing = (
            "configured max sequence and concurrency cover the observed p99 and peak concurrency",
        )
        limitations = (
            "A serving configuration that rejects the observed tokens or concurrency "
            "cannot be sized.",
        )
        statement = (
            "No deployment is recommended because the configured sequence or concurrency "
            "limit is below the observed workload."
        )
        confidence = Confidence.HIGH
    elif not quality_gate.passed or not quality_identity_matches:
        status = AnalysisStatus.QUALITY_GATE_FAILED
        limitations = (
            "Quality selection is a separate gate and did not pass for this model "
            "revision and precision.",
        )
        statement = (
            "No deployment is recommended because the independent quality gate did not pass."
        )
        confidence = Confidence.HIGH
    elif not curve_identity_matches:
        status = AnalysisStatus.INSUFFICIENT_EVIDENCE
        if service_curve is None:
            missing = ("workload- and config-matched measured service curve",)
        else:
            missing = (
                "measured service curve with matching workload hash, config hash, and metric scope",
            )
        limitations = (
            "Memory capacity establishes only a lower bound, not production GPU count.",
            "No throughput or latency extrapolation is made from mismatched evidence.",
        )
        lower_bound = memory.aggregate_capacity_lower_bound_gpus
        statement = (
            "Memory-fit lower bound is "
            f"{'null' if lower_bound is None else lower_bound} GPU(s), "
            "but recommended_gpu_count is null without a matched measured service curve."
        )
        confidence = Confidence.HIGH
    else:
        assert service_curve is not None
        good_points = tuple(
            point for point in service_curve.points if _point_meets_request_slo(point, profile)
        )
        matched_curve_id = service_curve.curve_id
        if not good_points:
            status = AnalysisStatus.SLO_NOT_MET
            limitations = ("The measured curve has no point satisfying every declared SLO.",)
            statement = "No deployment is recommended because measured points miss the SLO."
            confidence = Confidence.HIGH
        else:
            best_point = max(good_points, key=lambda point: point.achieved_goodput_rps)
            max_goodput = best_point.achieved_goodput_rps
            active_replicas = max(1, math.ceil(required_peak / max_goodput))
            aggregate_tps_ok = (
                profile.slo.min_aggregate_tps is None
                or active_replicas * best_point.aggregate_tps >= profile.slo.min_aggregate_tps
            )
            if not aggregate_tps_ok:
                status = AnalysisStatus.SLO_NOT_MET
                limitations = (
                    "Per-replica request SLOs pass, but replica aggregate tokens/s miss "
                    "the system aggregate target.",
                )
                statement = (
                    "No deployment is recommended because aggregate tokens/s stay below the SLO."
                )
                confidence = Confidence.HIGH
                active_replicas = None
                max_goodput = None
            else:
                total_replicas = (
                    active_replicas
                    + policy.failure_reserve_replicas
                    + policy.rollout_reserve_replicas
                )
                recommended_gpu_count = total_replicas * gpus_per_replica
                status = AnalysisStatus.SUPPORTED
                limitations = (
                    "Recommendation is conditional on workload stationarity and "
                    "service-curve equivalence.",
                    "Cost, placement, and failure domains are not optimized.",
                )
                statement = (
                    f"Recommend {recommended_gpu_count} GPU(s): "
                    f"{active_replicas} active replica(s) "
                    f"plus {policy.failure_reserve_replicas} failure and "
                    f"{policy.rollout_reserve_replicas} rollout reserve replica(s)."
                )
                confidence = Confidence.HIGH

    recommendation = DeploymentRecommendation(
        status=status,
        memory_fit_lower_bound_gpus=memory.aggregate_capacity_lower_bound_gpus,
        recommended_gpu_count=recommended_gpu_count,
        active_replicas=active_replicas,
        total_replicas_with_reserve=total_replicas,
        gpus_per_replica=gpus_per_replica,
        matched_service_curve_id=matched_curve_id,
        max_slo_goodput_rps_per_replica=max_goodput,
        required_peak_rps_with_headroom=round(required_peak, 6),
        quality_gate_evaluation_id=quality_gate.evaluation_id,
        assumptions=(
            "Each service-curve point represents one replica at the declared TP/PP topology.",
            "Replica capacity scales linearly only after identical replicas are measured.",
            "Failure and rollout reserves are additive and unavailable to steady-state load.",
        ),
        limitations=limitations,
    )
    benchmark_plan = _benchmark_plan(
        run,
        profile,
        config,
        required=status is not AnalysisStatus.SUPPORTED,
    )
    evidence_ids = tuple(item.evidence_id for item in evidence)
    claim = Claim(
        claim_id="deployment.recommendation",
        claim_type="deployment_sizing",
        statement=statement,
        status=status,
        evidence_ids=evidence_ids,
        supporting_evidence_ids=evidence_ids,
        method_id=METHOD_ID,
        units=(
            UnitRecord(field="memory_fit_lower_bound_gpus", unit="GPU"),
            UnitRecord(field="recommended_gpu_count", unit="GPU"),
            UnitRecord(field="required_peak_rps_with_headroom", unit="requests/s"),
            UnitRecord(field="max_slo_goodput_rps_per_replica", unit="good requests/s/replica"),
        ),
        assumptions=recommendation.assumptions,
        limitations=limitations,
        confidence=confidence,
        missing_telemetry=missing,
        next_discriminating_experiment=(
            None
            if status is AnalysisStatus.SUPPORTED
            else (
                "Execute the generated benchmark plan and ingest the complete measured "
                "service curve."
            )
        ),
    )
    return SizingAnalysis(
        run=run,
        evidence=evidence,
        claims=(claim,),
        recommendation=recommendation,
        benchmark_plan=benchmark_plan,
    )
