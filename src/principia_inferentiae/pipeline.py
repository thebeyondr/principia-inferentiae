"""Composition of the first vertical slice into one validated analysis."""

from __future__ import annotations

from principia_inferentiae.contracts import (
    EngineeringAnalysis,
    HardwareSpec,
    ModelSpec,
    QualityGate,
    RooflineMeasurement,
    RunMetadata,
    ServiceCurve,
    ServingConfig,
    SizingPolicy,
    SLOSpec,
    TraceRequest,
)
from principia_inferentiae.memory import estimate_memory
from principia_inferentiae.roofline import analyze_roofline
from principia_inferentiae.sizing import size_deployment
from principia_inferentiae.workload import characterize_workload


def analyze_serving_candidate(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    roofline_measurement: RooflineMeasurement,
    quality_gate: QualityGate,
    sizing_policy: SizingPolicy,
    service_curve: ServiceCurve | None = None,
) -> EngineeringAnalysis:
    """Execute the deterministic workload → fit → roofline → sizing pipeline."""

    workload = characterize_workload(run, trace, slo)
    memory = estimate_memory(run, model, hardware, config)
    roofline = analyze_roofline(run, roofline_measurement, hardware)
    sizing = size_deployment(
        run,
        workload.profile,
        memory.estimate,
        model,
        hardware,
        config,
        quality_gate,
        sizing_policy,
        service_curve,
    )
    return EngineeringAnalysis(
        run=run,
        evidence=workload.evidence + memory.evidence + roofline.evidence + sizing.evidence,
        claims=workload.claims + memory.claims + roofline.claims + sizing.claims,
        profile=workload.profile,
        memory=memory.estimate,
        roofline=roofline.metrics,
        sizing=sizing.recommendation,
        benchmark_plan=sizing.benchmark_plan,
    )
