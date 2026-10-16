"""Measured kernel/phase roofline analysis with capacity kept separate."""

from __future__ import annotations

from principia_inferentiae.contracts import (
    AnalysisStatus,
    BoundClassification,
    Claim,
    Confidence,
    EvidenceKind,
    EvidenceRecord,
    HardwareSpec,
    RooflineAnalysis,
    RooflineMeasurement,
    RooflineMetrics,
    RunMetadata,
    UnitRecord,
)
from principia_inferentiae.io import digest_value

METHOD_ID = "empirical-roofline/1.0.0"
UNDERFILLED_EFFICIENCY = 0.60
RIDGE_LOWER_RATIO = 0.80
RIDGE_UPPER_RATIO = 1.25


def _insufficient_result(
    run: RunMetadata,
    measurement: RooflineMeasurement,
    hardware: HardwareSpec,
    missing: tuple[str, ...],
) -> RooflineAnalysis:
    measurement_digest = digest_value(measurement)
    hardware_digest = digest_value(hardware)
    evidence = (
        EvidenceRecord(
            evidence_id="roofline.measurement",
            kind=EvidenceKind.INPUT,
            description="Roofline measurement contract",
            observed_value=measurement_digest.value,
            unit="sha256",
            source_digest=measurement_digest,
        ),
        EvidenceRecord(
            evidence_id="roofline.hardware",
            kind=EvidenceKind.INPUT,
            description="Empirical hardware ceiling contract",
            observed_value=hardware_digest.value,
            unit="sha256",
            source_digest=hardware_digest,
        ),
    )
    metrics = RooflineMetrics(
        status=AnalysisStatus.INSUFFICIENT_EVIDENCE,
        classification=BoundClassification.INSUFFICIENT_EVIDENCE,
        arithmetic_intensity_flops_per_byte=None,
        machine_balance_flops_per_byte=None,
        achieved_tflops=None,
        roofline_ceiling_tflops=None,
        roofline_efficiency_fraction=None,
        achieved_hbm_bandwidth_gbps=None,
        capacity_pressure_fraction=None,
        assumptions=(
            "Roofline classification is valid only for a measured kernel or defined phase.",
            "Generic GPU utilization is not evidence of compute saturation.",
        ),
    )
    claim = Claim(
        claim_id="roofline.classification",
        claim_type="kernel_or_phase_roofline",
        statement="No roofline bottleneck classification is supported by the supplied evidence.",
        status=AnalysisStatus.INSUFFICIENT_EVIDENCE,
        evidence_ids=tuple(item.evidence_id for item in evidence),
        method_id=METHOD_ID,
        units=(
            UnitRecord(field="arithmetic_intensity", unit="FLOP/byte"),
            UnitRecord(field="achieved_compute", unit="TFLOP/s"),
            UnitRecord(field="achieved_hbm_bandwidth", unit="GB/s"),
        ),
        assumptions=metrics.assumptions,
        limitations=(
            "HBM capacity pressure, if known, is reported separately and cannot select "
            "a roofline class.",
        ),
        confidence=Confidence.NONE,
        missing_telemetry=missing,
        next_discriminating_experiment=(
            "Profile the named kernel or phase and record FLOPs, HBM bytes moved, duration, "
            "precision, and the raw-artifact digest."
        ),
    )
    return RooflineAnalysis(run=run, evidence=evidence, claims=(claim,), metrics=metrics)


def analyze_roofline(
    run: RunMetadata,
    measurement: RooflineMeasurement,
    hardware: HardwareSpec,
) -> RooflineAnalysis:
    """Classify one measured scope against empirical precision-specific ceilings."""

    if measurement.measurement_status != "measured":
        return _insufficient_result(
            run,
            measurement,
            hardware,
            ("measured FLOPs", "measured HBM bytes", "measured duration", "raw artifact"),
        )
    if (
        measurement.hardware_id != hardware.hardware_id
        or measurement.hardware_revision != hardware.revision
    ):
        return _insufficient_result(
            run,
            measurement,
            hardware,
            ("roofline measurement hardware identity matching the ceiling contract",),
        )
    ceiling = next(
        (item for item in hardware.compute_ceilings if item.precision == measurement.precision),
        None,
    )
    if ceiling is None:
        return _insufficient_result(
            run,
            measurement,
            hardware,
            (f"empirical compute ceiling for precision {measurement.precision}",),
        )
    if (
        measurement.provenance.evidence_class != "measured"
        or ceiling.provenance.evidence_class != "measured"
        or hardware.provenance.evidence_class != "measured"
    ):
        return _insufficient_result(
            run,
            measurement,
            hardware,
            ("measured provenance for the profile and both empirical ceilings",),
        )

    assert measurement.flops is not None
    assert measurement.hbm_bytes_moved is not None
    assert measurement.duration_ms is not None
    duration_seconds = measurement.duration_ms / 1000
    arithmetic_intensity = measurement.flops / measurement.hbm_bytes_moved
    machine_balance = ceiling.empirical_tflops * 1000 / hardware.empirical_hbm_bandwidth_gbps
    achieved_tflops = measurement.flops / duration_seconds / 1e12
    achieved_hbm_gbps = measurement.hbm_bytes_moved / duration_seconds / 1e9
    bandwidth_roof_tflops = hardware.empirical_hbm_bandwidth_gbps * arithmetic_intensity / 1000
    roofline_ceiling_tflops = min(ceiling.empirical_tflops, bandwidth_roof_tflops)
    efficiency = achieved_tflops / roofline_ceiling_tflops
    capacity_pressure = (
        None
        if measurement.capacity_used_bytes is None
        else measurement.capacity_used_bytes / hardware.device_memory_bytes
    )

    measurement_digest = digest_value(measurement)
    hardware_digest = digest_value(hardware)
    evidence = (
        EvidenceRecord(
            evidence_id="roofline.measurement",
            kind=EvidenceKind.MEASURED_METRIC,
            description=f"Measured {measurement.scope_kind}: {measurement.scope_name}",
            observed_value=measurement_digest.value,
            unit="sha256",
            source_digest=measurement_digest,
        ),
        EvidenceRecord(
            evidence_id="roofline.hardware",
            kind=EvidenceKind.MEASURED_METRIC
            if hardware.provenance.evidence_class == "measured"
            else EvidenceKind.INPUT,
            description="Empirical precision-specific compute and HBM ceilings",
            observed_value=hardware_digest.value,
            unit="sha256",
            source_digest=hardware_digest,
        ),
    )

    support: tuple[str, ...]
    contrary: tuple[str, ...]
    missing: tuple[str, ...]
    if efficiency > 1.05:
        status = AnalysisStatus.INCONCLUSIVE
        classification = BoundClassification.INSUFFICIENT_EVIDENCE
        support = ("roofline.measurement",)
        contrary = ("roofline.hardware",)
        confidence = Confidence.NONE
        statement = (
            "Measured throughput exceeds the supplied empirical roofline by more than 5%; "
            "units, scope, or ceiling provenance conflict."
        )
        missing = ("compatible empirical ceilings or corrected measurement units",)
        next_experiment = (
            "Re-measure the scope and ceilings on the same hardware state with identical precision."
        )
    else:
        status = AnalysisStatus.SUPPORTED
        ratio = arithmetic_intensity / machine_balance
        if efficiency < UNDERFILLED_EFFICIENCY:
            classification = BoundClassification.UNDERFILLED
        elif ratio < RIDGE_LOWER_RATIO:
            classification = BoundClassification.HBM_BANDWIDTH_BOUND
        elif ratio > RIDGE_UPPER_RATIO:
            classification = BoundClassification.COMPUTE_BOUND
        else:
            classification = BoundClassification.MIXED
        support = tuple(item.evidence_id for item in evidence)
        contrary = ()
        confidence = Confidence.HIGH
        statement = (
            f"The measured {measurement.scope_kind} {measurement.scope_name!r} is "
            f"{classification.value} at {arithmetic_intensity:.3f} FLOP/byte and "
            f"{efficiency:.3f} of its empirical roofline."
        )
        missing = ()
        next_experiment = (
            "Vary batch/token shape while preserving precision and profile the same scope "
            "to test whether the classification moves across the ridge point."
        )

    assumptions = (
        "FLOPs and HBM bytes cover the same measured scope and duration.",
        "Decimal GB/s and TFLOP/s units are used.",
        "Compute and HBM ceilings are empirical for the declared precision.",
        "Capacity pressure is independent of roofline classification.",
        "Bound labels require at least 0.60 of the applicable empirical roof.",
        "The ridge band is 0.80 to 1.25 times machine balance.",
    )
    metrics = RooflineMetrics(
        status=status,
        classification=classification,
        arithmetic_intensity_flops_per_byte=round(arithmetic_intensity, 6),
        machine_balance_flops_per_byte=round(machine_balance, 6),
        achieved_tflops=round(achieved_tflops, 6),
        roofline_ceiling_tflops=round(roofline_ceiling_tflops, 6),
        roofline_efficiency_fraction=round(efficiency, 6),
        achieved_hbm_bandwidth_gbps=round(achieved_hbm_gbps, 6),
        capacity_pressure_fraction=(
            None if capacity_pressure is None else round(capacity_pressure, 6)
        ),
        assumptions=assumptions,
    )
    claim = Claim(
        claim_id="roofline.classification",
        claim_type="kernel_or_phase_roofline",
        statement=statement,
        status=status,
        evidence_ids=tuple(item.evidence_id for item in evidence),
        supporting_evidence_ids=support,
        contrary_evidence_ids=contrary,
        method_id=METHOD_ID,
        units=(
            UnitRecord(field="arithmetic_intensity", unit="FLOP/byte"),
            UnitRecord(field="achieved_tflops", unit="TFLOP/s"),
            UnitRecord(field="achieved_hbm_bandwidth_gbps", unit="GB/s"),
            UnitRecord(field="capacity_pressure_fraction", unit="fraction"),
        ),
        assumptions=assumptions,
        limitations=(
            "The result applies only to the named measured scope, not end-to-end serving.",
            "Generic GPU utilization was ignored.",
        ),
        confidence=confidence,
        missing_telemetry=missing,
        next_discriminating_experiment=next_experiment,
    )
    return RooflineAnalysis(run=run, evidence=evidence, claims=(claim,), metrics=metrics)
