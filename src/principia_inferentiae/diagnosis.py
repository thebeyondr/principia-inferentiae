"""Evidence-ranked incident diagnosis with an explicit causal gate."""

from __future__ import annotations

from typing import Literal

from principia_inferentiae.contracts import (
    AnalysisStatus,
    Claim,
    Confidence,
    Diagnosis,
    DiagnosisAnalysis,
    EvidenceKind,
    HypothesisSpec,
    IncidentEvidence,
    IncidentSpec,
    RankedHypothesis,
    RunMetadata,
    UnitRecord,
)

METHOD_ID = "serving-regression-rca/1.0.0"

_EVIDENCE_WEIGHTS = {
    EvidenceKind.DIRECT_EVENT: 5,
    EvidenceKind.INTERVENTION: 4,
    EvidenceKind.MEASURED_METRIC: 3,
    EvidenceKind.CONFIGURATION: 2,
    EvidenceKind.CORRELATION: 1,
    EvidenceKind.ARTIFACT: 1,
    EvidenceKind.ESTIMATE: 1,
    EvidenceKind.INPUT: 0,
    EvidenceKind.SYNTHETIC_FIXTURE: 0,
}
_CAUSAL_EVIDENCE = {
    EvidenceKind.DIRECT_EVENT,
    EvidenceKind.INTERVENTION,
    EvidenceKind.MEASURED_METRIC,
}


def _metric_value(observed: str) -> float | None:
    try:
        return float(observed)
    except ValueError:
        return None


def _has_capacity_timeline(
    hypothesis: HypothesisSpec,
    links: tuple[IncidentEvidence, ...],
) -> bool:
    if hypothesis.category != "kv_capacity":
        return True
    supporting = tuple(item for item in links if item.stance == "supporting")
    events = tuple(
        item
        for item in supporting
        if item.evidence.event_type in {"oom", "preemption"}
        and item.evidence.timestamp_ms is not None
    )
    for event in events:
        assert event.evidence.timestamp_ms is not None
        if any(
            item.evidence.kind is EvidenceKind.MEASURED_METRIC
            and "kv" in item.evidence.description.lower()
            and item.evidence.timestamp_ms is not None
            and item.evidence.timestamp_ms < event.evidence.timestamp_ms
            for item in supporting
        ):
            return True
    return False


def _causal_gate(
    hypothesis: HypothesisSpec,
    links: tuple[IncidentEvidence, ...],
    incident: IncidentSpec,
) -> bool:
    evidence_by_id = {item.evidence.evidence_id: item.evidence for item in links}
    interventions = tuple(
        item for item in incident.interventions if item.hypothesis_id == hypothesis.hypothesis_id
    )
    if len(interventions) != 1:
        return False
    intervention = interventions[0]
    baseline = evidence_by_id.get(intervention.baseline_evidence_id)
    rerun = evidence_by_id.get(intervention.rerun_evidence_id)
    if baseline is None or rerun is None:
        return False
    baseline_value = _metric_value(baseline.observed_value)
    rerun_value = _metric_value(rerun.observed_value)
    if baseline_value is None or rerun_value is None:
        return False
    measured_delta = rerun_value - baseline_value
    direction_matches = (
        measured_delta < 0
        if hypothesis.expected_metric_direction == "decrease"
        else measured_delta > 0
    )
    basic_gate = (
        bool(intervention.explicit_hypothesis.strip())
        and bool(intervention.changed_variable.strip())
        and intervention.rerun_id != intervention.baseline_run_id
        and intervention.baseline_workload_hash == intervention.rerun_workload_hash
        and intervention.baseline_config_hash != intervention.rerun_config_hash
        and direction_matches
        and abs(measured_delta) > 2 * intervention.dispersion
        and baseline.kind in _CAUSAL_EVIDENCE
        and rerun.kind in _CAUSAL_EVIDENCE
        and baseline.timestamp_ms is not None
        and rerun.timestamp_ms is not None
        and rerun.timestamp_ms > baseline.timestamp_ms
        and intervention.baseline_run_id == incident.baseline_run_id
    )
    return basic_gate and _has_capacity_timeline(hypothesis, links)


def diagnose_incident(run: RunMetadata, incident: IncidentSpec) -> DiagnosisAnalysis:
    """Rank evidence while reserving causal language for controlled reruns."""

    ranked: list[RankedHypothesis] = []
    for hypothesis in incident.hypotheses:
        links = tuple(
            item for item in incident.evidence if item.hypothesis_id == hypothesis.hypothesis_id
        )
        hypothesis_supporting = tuple(
            item.evidence.evidence_id for item in links if item.stance == "supporting"
        )
        hypothesis_contrary = tuple(
            item.evidence.evidence_id for item in links if item.stance == "contrary"
        )
        score = sum(
            _EVIDENCE_WEIGHTS[item.evidence.kind] * (1 if item.stance == "supporting" else -1)
            for item in links
        )
        causal_gate = _causal_gate(hypothesis, links, incident)
        hypothesis_status: Literal["confirmed", "correlated", "inconclusive", "unsupported"]
        if hypothesis_supporting and hypothesis_contrary:
            hypothesis_status = "inconclusive"
        elif causal_gate and hypothesis_supporting:
            hypothesis_status = "confirmed"
        elif hypothesis_supporting:
            hypothesis_status = "correlated"
        else:
            hypothesis_status = "unsupported"
        ranked.append(
            RankedHypothesis(
                hypothesis_id=hypothesis.hypothesis_id,
                category=hypothesis.category,
                score=score,
                status=hypothesis_status,
                supporting_evidence_ids=hypothesis_supporting,
                contrary_evidence_ids=hypothesis_contrary,
                causal_gate_passed=causal_gate,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.hypothesis_id))
    confirmed = tuple(item for item in ranked if item.status == "confirmed")
    top = ranked[0]

    available_telemetry = {item.evidence.description for item in incident.evidence}
    missing_telemetry = tuple(sorted(set(incident.required_telemetry) - available_telemetry))
    if len(confirmed) == 1:
        status = AnalysisStatus.SUPPORTED
        root_cause = confirmed[0].hypothesis_id
        statement = (
            f"Root cause {root_cause!r} is confirmed by a baseline, explicit hypothesis, "
            "single intervention, equivalent rerun, and measured delta."
        )
        confidence = Confidence.HIGH
        claim_supporting = confirmed[0].supporting_evidence_ids
        claim_contrary: tuple[str, ...] = ()
        next_experiment = None
    elif len(confirmed) > 1 or top.status == "inconclusive":
        status = AnalysisStatus.INCONCLUSIVE
        root_cause = None
        statement = (
            "The incident remains inconclusive because the evidence conflicts or is ambiguous."
        )
        confidence = Confidence.LOW
        claim_supporting = top.supporting_evidence_ids
        claim_contrary = top.contrary_evidence_ids
        next_experiment = (
            "Isolate the top hypotheses with one controlled intervention each under "
            "equivalent load."
        )
    else:
        status = AnalysisStatus.INSUFFICIENT_EVIDENCE
        root_cause = None
        statement = (
            f"Hypothesis {top.hypothesis_id!r} ranks first, but correlation does not establish "
            "root cause."
        )
        confidence = Confidence.LOW if top.supporting_evidence_ids else Confidence.NONE
        claim_supporting = top.supporting_evidence_ids
        claim_contrary = top.contrary_evidence_ids
        next_experiment = (
            f"Test {top.hypothesis_id!r}: record a baseline, change one variable, rerun the "
            "equivalent workload, and measure the predicted delta."
        )

    diagnosis = Diagnosis(
        status=status,
        root_cause_hypothesis_id=root_cause,
        ranked_hypotheses=tuple(ranked),
        missing_telemetry=missing_telemetry,
        next_discriminating_experiment=next_experiment,
    )
    evidence_by_id = {item.evidence.evidence_id: item.evidence for item in incident.evidence}
    evidence = tuple(evidence_by_id[item] for item in sorted(evidence_by_id))
    evidence_ids = tuple(item.evidence_id for item in evidence)
    claim = Claim(
        claim_id="incident.root_cause",
        claim_type="serving_regression_root_cause",
        statement=statement,
        status=status,
        evidence_ids=evidence_ids,
        supporting_evidence_ids=claim_supporting,
        contrary_evidence_ids=claim_contrary,
        method_id=METHOD_ID,
        units=(UnitRecord(field="hypothesis_score", unit="ordinal evidence points"),),
        assumptions=(
            "Incident evidence timestamps and metric scopes are aligned.",
            "An equivalent rerun changes exactly the declared intervention variable.",
        ),
        limitations=(
            "Evidence ranking prioritizes experiments; scores are not probabilities.",
            "Correlation-only telemetry cannot confirm a root cause.",
        ),
        confidence=confidence,
        missing_telemetry=missing_telemetry,
        next_discriminating_experiment=next_experiment,
    )
    return DiagnosisAnalysis(
        run=run,
        evidence=evidence,
        claims=(claim,),
        diagnosis=diagnosis,
    )
