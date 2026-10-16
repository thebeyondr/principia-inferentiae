"""Workload characterization from request traces and explicit SLOs."""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from principia_inferentiae.contracts import (
    AnalysisStatus,
    Claim,
    Confidence,
    EvidenceKind,
    EvidenceRecord,
    Percentiles,
    RunMetadata,
    SLOSpec,
    TraceRequest,
    UnitRecord,
    VllmTimedTraceRecord,
    WorkloadAnalysis,
    WorkloadProfile,
)
from principia_inferentiae.io import digest_value

METHOD_ID = "workload-characterization/1.0.0"


def _nearest_rank(values: tuple[int, ...], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _percentiles(values: tuple[int, ...]) -> Percentiles:
    return Percentiles(
        p50=_nearest_rank(values, 0.50),
        p95=_nearest_rank(values, 0.95),
        p99=_nearest_rank(values, 0.99),
    )


def workload_hash_payload(profile: WorkloadProfile) -> dict[str, object]:
    payload = profile.model_dump(mode="json")
    payload["workload_hash"] = ""
    return payload


def assert_workload_hash(profile: WorkloadProfile) -> None:
    if digest_value(workload_hash_payload(profile)).value != profile.workload_hash:
        raise ValueError("workload profile statistics do not match workload_hash")


def _sliding_peak_one_second(times: tuple[int, ...]) -> int:
    ordered = tuple(sorted(times))
    if not ordered:
        return 0
    start = 0
    peak = 0
    for end, timestamp in enumerate(ordered):
        while timestamp - ordered[start] >= 1000:
            start += 1
        peak = max(peak, end - start + 1)
    return peak


def _peak_concurrency(records: tuple[TraceRequest, ...]) -> int | None:
    if any(record.completed_time_ms is None for record in records):
        return None
    events: list[tuple[int, int]] = []
    for record in records:
        assert record.completed_time_ms is not None
        events.append((record.arrival_time_ms, 1))
        events.append((record.completed_time_ms, -1))
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        peak = max(peak, active)
    return peak


def prepare_vllm_timed_trace(
    records: tuple[TraceRequest, ...],
    chunk_hash_size: int,
) -> tuple[VllmTimedTraceRecord, ...]:
    """Create deterministic vLLM timed-trace records with explicit prefix chunks."""

    if chunk_hash_size <= 0:
        raise ValueError("chunk_hash_size must be positive")
    output: list[VllmTimedTraceRecord] = []
    origin = min(record.arrival_time_ms for record in records)
    for record in sorted(records, key=lambda item: (item.arrival_time_ms, item.request_id)):
        chunk_count = math.ceil(record.prompt_tokens / chunk_hash_size)
        shared_chunks = record.shared_prefix_tokens // chunk_hash_size
        hash_ids: list[int] = []
        for chunk_index in range(chunk_count):
            if record.prefix_key is not None and chunk_index < shared_chunks:
                identity = f"shared:{record.prefix_key}:{chunk_index}"
            else:
                identity = f"request:{record.request_id}:{chunk_index}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            hash_ids.append(int(digest[:15], 16))
        output.append(
            VllmTimedTraceRecord(
                timestamp=record.arrival_time_ms - origin,
                input_length=record.prompt_tokens,
                output_length=record.output_tokens,
                hash_ids=tuple(hash_ids),
            )
        )
    return tuple(output)


def characterize_workload(
    run: RunMetadata,
    records: tuple[TraceRequest, ...],
    slo: SLOSpec,
) -> WorkloadAnalysis:
    """Build an order-invariant workload profile."""

    ordered = tuple(sorted(records, key=lambda item: (item.arrival_time_ms, item.request_id)))
    prompt_tokens = tuple(item.prompt_tokens for item in ordered)
    output_tokens = tuple(item.output_tokens for item in ordered)
    total_tokens = tuple(item.prompt_tokens + item.output_tokens for item in ordered)
    first_arrival = min(item.arrival_time_ms for item in ordered)
    last_arrival = max(item.arrival_time_ms for item in ordered)
    observation_window_ms = max(1, last_arrival - first_arrival + 1)
    mean_arrival_rps = len(ordered) / (observation_window_ms / 1000)

    peak_one_second_arrival_rps = float(
        _sliding_peak_one_second(tuple(item.arrival_time_ms for item in ordered))
    )
    if observation_window_ms < 1000 or peak_one_second_arrival_rps + 1e-9 < mean_arrival_rps:
        burstiness = 1.0
    else:
        burstiness = peak_one_second_arrival_rps / mean_arrival_rps

    prefix_counts = Counter(item.prefix_key for item in ordered if item.prefix_key is not None)
    reused_requests = sum(max(0, count - 1) for count in prefix_counts.values())
    prefix_reuse_fraction = reused_requests / len(ordered)
    shared_prefix_tokens = tuple(item.shared_prefix_tokens for item in ordered)

    trace_digest = digest_value(ordered)
    slo_digest = digest_value(slo)
    peak_concurrency = _peak_concurrency(ordered)
    missing_items: list[str] = []
    if observation_window_ms < 1000:
        missing_items.append("observation window of at least one second")
    elif peak_one_second_arrival_rps + 1e-9 < mean_arrival_rps:
        missing_items.append("peak one-second rate below mean arrival rate")
    if peak_concurrency is None:
        missing_items.append("request completion timestamps")
    if any(item.prefix_key is not None and item.shared_prefix_tokens == 0 for item in ordered):
        missing_items.append("shared prefix token lengths")
    missing = tuple(missing_items)

    profile = WorkloadProfile(
        workload_hash="0" * 64,
        trace_digest=trace_digest,
        request_count=len(ordered),
        observation_window_ms=observation_window_ms,
        prompt_tokens=_percentiles(prompt_tokens),
        output_tokens=_percentiles(output_tokens),
        total_tokens=_percentiles(total_tokens),
        mean_arrival_rps=round(mean_arrival_rps, 6),
        peak_one_second_arrival_rps=round(peak_one_second_arrival_rps, 6),
        burstiness_peak_to_mean=round(burstiness, 6),
        peak_observed_concurrency=peak_concurrency,
        prefix_reuse_fraction=round(prefix_reuse_fraction, 6),
        shared_prefix_tokens=_percentiles(shared_prefix_tokens),
        tenant_count=len({item.tenant_id for item in ordered}),
        slo=slo,
        missing_telemetry=missing,
    )
    profile = profile.model_copy(
        update={"workload_hash": digest_value(workload_hash_payload(profile)).value}
    )
    trace_evidence = EvidenceRecord(
        evidence_id="workload.trace",
        kind=EvidenceKind.INPUT,
        description="Canonicalized request trace",
        observed_value=trace_digest.value,
        unit="sha256",
        source_digest=trace_digest,
    )
    slo_evidence = EvidenceRecord(
        evidence_id="workload.slo",
        kind=EvidenceKind.CONFIGURATION,
        description="Validated serving SLO contract",
        observed_value=slo_digest.value,
        unit="sha256",
        source_digest=slo_digest,
    )
    claim = Claim(
        claim_id="workload.characterization",
        claim_type="workload_characterization",
        statement=(
            f"Observed {len(ordered)} requests; prompt p99={profile.prompt_tokens.p99:.0f} "
            f"tokens, output p99={profile.output_tokens.p99:.0f} tokens, and peak "
            f"one-second arrival rate={profile.peak_one_second_arrival_rps:.3f} requests/s."
        ),
        status=AnalysisStatus.SUPPORTED,
        evidence_ids=(trace_evidence.evidence_id, slo_evidence.evidence_id),
        supporting_evidence_ids=(trace_evidence.evidence_id, slo_evidence.evidence_id),
        method_id=METHOD_ID,
        units=(
            UnitRecord(field="prompt_tokens.p99", unit="tokens"),
            UnitRecord(field="output_tokens.p99", unit="tokens"),
            UnitRecord(field="peak_one_second_arrival_rps", unit="requests/s"),
            UnitRecord(field="shared_prefix_tokens.p99", unit="tokens"),
        ),
        assumptions=(
            "Nearest-rank percentiles are computed over the supplied observation window.",
            "One-second arrival buckets are aligned to the first observed arrival.",
        ),
        limitations=("The trace describes the supplied window, not future demand.",),
        confidence=Confidence.HIGH if not missing else Confidence.MEDIUM,
        missing_telemetry=missing,
        next_discriminating_experiment=(
            None
            if not missing
            else "Capture completion timestamps to measure observed concurrency."
        ),
    )
    return WorkloadAnalysis(
        run=run,
        evidence=(trace_evidence, slo_evidence),
        claims=(claim,),
        profile=profile,
    )
