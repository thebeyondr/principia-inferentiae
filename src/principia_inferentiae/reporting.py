"""Deterministic Markdown views over validated analysis JSON."""

from __future__ import annotations

from principia_inferentiae.contracts import EngineeringAnalysis, EvidenceEnvelope


def render_stakeholder_brief(analysis: EvidenceEnvelope) -> str:
    """Render only validated claims; never infer or soften uncertainty."""

    lines = [
        "# Inference decision brief",
        "",
        f"- Run: `{analysis.run.run_id}`",
        f"- Evidence tier: `{analysis.run.reproducibility_tier}`",
        f"- Method revision: `{analysis.run.method_revision}`",
        "",
        "## Decisions",
    ]
    for claim in sorted(analysis.claims, key=lambda item: item.claim_id):
        lines.extend(
            [
                f"- **{claim.status.value} — {claim.claim_type}:** {claim.statement}",
                f"  Confidence: `{claim.confidence.value}`.",
            ]
        )
        if claim.missing_telemetry:
            lines.append(f"  Missing: {'; '.join(claim.missing_telemetry)}.")
        if claim.next_discriminating_experiment is not None:
            lines.append(f"  Next: {claim.next_discriminating_experiment}")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "Null recommendations and inconclusive findings are intentional fail-closed outcomes.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_engineering_report(analysis: EngineeringAnalysis) -> str:
    """Render a detailed, claim-traceable engineering view."""

    recommended_gpu_count = (
        "null"
        if analysis.sizing.recommended_gpu_count is None
        else str(analysis.sizing.recommended_gpu_count)
    )
    load_points = ", ".join(str(item) for item in analysis.benchmark_plan.load_points_rps)
    lines = [
        "# Engineering analysis",
        "",
        "## Run contract",
        "",
        f"- Run ID: `{analysis.run.run_id}`",
        f"- Input time: `{analysis.run.created_at_utc.isoformat()}`",
        f"- Code revision: `{analysis.run.code_revision}`",
        f"- Reproducibility tier: `{analysis.run.reproducibility_tier}`",
        "",
        "## Workload",
        "",
        f"- Workload hash: `{analysis.profile.workload_hash}`",
        f"- Requests: {analysis.profile.request_count}",
        f"- Prompt tokens p50/p95/p99: {analysis.profile.prompt_tokens.p50:.0f} / "
        f"{analysis.profile.prompt_tokens.p95:.0f} / {analysis.profile.prompt_tokens.p99:.0f}",
        f"- Output tokens p50/p95/p99: {analysis.profile.output_tokens.p50:.0f} / "
        f"{analysis.profile.output_tokens.p95:.0f} / {analysis.profile.output_tokens.p99:.0f}",
        f"- Peak one-second arrival rate: "
        f"{analysis.profile.peak_one_second_arrival_rps:.3f} requests/s",
        "",
        "## Capacity and performance",
        "",
        f"- Per-rank memory fit: `{analysis.memory.fits_per_rank}`",
        f"- Required / usable bytes per rank: {analysis.memory.required_per_rank_bytes} / "
        f"{analysis.memory.usable_device_bytes}",
        f"- Roofline class: `{analysis.roofline.classification.value}`",
        "- Memory-fit lower bound: "
        + (
            "null"
            if analysis.sizing.memory_fit_lower_bound_gpus is None
            else f"{analysis.sizing.memory_fit_lower_bound_gpus} GPU(s)"
        ),
        f"- GPUs per configured replica: {analysis.sizing.gpus_per_replica}",
        f"- Recommended GPU count: {recommended_gpu_count}",
        "",
        "## Claims",
        "",
    ]
    for claim in sorted(analysis.claims, key=lambda item: item.claim_id):
        lines.extend(
            [
                f"### `{claim.claim_id}`",
                "",
                f"- Type / status / confidence: `{claim.claim_type}` / `{claim.status.value}` / "
                f"`{claim.confidence.value}`",
                f"- Conclusion: {claim.statement}",
                f"- Method: `{claim.method_id}`",
                f"- Evidence: {', '.join(f'`{item}`' for item in claim.evidence_ids)}",
                f"- Assumptions: {'; '.join(claim.assumptions) or 'none'}",
                f"- Limitations: {'; '.join(claim.limitations) or 'none'}",
                f"- Missing telemetry: {'; '.join(claim.missing_telemetry) or 'none'}",
                f"- Next experiment: {claim.next_discriminating_experiment or 'none'}",
                "",
            ]
        )
    lines.extend(
        [
            "## Benchmark plan",
            "",
            f"- Status: `{analysis.benchmark_plan.status}`",
            f"- Workload hash: `{analysis.benchmark_plan.required_workload_hash}`",
            f"- Config hash: `{analysis.benchmark_plan.required_config_hash}`",
            f"- Load points: {load_points} requests/s",
            f"- Command template: `{analysis.benchmark_plan.command_template}`",
        ]
    )
    return "\n".join(lines) + "\n"
