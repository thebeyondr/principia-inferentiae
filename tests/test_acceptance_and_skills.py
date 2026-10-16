from __future__ import annotations

import re
from pathlib import Path

import yaml
from typer.testing import CliRunner

from principia_inferentiae.cli import app
from principia_inferentiae.contracts import (
    AnalysisStatus,
    Digest,
    HardwareCatalog,
    HardwareSpec,
    ModelCatalog,
    ModelSpec,
    QualityGate,
    RooflineMeasurement,
    RunMetadata,
    RuntimeCatalog,
    ServingConfig,
    SizingPolicy,
    SLOSpec,
    TraceRequest,
)
from principia_inferentiae.io import canonical_json_bytes, load_model
from principia_inferentiae.pipeline import analyze_serving_candidate
from principia_inferentiae.reporting import (
    render_engineering_report,
    render_stakeholder_brief,
)
from principia_inferentiae.sources import (
    SourceCatalog,
    SourceEntry,
    SourceFact,
    ingest_source,
)

ROOT = Path(__file__).parents[1]
SKILL_NAMES = (
    "ingest-inference-source",
    "characterize-serving-workload",
    "analyze-memory-vs-compute",
    "size-model-deployment",
    "diagnose-serving-regression",
    "write-inference-brief",
)


def _planned_measurement(hardware: HardwareSpec) -> RooflineMeasurement:
    return RooflineMeasurement(
        measurement_id="planned-profile",
        measurement_status="planned",
        scope_kind="phase",
        scope_name="decode",
        phase="decode_step",
        hardware_id=hardware.hardware_id,
        hardware_revision=hardware.revision,
        precision="bf16",
        provenance=hardware.provenance,
    )


def test_first_vertical_slice_fails_closed_and_is_deterministic(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    quality_gate: QualityGate,
    policy: SizingPolicy,
) -> None:
    first = analyze_serving_candidate(
        run,
        trace,
        slo,
        model,
        hardware,
        config,
        _planned_measurement(hardware),
        quality_gate,
        policy,
    )
    second = analyze_serving_candidate(
        run,
        tuple(reversed(trace)),
        slo,
        model,
        hardware,
        config,
        _planned_measurement(hardware),
        quality_gate,
        policy,
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first.sizing.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert first.sizing.recommended_gpu_count is None
    assert first.roofline.status is AnalysisStatus.INSUFFICIENT_EVIDENCE

    evidence_ids = {item.evidence_id for item in first.evidence}
    for claim in first.claims:
        assert set(claim.evidence_ids) <= evidence_ids
        assert claim.method_id
        assert claim.units
        assert claim.assumptions
        assert claim.limitations


def test_golden_reports_contain_exact_validated_claims(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    quality_gate: QualityGate,
    policy: SizingPolicy,
) -> None:
    analysis = analyze_serving_candidate(
        run,
        trace,
        slo,
        model,
        hardware,
        config,
        _planned_measurement(hardware),
        quality_gate,
        policy,
    )
    engineering = render_engineering_report(analysis)
    stakeholder = render_stakeholder_brief(analysis)
    assert engineering == render_engineering_report(analysis)
    assert stakeholder == render_stakeholder_brief(analysis)
    for claim in analysis.claims:
        assert claim.statement in engineering
        assert claim.statement in stakeholder
    assert "Recommended GPU count: null" in engineering


def test_cli_exposes_every_skill_contract() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "ingest-source",
        "characterize-workload",
        "prepare-vllm-trace",
        "analyze-roofline",
        "estimate-memory",
        "size-deployment",
        "diagnose-regression",
        "write-brief",
    ):
        assert command in result.stdout


def test_skill_metadata_matches_open_agent_skills_basics() -> None:
    for skill_name in SKILL_NAMES:
        path = ROOT / ".agents" / "skills" / skill_name / "SKILL.md"
        text = path.read_text(encoding="utf-8")
        opening, metadata_text, body = text.split("---", maxsplit=2)
        assert opening == ""
        metadata = yaml.safe_load(metadata_text)
        assert metadata["name"] == skill_name
        assert isinstance(metadata["description"], str)
        assert len(metadata["description"]) <= 1024
        assert metadata["license"] == "Apache-2.0"
        assert "use when" in metadata["description"].lower()
        assert "principia" in body


def test_versioned_catalogs_validate_with_typed_entries() -> None:
    model_catalog = load_model(ROOT / "data/catalogs/models.yaml", ModelCatalog)
    hardware_catalog = load_model(ROOT / "data/catalogs/hardware.yaml", HardwareCatalog)
    runtime_catalog = load_model(ROOT / "data/catalogs/runtimes.yaml", RuntimeCatalog)
    source_catalog = load_model(ROOT / "sources/catalog.yaml", SourceCatalog)
    assert model_catalog.entries
    assert hardware_catalog.entries
    assert runtime_catalog.entries
    assert source_catalog.entries
    assert all(entry.provenance.evidence_class == "synthetic" for entry in model_catalog.entries)


def test_focused_knowledge_pages_have_retrieval_and_cataloged_citations() -> None:
    source_catalog = load_model(ROOT / "sources/catalog.yaml", SourceCatalog)
    catalog_urls = {entry.url for entry in source_catalog.entries}
    focused_pages = tuple(
        path
        for path in sorted((ROOT / "knowledge").glob("*.md"))
        if path.name not in {"index.md", "activity-log.md"}
    )
    assert len(focused_pages) == 8
    for page in focused_pages:
        text = page.read_text(encoding="utf-8")
        assert "## Theory" in text
        assert "## What must be measured" in text
        assert "## Failure modes" in text
        assert "## Retrieval checks" in text
        assert "## Sources" in text
        urls = set(re.findall(r"\]\((https://[^)]+)\)", text))
        assert urls
        assert urls <= catalog_urls


def test_source_ingestion_preserves_scoped_contradictions() -> None:
    first = SourceEntry(
        source_id="first",
        title="First",
        url="https://example.com/first",
        authors=("A",),
        publication_or_version_date="1",
        source_type="fixture",
        license_or_usage_note="fixture",
        topics=("roofline",),
        last_verified_date="2026-01-01",
        content_digest=Digest(algorithm="sha256", value="1" * 64),
        content_digest_semantics="fixture",
        facts=(
            SourceFact(
                fact_key="hbm_ceiling",
                value="100",
                unit="GB/s",
                scope="fixture-device",
            ),
        ),
    )
    second = first.model_copy(
        update={
            "source_id": "second",
            "url": "https://example.com/second",
            "content_digest": Digest(algorithm="sha256", value="2" * 64),
            "facts": (
                SourceFact(
                    fact_key="hbm_ceiling",
                    value="120",
                    unit="GB/s",
                    scope="fixture-device",
                ),
            ),
        }
    )
    result = ingest_source(
        SourceCatalog(schema_version="1.0.0", entries=(first,)),
        second,
    )
    assert tuple(item.source_id for item in result.catalog.entries) == ("first", "second")
    assert len(result.contradictions) == 1
    assert result.contradictions[0].existing_value == "100"
    assert result.contradictions[0].new_value == "120"
