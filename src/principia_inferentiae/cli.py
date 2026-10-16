"""Single deterministic command-line interface for all project skills."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import typer

from principia_inferentiae.contracts import (
    Claim,
    DiagnosisAnalysis,
    EngineeringAnalysis,
    EvidenceRecord,
    HardwareCatalog,
    HardwareSpec,
    IncidentSpec,
    MemoryAnalysis,
    ModelCatalog,
    ModelSpec,
    QualityGate,
    RooflineAnalysis,
    RooflineMeasurement,
    RunMetadata,
    RuntimeCatalog,
    ServiceCurve,
    ServingConfig,
    SizingAnalysis,
    SizingPolicy,
    SLOSpec,
    TraceRequest,
    WorkloadAnalysis,
    WorkloadProfile,
)
from principia_inferentiae.diagnosis import diagnose_incident
from principia_inferentiae.io import (
    load_model,
    load_trace_jsonl,
    write_json,
    write_jsonl,
    write_text,
    write_yaml,
)
from principia_inferentiae.memory import estimate_memory
from principia_inferentiae.pipeline import analyze_serving_candidate
from principia_inferentiae.reporting import (
    render_engineering_report,
    render_stakeholder_brief,
)
from principia_inferentiae.roofline import analyze_roofline
from principia_inferentiae.sizing import size_deployment
from principia_inferentiae.sources import (
    SourceCatalog,
    SourceEntry,
    ingest_source,
)
from principia_inferentiae.workload import (
    characterize_workload,
    prepare_vllm_timed_trace,
)

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Evidence-first LLM inference engineering.",
)


class Audience(StrEnum):
    ENGINEERING = "engineering"
    STAKEHOLDER = "stakeholder"


@app.command("characterize-workload")
def characterize_workload_command(
    run_path: Path = typer.Option(..., "--run", exists=True, dir_okay=False),
    trace_path: Path = typer.Option(..., "--trace", exists=True, dir_okay=False),
    slo_path: Path = typer.Option(..., "--slo", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
) -> None:
    run = load_model(run_path, RunMetadata)
    trace = load_trace_jsonl(trace_path)
    slo = load_model(slo_path, SLOSpec)
    write_json(output_path, characterize_workload(run, trace, slo))


@app.command("prepare-vllm-trace")
def prepare_vllm_trace_command(
    trace_path: Path = typer.Option(..., "--trace", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
    chunk_hash_size: int = typer.Option(512, "--chunk-hash-size", min=1),
) -> None:
    trace = load_trace_jsonl(trace_path)
    write_jsonl(
        output_path,
        prepare_vllm_timed_trace(trace, chunk_hash_size),
    )


@app.command("estimate-memory")
def estimate_memory_command(
    run_path: Path = typer.Option(..., "--run", exists=True, dir_okay=False),
    model_path: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    hardware_path: Path = typer.Option(..., "--hardware", exists=True, dir_okay=False),
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
) -> None:
    run = load_model(run_path, RunMetadata)
    model = load_model(model_path, ModelSpec)
    hardware = load_model(hardware_path, HardwareSpec)
    config = load_model(config_path, ServingConfig)
    write_json(output_path, estimate_memory(run, model, hardware, config))


@app.command("analyze-roofline")
def analyze_roofline_command(
    run_path: Path = typer.Option(..., "--run", exists=True, dir_okay=False),
    measurement_path: Path = typer.Option(..., "--measurement", exists=True, dir_okay=False),
    hardware_path: Path = typer.Option(..., "--hardware", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
) -> None:
    run = load_model(run_path, RunMetadata)
    measurement = load_model(measurement_path, RooflineMeasurement)
    hardware = load_model(hardware_path, HardwareSpec)
    write_json(output_path, analyze_roofline(run, measurement, hardware))


@app.command("size-deployment")
def size_deployment_command(
    run_path: Path = typer.Option(..., "--run", exists=True, dir_okay=False),
    workload_path: Path = typer.Option(..., "--workload", exists=True, dir_okay=False),
    memory_path: Path = typer.Option(..., "--memory", exists=True, dir_okay=False),
    model_path: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    hardware_path: Path = typer.Option(..., "--hardware", exists=True, dir_okay=False),
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    quality_gate_path: Path = typer.Option(..., "--quality-gate", exists=True, dir_okay=False),
    policy_path: Path = typer.Option(..., "--policy", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
    service_curve_path: Path | None = typer.Option(
        None,
        "--service-curve",
        exists=True,
        dir_okay=False,
    ),
) -> None:
    run = load_model(run_path, RunMetadata)
    workload = load_model(workload_path, WorkloadAnalysis)
    memory = load_model(memory_path, MemoryAnalysis)
    model = load_model(model_path, ModelSpec)
    hardware = load_model(hardware_path, HardwareSpec)
    config = load_model(config_path, ServingConfig)
    quality_gate = load_model(quality_gate_path, QualityGate)
    policy = load_model(policy_path, SizingPolicy)
    service_curve = (
        None if service_curve_path is None else load_model(service_curve_path, ServiceCurve)
    )
    write_json(
        output_path,
        size_deployment(
            run,
            workload.profile,
            memory.estimate,
            model,
            hardware,
            config,
            quality_gate,
            policy,
            service_curve,
        ),
    )


@app.command("diagnose-regression")
def diagnose_regression_command(
    run_path: Path = typer.Option(..., "--run", exists=True, dir_okay=False),
    incident_path: Path = typer.Option(..., "--incident", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
) -> None:
    run = load_model(run_path, RunMetadata)
    incident = load_model(incident_path, IncidentSpec)
    write_json(output_path, diagnose_incident(run, incident))


@app.command("analyze")
def analyze_command(
    run_path: Path = typer.Option(..., "--run", exists=True, dir_okay=False),
    trace_path: Path = typer.Option(..., "--trace", exists=True, dir_okay=False),
    slo_path: Path = typer.Option(..., "--slo", exists=True, dir_okay=False),
    model_path: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    hardware_path: Path = typer.Option(..., "--hardware", exists=True, dir_okay=False),
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    measurement_path: Path = typer.Option(..., "--measurement", exists=True, dir_okay=False),
    quality_gate_path: Path = typer.Option(..., "--quality-gate", exists=True, dir_okay=False),
    policy_path: Path = typer.Option(..., "--policy", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
    service_curve_path: Path | None = typer.Option(
        None,
        "--service-curve",
        exists=True,
        dir_okay=False,
    ),
) -> None:
    service_curve = (
        None if service_curve_path is None else load_model(service_curve_path, ServiceCurve)
    )
    analysis = analyze_serving_candidate(
        run=load_model(run_path, RunMetadata),
        trace=load_trace_jsonl(trace_path),
        slo=load_model(slo_path, SLOSpec),
        model=load_model(model_path, ModelSpec),
        hardware=load_model(hardware_path, HardwareSpec),
        config=load_model(config_path, ServingConfig),
        roofline_measurement=load_model(measurement_path, RooflineMeasurement),
        quality_gate=load_model(quality_gate_path, QualityGate),
        sizing_policy=load_model(policy_path, SizingPolicy),
        service_curve=service_curve,
    )
    write_json(output_path, analysis)


@app.command("write-brief")
def write_brief_command(
    analysis_path: Path = typer.Option(..., "--analysis", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
    audience: Audience = typer.Option(
        Audience.STAKEHOLDER,
        "--audience",
    ),
) -> None:
    analysis = load_model(analysis_path, EngineeringAnalysis)
    report = (
        render_engineering_report(analysis)
        if audience is Audience.ENGINEERING
        else render_stakeholder_brief(analysis)
    )
    write_text(output_path, report)


@app.command("ingest-source")
def ingest_source_command(
    catalog_path: Path = typer.Option(..., "--catalog", exists=True, dir_okay=False),
    entry_path: Path = typer.Option(..., "--entry", exists=True, dir_okay=False),
    catalog_output_path: Path = typer.Option(..., "--catalog-output", dir_okay=False),
    result_output_path: Path = typer.Option(..., "--result-output", dir_okay=False),
) -> None:
    result = ingest_source(
        load_model(catalog_path, SourceCatalog),
        load_model(entry_path, SourceEntry),
    )
    write_yaml(catalog_output_path, result.catalog)
    write_json(result_output_path, result)


@app.command("export-schemas")
def export_schemas_command(
    output_directory: Path = typer.Option(..., "--output-directory", file_okay=False),
) -> None:
    schema_models = (
        RunMetadata,
        ModelCatalog,
        HardwareCatalog,
        RuntimeCatalog,
        SourceCatalog,
        SourceEntry,
        Claim,
        EvidenceRecord,
        ModelSpec,
        HardwareSpec,
        TraceRequest,
        WorkloadProfile,
        SLOSpec,
        ServingConfig,
        QualityGate,
        SizingPolicy,
        ServiceCurve,
        RooflineMeasurement,
        IncidentSpec,
        WorkloadAnalysis,
        MemoryAnalysis,
        RooflineAnalysis,
        SizingAnalysis,
        DiagnosisAnalysis,
        EngineeringAnalysis,
    )
    for schema_model in schema_models:
        filename = f"{schema_model.__name__.replace('Analysis', '-analysis').lower()}.schema.json"
        write_json(output_directory / filename, schema_model.model_json_schema())


if __name__ == "__main__":
    app()
