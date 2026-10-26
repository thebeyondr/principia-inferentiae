"""Single deterministic command-line interface for all project skills."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import NoReturn

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
    load_document,
    load_model,
    load_trace_jsonl,
    write_json,
    write_jsonl,
    write_text,
    write_yaml,
)
from principia_inferentiae.practice.dag import ConceptDAG, GradingType, Status
from principia_inferentiae.practice.session import PracticeError
from principia_inferentiae.practice.session import graph_payload as practice_graph_payload
from principia_inferentiae.practice.session import ledger_text as practice_ledger_text
from principia_inferentiae.practice.session import serve_diagnose, serve_next, status_payload, submit_answer
from principia_inferentiae.practice.state import (
    AttemptRecord,
    Choice,
    ConceptProgress,
    ProbeRecord,
    StudentState,
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


practice_app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Adaptive practice over the concept graph.",
)
app.add_typer(practice_app, name="practice")

_DEFAULT_STATE = Path.home() / ".local" / "state" / "principia" / "practice.sqlite"


def _state_path(path: Path | None) -> Path:
    return _DEFAULT_STATE if path is None else path


def _parse_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise typer.BadParameter("--now must include a timezone offset")
    return parsed


def _concepts() -> ConceptDAG:
    directory = Path(__file__).resolve().parents[2] / "data" / "concepts"
    documents: list[object] = []
    for path in sorted(directory.glob("*.yaml")):
        loaded = load_document(path)
        if not isinstance(loaded, list):
            raise typer.BadParameter(f"{path.name} must be a list of concepts")
        documents.extend(loaded)
    return ConceptDAG.from_documents(documents)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS concept_progress (
            concept_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            mastery_score REAL NOT NULL,
            weakness REAL NOT NULL,
            repetitions INTEGER NOT NULL,
            easiness REAL NOT NULL,
            interval_days INTEGER NOT NULL,
            next_due_utc TEXT,
            last_seen_utc TEXT,
            streak INTEGER NOT NULL,
            updated_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts (
            attempt_id TEXT PRIMARY KEY,
            seq INTEGER NOT NULL,
            concept_id TEXT NOT NULL,
            seed INTEGER NOT NULL,
            difficulty REAL NOT NULL,
            prompt TEXT NOT NULL,
            canonical_answer TEXT NOT NULL,
            facts_json TEXT NOT NULL,
            grading_type TEXT NOT NULL,
            choices_json TEXT,
            unit TEXT,
            tolerance REAL NOT NULL,
            order_threshold REAL NOT NULL,
            label TEXT NOT NULL,
            domain TEXT NOT NULL,
            served_at_utc TEXT NOT NULL,
            answered_at_utc TEXT,
            response TEXT,
            correct INTEGER,
            cold INTEGER NOT NULL,
            seconds REAL,
            instance_key TEXT NOT NULL,
            origin TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        """
    )
    return connection


def _dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _load_state(path: Path) -> StudentState:
    if not path.exists():
        return StudentState()
    connection = _connect(path)
    try:
        progress: dict[str, ConceptProgress] = {}
        for row in connection.execute("SELECT * FROM concept_progress"):
            progress[str(row[0])] = ConceptProgress(
                concept_id=str(row[0]),
                status=Status(str(row[1])),
                mastery_score=float(row[2]),
                weakness=float(row[3]),
                repetitions=int(row[4]),
                easiness=float(row[5]),
                interval_days=int(row[6]),
                next_due=_dt(None if row[7] is None else str(row[7])),
                last_seen=_dt(None if row[8] is None else str(row[8])),
                streak=int(row[9]),
            )
        attempts: list[AttemptRecord] = []
        query = "SELECT * FROM attempts ORDER BY seq"
        for row in connection.execute(query):
            raw_choices = None if row["choices_json"] is None else json.loads(str(row["choices_json"]))
            choices = None
            if isinstance(raw_choices, list):
                choices = tuple(Choice.model_validate(item) for item in raw_choices)
            facts = json.loads(str(row["facts_json"]))
            if not isinstance(facts, dict):
                raise ValueError("attempt facts must be an object")
            correct = row["correct"]
            attempts.append(
                AttemptRecord(
                    attempt_id=str(row["attempt_id"]),
                    seq=int(row["seq"]),
                    concept_id=str(row["concept_id"]),
                    seed=int(row["seed"]),
                    difficulty=float(row["difficulty"]),
                    prompt=str(row["prompt"]),
                    canonical_answer=str(row["canonical_answer"]),
                    facts={str(key): value for key, value in facts.items()},
                    grading_type=GradingType(str(row["grading_type"])),
                    choices=choices,
                    unit=None if row["unit"] is None else str(row["unit"]),
                    tolerance=float(row["tolerance"]),
                    order_threshold=float(row["order_threshold"]),
                    label=str(row["label"]),
                    domain=str(row["domain"]),
                    served_at=_required_dt(str(row["served_at_utc"])),
                    answered_at=_dt(None if row["answered_at_utc"] is None else str(row["answered_at_utc"])),
                    response=None if row["response"] is None else str(row["response"]),
                    correct=None if correct is None else bool(correct),
                    cold=bool(row["cold"]),
                    seconds=None if row["seconds"] is None else float(row["seconds"]),
                    instance_key=str(row["instance_key"]),
                    origin=str(row["origin"]),
                )
            )
        payload = {"last_concept_id": None, "last_subdomain": None, "probes": []}
        cursor = connection.execute(
            "SELECT payload_json FROM sessions WHERE session_id = ?",
            ("learner",),
        ).fetchone()
        if cursor is not None:
            loaded = json.loads(str(cursor[0]))
            if isinstance(loaded, dict):
                payload = loaded
        probes: list[ProbeRecord] = []
        raw_probes = payload.get("probes", [])
        if isinstance(raw_probes, list):
            probes.extend(ProbeRecord.model_validate(item) for item in raw_probes)
        last_concept = payload.get("last_concept_id")
        last_subdomain = payload.get("last_subdomain")
        return StudentState(
            progress=progress,
            attempts=tuple(attempts),
            probes=tuple(probes),
            last_concept_id=None if last_concept is None else str(last_concept),
            last_subdomain=None if last_subdomain is None else str(last_subdomain),
        )
    finally:
        connection.close()


def _required_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("stored timestamps must include a timezone")
    return parsed


def _save_state(path: Path, state: StudentState, now: datetime) -> None:
    connection = _connect(path)
    try:
        with connection:
            connection.execute("DELETE FROM concept_progress")
            connection.execute("DELETE FROM attempts")
            connection.execute("DELETE FROM sessions")
            for progress in state.progress.values():
                connection.execute(
                    """
                    INSERT INTO concept_progress (
                        concept_id, status, mastery_score, weakness, repetitions, easiness,
                        interval_days, next_due_utc, last_seen_utc, streak, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        progress.concept_id,
                        progress.status.value,
                        progress.mastery_score,
                        progress.weakness,
                        progress.repetitions,
                        progress.easiness,
                        progress.interval_days,
                        None if progress.next_due is None else progress.next_due.isoformat(),
                        None if progress.last_seen is None else progress.last_seen.isoformat(),
                        progress.streak,
                        now.isoformat(),
                    ),
                )
            for attempt in state.attempts:
                choices_json = None
                if attempt.choices is not None:
                    choices_json = json.dumps(
                        [{"id": choice.id, "label": choice.label} for choice in attempt.choices]
                    )
                connection.execute(
                    """
                    INSERT INTO attempts (
                        attempt_id, seq, concept_id, seed, difficulty, prompt, canonical_answer,
                        facts_json, grading_type, choices_json, unit, tolerance, order_threshold,
                        label, domain, served_at_utc, answered_at_utc, response, correct, cold,
                        seconds, instance_key, origin
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt.attempt_id,
                        attempt.seq,
                        attempt.concept_id,
                        attempt.seed,
                        attempt.difficulty,
                        attempt.prompt,
                        attempt.canonical_answer,
                        json.dumps(attempt.facts),
                        attempt.grading_type.value,
                        choices_json,
                        attempt.unit,
                        attempt.tolerance,
                        attempt.order_threshold,
                        attempt.label,
                        attempt.domain,
                        attempt.served_at.isoformat(),
                        None if attempt.answered_at is None else attempt.answered_at.isoformat(),
                        attempt.response,
                        None if attempt.correct is None else int(attempt.correct),
                        int(attempt.cold),
                        attempt.seconds,
                        attempt.instance_key,
                        attempt.origin,
                    ),
                )
            connection.execute(
                """
                INSERT INTO sessions (session_id, started_at_utc, kind, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    "learner",
                    now.isoformat(),
                    "practice",
                    json.dumps(
                        {
                            "last_concept_id": state.last_concept_id,
                            "last_subdomain": state.last_subdomain,
                            "probes": [probe.model_dump() for probe in state.probes],
                        }
                    ),
                ),
            )
    finally:
        connection.close()


def _emit(payload: dict[str, object], as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        return
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        typer.echo(prompt)
        return
    typer.echo(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def _fail(error: PracticeError) -> NoReturn:
    typer.echo(str(error), err=True)
    raise typer.Exit(code=1)


@practice_app.command("next")
def practice_next(
    state: Path | None = typer.Option(None, "--state"),
    now: str | None = typer.Option(None, "--now"),
    seed: int = typer.Option(0, "--seed"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    path = _state_path(state)
    moment = _parse_now(now)
    try:
        updated, payload = serve_next(_load_state(path), _concepts(), moment, seed)
    except ModuleNotFoundError as error:
        if error.name != "principia_inferentiae.sim":
            raise
        _fail(PracticeError("principia_inferentiae.sim is required for practice generators"))
    except PracticeError as error:
        _fail(error)
    _save_state(path, updated, moment)
    _emit(payload, json_output)


@practice_app.command("diagnose")
def practice_diagnose(
    state: Path | None = typer.Option(None, "--state"),
    now: str | None = typer.Option(None, "--now"),
    seed: int = typer.Option(0, "--seed"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    path = _state_path(state)
    moment = _parse_now(now)
    try:
        updated, payload = serve_diagnose(_load_state(path), _concepts(), moment, seed)
    except ModuleNotFoundError as error:
        if error.name != "principia_inferentiae.sim":
            raise
        _fail(PracticeError("principia_inferentiae.sim is required for practice generators"))
    except PracticeError as error:
        _fail(error)
    _save_state(path, updated, moment)
    _emit(payload, json_output)


@practice_app.command("answer")
def practice_answer(
    attempt_id: str = typer.Argument(...),
    response: str = typer.Argument(...),
    state: Path | None = typer.Option(None, "--state"),
    now: str | None = typer.Option(None, "--now"),
    cold: bool = typer.Option(False, "--cold"),
    seconds: float | None = typer.Option(None, "--seconds"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    path = _state_path(state)
    moment = _parse_now(now)
    try:
        updated, payload = submit_answer(
            _load_state(path),
            _concepts(),
            moment,
            attempt_id,
            response,
            cold=cold,
            seconds=seconds,
        )
    except PracticeError as error:
        _fail(error)
    _save_state(path, updated, moment)
    _emit(payload, json_output)


@practice_app.command("status")
def practice_status(
    state: Path | None = typer.Option(None, "--state"),
    now: str | None = typer.Option(None, "--now"),
    domain: str | None = typer.Option(None, "--domain"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    payload = status_payload(
        _concepts(),
        _load_state(_state_path(state)),
        _parse_now(now),
        domain=domain,
    )
    _emit(payload, json_output)


@practice_app.command("ledger")
def practice_ledger(
    state: Path | None = typer.Option(None, "--state"),
    now: str | None = typer.Option(None, "--now"),
) -> None:
    typer.echo(
        practice_ledger_text(
            _concepts(),
            _load_state(_state_path(state)),
            _parse_now(now),
        ).rstrip("\n")
    )


@practice_app.command("export-graph")
def practice_export_graph(
    output: Path = typer.Option(..., "--output"),
) -> None:
    write_json(output, practice_graph_payload(_concepts()))


if __name__ == "__main__":
    app()
