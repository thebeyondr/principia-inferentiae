"""Serve, grade, and report. Every function takes ``now`` and ``state``."""

from __future__ import annotations

import hashlib
import importlib
from datetime import datetime

from principia_inferentiae.practice.dag import DOMAIN_LABELS, DOMAIN_ORDER, ConceptDAG, Status
from principia_inferentiae.practice.diagnostic import is_slow, note_probe, select_probe
from principia_inferentiae.practice.evidence import NEUTRAL_DIFFICULTY
from principia_inferentiae.practice.generators.registry import ConceptGenerator, check_generator
from principia_inferentiae.practice.grader import Problem, grade
from principia_inferentiae.practice.mastery import read_status, record_outcome
from principia_inferentiae.practice.scheduler import select
from principia_inferentiae.practice.state import AttemptRecord, StudentState


class PracticeError(ValueError):
    """Caller-facing practice failure (unknown attempt, double submit)."""


def _attempt_id(concept_id: str, seed: int, now: datetime, seq: int) -> str:
    payload = f"{concept_id}|{seed}|{now.isoformat()}|{seq}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _question_seed(seed: int, concept_id: str, seq: int) -> int:
    digest = hashlib.sha256(f"{seed}|{concept_id}|{seq}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _public_item(attempt: AttemptRecord) -> dict[str, object]:
    choices: object
    if attempt.choices is None:
        choices = None
    else:
        choices = [{"id": choice.id, "label": choice.label} for choice in attempt.choices]
    return {
        "attempt_id": attempt.attempt_id,
        "concept_id": attempt.concept_id,
        "label": attempt.label,
        "domain": attempt.domain,
        "grading_type": attempt.grading_type.value,
        "prompt": attempt.prompt,
        "choices": choices,
        "unit": attempt.unit,
        "cold_offered": True,
    }


_MODULE_FOR_PREFIX = {
    "units": "units",
    "model": "model_memory",
    "kv": "kv_cache",
    "phases": "phases",
    "metrics": "metrics",
    "workload": "workload",
    "roofline": "roofline",
    "sizing": "sizing",
    "diagnosis": "diagnosis",
    "policy": "policy",
}


def _generator_for(concept_id: str) -> ConceptGenerator:
    # Load only this concept's module. Policy imports principia_inferentiae.sim.
    from principia_inferentiae.practice.generators.registry import GENERATORS

    prefix = concept_id.split(".", 1)[0]
    module_name = _MODULE_FOR_PREFIX.get(prefix)
    if module_name is None:
        raise PracticeError(f"no generator module for {concept_id}")
    importlib.import_module(f"principia_inferentiae.practice.generators.{module_name}")
    if concept_id not in GENERATORS:
        raise PracticeError(f"no generator registered for {concept_id}")
    check_generator(concept_id)
    return GENERATORS[concept_id]


def _serve(
    state: StudentState,
    dag: ConceptDAG,
    now: datetime,
    seed: int,
    *,
    origin: str,
    concept_id: str,
    difficulty: float,
) -> tuple[StudentState, dict[str, object]]:
    generator = _generator_for(concept_id)
    concept = dag.get(concept_id)
    question_seed = _question_seed(seed, concept_id, len(state.attempts))
    problem = generator.generate(question_seed, difficulty)
    attempt = AttemptRecord(
        attempt_id=_attempt_id(concept_id, seed, now, len(state.attempts)),
        seq=len(state.attempts),
        concept_id=concept_id,
        seed=question_seed,
        difficulty=difficulty,
        prompt=problem.prompt,
        canonical_answer=problem.answer,
        facts=problem.facts,
        grading_type=problem.grading_type,
        choices=problem.choices,
        unit=problem.unit,
        tolerance=problem.tolerance,
        order_threshold=problem.order_threshold,
        label=concept.label,
        domain=concept.domain,
        served_at=now,
        instance_key=problem.instance_key,
        origin=origin,
    )
    updated = state.with_attempt(attempt).model_copy(
        update={"last_concept_id": concept.id, "last_subdomain": concept.subdomain}
    )
    return updated, _public_item(attempt)


def serve_next(
    state: StudentState,
    dag: ConceptDAG,
    now: datetime,
    seed: int,
) -> tuple[StudentState, dict[str, object]]:
    selection = select(dag, state, now, seed)
    return _serve(
        state,
        dag,
        now,
        seed,
        origin="practice",
        concept_id=selection.concept_id,
        difficulty=selection.difficulty,
    )


def serve_diagnose(
    state: StudentState,
    dag: ConceptDAG,
    now: datetime,
    seed: int,
) -> tuple[StudentState, dict[str, object]]:
    concept_id = select_probe(dag, state, now, seed)
    return _serve(
        state,
        dag,
        now,
        seed,
        origin="diagnose",
        concept_id=concept_id,
        difficulty=NEUTRAL_DIFFICULTY,
    )


def _problem_from(attempt: AttemptRecord) -> Problem:
    return Problem(
        concept_id=attempt.concept_id,
        prompt=attempt.prompt,
        answer=attempt.canonical_answer,
        facts=attempt.facts,
        choices=attempt.choices,
        unit=attempt.unit,
        tolerance=attempt.tolerance,
        instance_key=attempt.instance_key,
        order_threshold=attempt.order_threshold,
        grading_type=attempt.grading_type,
    )


def submit_answer(
    state: StudentState,
    dag: ConceptDAG,
    now: datetime,
    attempt_id: str,
    response: str,
    *,
    cold: bool,
    seconds: float | None,
) -> tuple[StudentState, dict[str, object]]:
    if seconds is not None and seconds < 0:
        raise PracticeError("seconds must be >= 0")
    attempt = next((item for item in state.attempts if item.attempt_id == attempt_id), None)
    if attempt is None:
        raise PracticeError(f"unknown attempt {attempt_id}")
    if attempt.correct is not None:
        raise PracticeError(f"attempt {attempt_id} is already answered")
    result = grade(_problem_from(attempt), response)
    graded = attempt.model_copy(update={"response": response})
    updated = record_outcome(
        state,
        dag,
        attempt.concept_id,
        correct=result.correct,
        difficulty=attempt.difficulty,
        cold=cold,
        seconds=seconds,
        instance_key=attempt.instance_key,
        now=now,
        attempt=graded,
    )
    if attempt.origin == "diagnose":
        concept = dag.get(attempt.concept_id)
        updated = note_probe(
            updated,
            dag,
            attempt.concept_id,
            correct=result.correct,
            slow=is_slow(seconds, concept.mastery_threshold.avg_time_seconds),
            now=now,
        )
    progress = updated.progress[attempt.concept_id]
    payload: dict[str, object] = {
        "correct": result.correct,
        "feedback": result.feedback,
        "status": read_status(progress, now).value,
        "mastery_score": progress.mastery_score,
        "concept_id": attempt.concept_id,
    }
    return updated, payload


def status_payload(
    dag: ConceptDAG,
    state: StudentState,
    now: datetime,
    *,
    domain: str | None,
) -> dict[str, object]:
    domains: list[dict[str, object]] = []
    for domain_id in DOMAIN_ORDER:
        if domain is not None and domain_id != domain:
            continue
        rows: list[dict[str, object]] = []
        mastered = 0
        for concept in dag.in_domain(domain_id):
            progress = state.progress_for(concept.id)
            status = read_status(progress, now)
            if status is Status.MASTERED:
                mastered += 1
            rows.append(
                {
                    "id": concept.id,
                    "label": concept.label,
                    "status": status.value,
                    "mastery_score": 0.0 if progress is None else progress.mastery_score,
                }
            )
        domains.append(
            {
                "id": domain_id,
                "label": DOMAIN_LABELS[domain_id],
                "mastered": mastered,
                "total": len(rows),
                "concepts": rows,
            }
        )
    return {"domains": domains}


def ledger_text(dag: ConceptDAG, state: StudentState, now: datetime) -> str:
    groups: dict[str, list[str]] = {}
    for concept in dag.topological():
        if read_status(state.progress_for(concept.id), now) is not Status.MASTERED:
            continue
        key = concept.ledger_evidence_id or "unmapped"
        groups.setdefault(key, []).append(concept.id)
    if not groups:
        return ""
    lines: list[str] = []
    for key in sorted(groups):
        lines.append(key)
        lines.extend(f"  {concept_id}" for concept_id in sorted(groups[key]))
    return "\n".join(lines) + "\n"


def graph_payload(dag: ConceptDAG) -> dict[str, object]:
    return {
        "concepts": [
            {
                "id": concept.id,
                "label": concept.label,
                "domain": concept.domain,
                "prerequisites": list(concept.prerequisites),
            }
            for concept in dag.topological()
        ]
    }
