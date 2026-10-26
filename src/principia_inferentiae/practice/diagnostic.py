"""Topological diagnostic. Placement never lowers a stored status."""

from __future__ import annotations

from datetime import datetime

from principia_inferentiae.practice.dag import ConceptDAG, Status
from principia_inferentiae.practice.mastery import optimistic_merge
from principia_inferentiae.practice.scheduler import select
from principia_inferentiae.practice.state import ConceptProgress, ProbeRecord, StudentState


def covering_set(dag: ConceptDAG) -> tuple[str, ...]:
    """One concept per median depth band, in topological order."""

    if len(dag) == 0:
        return ()
    depths = sorted({dag.depth(concept_id) for concept_id in dag.topological_ids()})
    median = depths[len(depths) // 2]
    return tuple(
        concept_id for concept_id in dag.topological_ids() if dag.depth(concept_id) == median
    )


def is_slow(seconds: float | None, avg_time_seconds: float | None) -> bool:
    if seconds is None or avg_time_seconds is None:
        return False
    return seconds >= 2 * avg_time_seconds


def _probes(state: StudentState, concept_id: str) -> tuple[ProbeRecord, ...]:
    return tuple(probe for probe in state.probes if probe.concept_id == concept_id)


def _good(probe: ProbeRecord) -> bool:
    return probe.correct and not probe.slow


def placement(probes: tuple[ProbeRecord, ...]) -> Status | None:
    """Return a placement after two agreeing probes, else None while still probing."""

    if len(probes) < 2:
        return None
    tail = probes[-2:]
    if all(_good(probe) for probe in tail):
        return Status.PRACTICING
    if all(not _good(probe) for probe in tail):
        return Status.UNSEEN
    if len(probes) < 3:
        return None
    goods = sum(1 for probe in probes if _good(probe))
    if goods >= len(probes) - goods:
        return Status.PRACTICING
    return Status.UNSEEN


def _ancestors(dag: ConceptDAG, concept_id: str) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    stack = list(dag.get(concept_id).prerequisites)
    while stack:
        item = stack.pop()
        if item in seen:
            continue
        seen.add(item)
        found.append(item)
        stack.extend(dag.get(item).prerequisites)
    return tuple(found)


def select_probe(dag: ConceptDAG, state: StudentState, now: datetime, seed: int) -> str:
    if state.probes:
        last = state.probes[-1]
        if placement(_probes(state, last.concept_id)) is None:
            return last.concept_id
        if _good(last):
            forward = tuple(
                concept_id
                for concept_id in dag.dependents[last.concept_id]
                if placement(_probes(state, concept_id)) is None
            )
            if forward:
                return sorted(forward, key=lambda concept_id: (dag.depth(concept_id), concept_id))[0]
        else:
            backward = tuple(
                concept_id
                for concept_id in dag.get(last.concept_id).prerequisites
                if placement(_probes(state, concept_id)) is None
            )
            if backward:
                return sorted(backward, key=lambda concept_id: (-dag.depth(concept_id), concept_id))[0]
    else:
        cover = covering_set(dag)
        if cover:
            return cover[0]
    for concept_id in dag.topological_ids():
        if placement(_probes(state, concept_id)) is None:
            return concept_id
    return select(dag, state, now, seed).concept_id


def note_probe(
    state: StudentState,
    dag: ConceptDAG,
    concept_id: str,
    *,
    correct: bool,
    slow: bool,
    now: datetime,
) -> StudentState:
    state = state.with_probe(ProbeRecord(concept_id=concept_id, correct=correct, slow=slow))
    placed = placement(_probes(state, concept_id))
    if placed is not Status.PRACTICING:
        return state
    current = state.progress.get(concept_id)
    stored = Status.UNSEEN if current is None else current.status
    merged = optimistic_merge(stored, Status.PRACTICING)
    if current is None:
        state = state.with_progress(
            ConceptProgress(concept_id=concept_id, status=merged, last_seen=now)
        )
    elif merged is not stored:
        state = state.with_progress(current.model_copy(update={"status": merged}))
    for ancestor in _ancestors(dag, concept_id):
        ancestor_progress = state.progress.get(ancestor)
        ancestor_status = Status.UNSEEN if ancestor_progress is None else ancestor_progress.status
        if ancestor_status is not Status.UNSEEN:
            continue
        if ancestor_progress is None:
            state = state.with_progress(
                ConceptProgress(concept_id=ancestor, status=Status.LEARNING, last_seen=now)
            )
        else:
            state = state.with_progress(ancestor_progress.model_copy(update={"status": Status.LEARNING}))
    return state
