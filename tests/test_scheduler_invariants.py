"""Scheduler hard rules and the 70/30 new/review mix."""

from __future__ import annotations

from helpers import NOW, load_concept_dag

from principia_inferentiae.practice.dag import ConceptDAG, Status
from principia_inferentiae.practice.diagnostic import covering_set, note_probe
from principia_inferentiae.practice.mastery import prereq_satisfied, read_status, record_outcome
from principia_inferentiae.practice.scheduler import candidate_pools, select
from principia_inferentiae.practice.state import StudentState


def _advance(
    state: StudentState,
    dag: ConceptDAG,
    concept_id: str,
    step: int,
) -> StudentState:
    updated = record_outcome(
        state,
        dag,
        concept_id,
        correct=True,
        difficulty=0.7,
        cold=True,
        seconds=5,
        instance_key=f"{concept_id}-{step}",
        now=NOW,
    )
    concept = dag.get(concept_id)
    return updated.model_copy(
        update={"last_concept_id": concept.id, "last_subdomain": concept.subdomain}
    )


def test_unmet_prerequisites_never_surface_and_questions_do_not_repeat() -> None:
    dag = load_concept_dag()
    state = StudentState()
    for step in range(2000):
        selection = select(dag, state, NOW, seed=0)
        concept = dag.get(selection.concept_id)
        for prereq in concept.prerequisites:
            assert prereq_satisfied(read_status(state.progress_for(prereq), NOW))
        if state.last_concept_id is not None:
            assert selection.concept_id != state.last_concept_id
        if state.last_subdomain is not None:
            assert concept.subdomain != state.last_subdomain
        state = _advance(state, dag, selection.concept_id, step)


def test_new_review_mix_is_about_seventy_thirty() -> None:
    dag = load_concept_dag()
    state = StudentState()
    samples: list[bool] = []
    for step in range(200):
        new_ids, review_ids = candidate_pools(dag, state, NOW)
        selection = select(dag, state, NOW, seed=0)
        if new_ids and review_ids:
            samples.append(selection.concept_id in new_ids)
        state = _advance(state, dag, selection.concept_id, step)
    assert len(samples) >= 20
    ratio = sum(samples) / len(samples)
    assert 0.60 <= ratio <= 0.80


def test_same_inputs_select_the_same_concept() -> None:
    dag = load_concept_dag()
    state = StudentState()
    assert select(dag, state, NOW, seed=3) == select(dag, state, NOW, seed=3)


def test_diagnostic_two_passes_place_practicing_without_demotion() -> None:
    dag = load_concept_dag()
    concept_id = covering_set(dag)[0]
    state = note_probe(
        StudentState(),
        dag,
        concept_id,
        correct=True,
        slow=False,
        now=NOW,
    )
    placed = state.progress.get(concept_id)
    assert placed is None or placed.status is not Status.PRACTICING
    state = note_probe(state, dag, concept_id, correct=True, slow=False, now=NOW)
    assert state.progress[concept_id].status is Status.PRACTICING
