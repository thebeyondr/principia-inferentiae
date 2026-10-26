"""Priority score, hard eligibility rules, and the 70/30 new/review split."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from principia_inferentiae.practice.dag import Concept, ConceptDAG, Status
from principia_inferentiae.practice.evidence import NEUTRAL_DIFFICULTY
from principia_inferentiae.practice.mastery import prereq_satisfied, read_status
from principia_inferentiae.practice.state import StudentState

NEW_SLOTS = 7
SLOT_CYCLE = 10
RECENT_INTERFERENCE = 3


@dataclass(frozen=True)
class Selection:
    concept_id: str
    difficulty: float
    is_new: bool
    new_ids: tuple[str, ...]
    review_ids: tuple[str, ...]


def priority_score(
    *,
    days_since_seen: float,
    mastery: float,
    weakness: float,
    decaying: bool,
    encompasses_weak: bool,
    interference_recent: bool,
) -> float:
    return (
        0.5 * days_since_seen
        + 0.2 * (1.0 - mastery)
        + 0.3 * weakness
        + (5.0 if decaying else 0.0)
        + (2.0 if encompasses_weak else 0.0)
        - (3.0 if interference_recent else 0.0)
    )


def _days_since(now: datetime, seen: datetime | None) -> float:
    if seen is None:
        return 0.0
    return max(0.0, (now - seen).total_seconds() / 86400.0)


def _recent_groups(dag: ConceptDAG, state: StudentState) -> set[str]:
    groups: set[str] = set()
    for attempt in state.answered()[-RECENT_INTERFERENCE:]:
        group = dag.get(attempt.concept_id).interference_group
        if group is not None:
            groups.add(group)
    return groups


def _eligible(dag: ConceptDAG, state: StudentState, now: datetime) -> tuple[Concept, ...]:
    chosen: list[Concept] = []
    for concept in dag.topological():
        if concept.prerequisites and not all(
            prereq_satisfied(read_status(state.progress_for(prereq), now))
            for prereq in concept.prerequisites
        ):
            continue
        chosen.append(concept)
    return tuple(chosen)


def _ranked(
    concepts: tuple[Concept, ...],
    dag: ConceptDAG,
    state: StudentState,
    now: datetime,
) -> tuple[Concept, ...]:
    recent = _recent_groups(dag, state)

    def sort_key(concept: Concept) -> tuple[float, str]:
        progress = state.progress_for(concept.id)
        status = read_status(progress, now)
        mastery = 0.0 if progress is None else progress.mastery_score
        weakness = 0.0 if progress is None else progress.weakness
        days = 0.0 if progress is None else _days_since(now, progress.last_seen)
        encompasses_weak = False
        for covered in concept.encompasses:
            covered_progress = state.progress_for(covered)
            if covered_progress is not None and covered_progress.weakness > 0.3:
                encompasses_weak = True
                break
        group = concept.interference_group
        interference = group is not None and group in recent and concept.id != state.last_concept_id
        score = priority_score(
            days_since_seen=days,
            mastery=mastery,
            weakness=weakness,
            decaying=status is Status.DECAYING,
            encompasses_weak=encompasses_weak,
            interference_recent=interference,
        )
        return (-score, concept.id)

    return tuple(sorted(concepts, key=sort_key))


def candidate_pools(
    dag: ConceptDAG,
    state: StudentState,
    now: datetime,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """New and review ids that pass prerequisite, repeat, and subdomain rules."""

    eligible = _eligible(dag, state, now)
    filtered = tuple(
        concept
        for concept in eligible
        if concept.id != state.last_concept_id and concept.subdomain != state.last_subdomain
    )
    if not filtered:
        filtered = tuple(concept for concept in eligible if concept.id != state.last_concept_id)
    if not filtered:
        filtered = eligible
    ranked = _ranked(filtered, dag, state, now)
    new_ids = tuple(
        concept.id
        for concept in ranked
        if read_status(state.progress_for(concept.id), now) is Status.UNSEEN
    )
    review_ids = tuple(
        concept.id
        for concept in ranked
        if read_status(state.progress_for(concept.id), now) is not Status.UNSEEN
    )
    return new_ids, review_ids


def wants_new(answered_count: int, seed: int) -> bool:
    return (answered_count + seed) % SLOT_CYCLE < NEW_SLOTS


def served_difficulty(mastery: float, status: Status, seed: int, concept_id: str) -> float:
    base = NEUTRAL_DIFFICULTY if status is Status.UNSEEN else min(0.95, 0.25 + 0.7 * mastery)
    # Deterministic jitter in [-0.05, 0.05] from the seed and concept id.
    mix = sum(ord(char) for char in concept_id) + seed * 17
    jitter = ((mix % 11) - 5) / 100.0
    return min(0.95, max(0.05, base + jitter))


def select(dag: ConceptDAG, state: StudentState, now: datetime, seed: int) -> Selection:
    new_ids, review_ids = candidate_pools(dag, state, now)
    prefer_new = wants_new(len(state.answered()), seed)
    primary = new_ids if prefer_new else review_ids
    secondary = review_ids if prefer_new else new_ids
    pool = primary or secondary
    if not pool:
        raise ValueError("no concept is eligible")
    concept_id = pool[0]
    status = read_status(state.progress_for(concept_id), now)
    progress = state.progress_for(concept_id)
    mastery = 0.0 if progress is None else progress.mastery_score
    return Selection(
        concept_id=concept_id,
        difficulty=served_difficulty(mastery, status, seed, concept_id),
        is_new=status is Status.UNSEEN,
        new_ids=new_ids,
        review_ids=review_ids,
    )
