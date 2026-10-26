"""Forward-only mastery transitions and weakness propagation."""

from __future__ import annotations

from datetime import datetime, timedelta

from principia_inferentiae.practice.dag import ConceptDAG, MasteryThresholdSpec, Status
from principia_inferentiae.practice.evidence import (
    COLD_MASTERED_FLOOR,
    DECAY_DAYS,
    LEARNING_SCORE,
    MASTERY_THRESHOLD,
    AttemptView,
    cold_correct_count,
    mastery_score,
)
from principia_inferentiae.practice.sm2 import SM2State, quality_score, review, speed_for
from principia_inferentiae.practice.state import AttemptRecord, ConceptProgress, StudentState

_STORED_RANK: dict[Status, int] = {
    Status.UNSEEN: 0,
    Status.LEARNING: 1,
    Status.PRACTICING: 2,
    Status.MASTERED: 3,
}


def read_status(progress: ConceptProgress | None, now: datetime) -> Status:
    """``DECAYING`` is a read-time view of ``MASTERED`` after 14 days."""

    if progress is None or progress.status is Status.UNSEEN:
        return Status.UNSEEN
    if progress.status is Status.MASTERED and progress.last_seen is not None:
        elapsed = (now - progress.last_seen).total_seconds()
        if elapsed >= DECAY_DAYS * 86400:
            return Status.DECAYING
    return progress.status


def prereq_satisfied(status: Status) -> bool:
    return status in {Status.PRACTICING, Status.MASTERED, Status.DECAYING}


def next_status(
    current: Status,
    *,
    score: float,
    answered_count: int,
    streak: int,
    cold_correct: int,
    threshold: MasteryThresholdSpec,
) -> Status:
    """Move only forward. ``DECAYING`` is not stored."""

    if current is Status.DECAYING:
        current = Status.MASTERED
    if current is Status.MASTERED:
        return Status.MASTERED
    proposed: Status = Status.LEARNING if current is Status.UNSEEN else current
    if proposed is Status.LEARNING and answered_count >= 2 and score >= LEARNING_SCORE:
        proposed = Status.PRACTICING
    if (
        proposed is Status.PRACTICING
        and answered_count >= threshold.evidence_floor
        and score >= MASTERY_THRESHOLD
        and streak >= threshold.streak
        and cold_correct >= COLD_MASTERED_FLOOR
    ):
        proposed = Status.MASTERED
    if _STORED_RANK[proposed] < _STORED_RANK[current]:
        return current
    return proposed


def optimistic_merge(current: Status, proposed: Status) -> Status:
    """Keep the higher stored rung. Used when a diagnostic probe is placed."""

    if current is Status.DECAYING:
        current = Status.MASTERED
    if proposed is Status.DECAYING:
        proposed = Status.MASTERED
    if proposed is Status.UNSEEN:
        return current
    if _STORED_RANK[proposed] > _STORED_RANK[current]:
        return proposed
    return current


def _views(attempts: tuple[AttemptRecord, ...], concept_id: str) -> tuple[AttemptView, ...]:
    return tuple(
        AttemptView(
            correct=bool(attempt.correct),
            difficulty=attempt.difficulty,
            instance_key=attempt.instance_key,
            cold=attempt.cold,
            seconds=attempt.seconds,
        )
        for attempt in attempts
        if attempt.concept_id == concept_id and attempt.correct is not None
    )


def bump_weakness(weakness: float, *, correct: bool) -> float:
    if correct:
        return max(0.0, weakness - 0.15)
    return min(1.0, weakness + 0.4)


def propagate_weakness(
    state: StudentState,
    dag: ConceptDAG,
    concept_id: str,
    weakness: float,
) -> StudentState:
    """A miss with weakness above 0.3 adds ``weakness * 0.3`` to each dependent."""

    if weakness <= 0.3:
        return state
    updated = state
    for dependent_id in dag.dependents[concept_id]:
        current = updated.progress.get(dependent_id)
        base = 0.0 if current is None else current.weakness
        grown = min(1.0, base + weakness * 0.3)
        if current is None:
            progress = ConceptProgress(concept_id=dependent_id, weakness=grown)
        else:
            progress = current.model_copy(update={"weakness": grown})
        updated = updated.with_progress(progress)
    return updated


def record_outcome(
    state: StudentState,
    dag: ConceptDAG,
    concept_id: str,
    *,
    correct: bool,
    difficulty: float,
    cold: bool,
    seconds: float | None,
    instance_key: str,
    now: datetime,
    attempt: AttemptRecord | None = None,
) -> StudentState:
    """Update mastery for one graded answer. Status never moves backward."""

    concept = dag.get(concept_id)
    prior = state.progress.get(concept_id)
    stored = Status.UNSEEN if prior is None else prior.status
    if attempt is None:
        synthetic = AttemptRecord(
            attempt_id=f"synthetic-{concept_id}-{len(state.attempts)}",
            seq=len(state.attempts),
            concept_id=concept_id,
            seed=0,
            difficulty=difficulty,
            prompt="",
            canonical_answer="",
            facts={},
            grading_type=concept.grading_type,
            label=concept.label,
            domain=concept.domain,
            served_at=now,
            answered_at=now,
            response="",
            correct=correct,
            cold=cold,
            seconds=seconds,
            instance_key=instance_key,
            origin="practice",
        )
        state = state.with_attempt(synthetic)
    else:
        graded = attempt.model_copy(
            update={
                "answered_at": now,
                "response": attempt.response,
                "correct": correct,
                "cold": cold,
                "seconds": seconds,
            }
        )
        state = state.replace_attempt(graded)
    views = _views(state.attempts, concept_id)
    score = mastery_score(views, concept.mastery_threshold.avg_time_seconds)
    streak = 0 if prior is None else prior.streak
    streak = streak + 1 if correct else 0
    weakness = bump_weakness(0.0 if prior is None else prior.weakness, correct=correct)
    status = next_status(
        stored,
        score=score,
        answered_count=len(views),
        streak=streak,
        cold_correct=cold_correct_count(views),
        threshold=concept.mastery_threshold,
    )
    sm2_prior = SM2State(
        repetitions=0 if prior is None else prior.repetitions,
        easiness=2.5 if prior is None else prior.easiness,
        interval_days=0 if prior is None else prior.interval_days,
    )
    quality = quality_score(
        correct=correct,
        difficulty=difficulty,
        seconds=seconds,
        avg_time_seconds=concept.mastery_threshold.avg_time_seconds,
    )
    paced = speed_for(seconds, concept.mastery_threshold.avg_time_seconds)
    scheduled = review(sm2_prior, quality, paced)
    progress = ConceptProgress(
        concept_id=concept_id,
        status=status,
        mastery_score=score,
        weakness=weakness,
        repetitions=scheduled.repetitions,
        easiness=scheduled.easiness,
        interval_days=scheduled.interval_days,
        next_due=now + timedelta(days=scheduled.interval_days),
        last_seen=now,
        streak=streak,
    )
    state = state.with_progress(progress)
    if not correct:
        state = propagate_weakness(state, dag, concept_id, weakness)
    return state
