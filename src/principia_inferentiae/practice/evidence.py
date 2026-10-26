"""Evidence window, mastery score, and rung ordering."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict

from principia_inferentiae.practice.dag import Status

WINDOW = 6
MASTERY_THRESHOLD = 0.6
MASTERY_EVIDENCE_FLOOR = 6
NEUTRAL_DIFFICULTY = 0.55
DECAY_DAYS = 14
LEARNING_SCORE = 0.35
COLD_MASTERED_FLOOR = 2


class Rung(IntEnum):
    UNSEEN = 0
    LEARNING = 1
    PRACTICING = 2
    MASTERED = 3


_RUNG_OF: dict[Status, Rung] = {
    Status.UNSEEN: Rung.UNSEEN,
    Status.LEARNING: Rung.LEARNING,
    Status.PRACTICING: Rung.PRACTICING,
    Status.MASTERED: Rung.MASTERED,
    Status.DECAYING: Rung.MASTERED,
}


class AttemptView(BaseModel):
    """Fields the mastery score is allowed to see."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    correct: bool
    difficulty: float
    instance_key: str
    cold: bool
    seconds: float | None = None


def window(attempts: tuple[AttemptView, ...] | list[AttemptView]) -> tuple[AttemptView, ...]:
    return tuple(attempts[-WINDOW:])


def mastery_score(attempts: tuple[AttemptView, ...] | list[AttemptView], avg_time_seconds: float | None) -> float:
    rows = window(attempts)
    if not rows:
        return 0.0
    correct = tuple(row for row in rows if row.correct)
    correctness = len(correct) / len(rows)
    if correct:
        mean_difficulty = sum(row.difficulty for row in correct) / len(correct)
        difficulty_term = min(1.0, mean_difficulty / NEUTRAL_DIFFICULTY)
    else:
        difficulty_term = 0.0
    variety = len({row.instance_key for row in rows}) / len(rows)
    if avg_time_seconds is None:
        return 0.40 * correctness + 0.25 * difficulty_term + 0.35 * variety
    on_time = (
        sum(1 for row in rows if row.seconds is not None and row.seconds <= avg_time_seconds)
        / len(rows)
    )
    return 0.35 * correctness + 0.20 * difficulty_term + 0.25 * variety + 0.20 * on_time


def highest_rung(statuses: tuple[Status, ...] | list[Status]) -> Rung:
    if not statuses:
        return Rung.UNSEEN
    return max(_RUNG_OF[status] for status in statuses)


def cold_correct_count(attempts: tuple[AttemptView, ...] | list[AttemptView]) -> int:
    return sum(1 for row in window(attempts) if row.correct and row.cold)
