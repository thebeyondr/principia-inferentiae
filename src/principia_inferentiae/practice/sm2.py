"""SM-2 scheduling. Quality below 3 resets repetitions; easiness only moves on success."""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_EASINESS = 2.5
MIN_EASINESS = 1.3
MIN_SPEED = 0.5
MAX_SPEED = 2.0
MAX_INTERVAL_DAYS = 365


@dataclass(frozen=True)
class SM2State:
    repetitions: int = 0
    easiness: float = DEFAULT_EASINESS
    interval_days: int = 0


def clamp_speed(speed: float) -> float:
    return min(MAX_SPEED, max(MIN_SPEED, speed))


def round_half_away(value: float) -> int:
    """Match the half-away-from-zero rounding used by the reference scheduler."""

    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def review(state: SM2State, quality: int, speed: float = 1.0) -> SM2State:
    """Apply one SM-2 review. ``speed`` is clamped to ``[0.5, 2.0]``."""

    if quality < 3:
        return SM2State(
            repetitions=0,
            easiness=state.easiness,
            interval_days=1,
        )
    delta = 5 - quality
    easiness = state.easiness + (0.1 - delta * (0.08 + delta * 0.02))
    easiness = max(MIN_EASINESS, easiness)
    repetitions = state.repetitions + 1
    paced = clamp_speed(speed)
    if repetitions == 1:
        interval = 1
    elif repetitions == 2:
        interval = 6
    else:
        interval = max(1, round_half_away(state.interval_days * easiness * paced))
    interval = min(interval, MAX_INTERVAL_DAYS)
    return SM2State(repetitions=repetitions, easiness=easiness, interval_days=interval)


def quality_score(
    *,
    correct: bool,
    difficulty: float,
    seconds: float | None,
    avg_time_seconds: float | None,
) -> int:
    """Map a graded attempt onto the SM-2 quality scale."""

    if not correct:
        return 2
    if difficulty >= 0.75:
        quality = 5
    elif difficulty >= 0.45:
        quality = 4
    else:
        quality = 3
    if avg_time_seconds is not None and seconds is not None and seconds > avg_time_seconds:
        quality = max(3, quality - 1)
    return quality


def speed_for(seconds: float | None, avg_time_seconds: float | None) -> float:
    if avg_time_seconds is None or seconds is None or seconds <= 0:
        return 1.0
    return clamp_speed(avg_time_seconds / seconds)
