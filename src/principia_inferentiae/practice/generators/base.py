"""Numeric and choice problem constructors shared by generators."""

from __future__ import annotations

import random
from typing import TypeVar

from principia_inferentiae.practice.dag import GradingType
from principia_inferentiae.practice.grader import Problem, display_amount, format_number
from principia_inferentiae.practice.state import Choice

_T = TypeVar("_T")


def band(difficulty: float) -> int:
    if difficulty < 0.34:
        return 0
    if difficulty < 0.67:
        return 1
    return 2


def steps_hidden(difficulty: float) -> bool:
    return band(difficulty) >= 2


def pick(seed: int, difficulty: float, options: tuple[_T, ...]) -> _T:
    """Low difficulty stays on the small end; high difficulty uses the large end."""

    if not options:
        raise ValueError("empty options")
    level = band(difficulty)
    if len(options) == 1 or level == 1:
        pool = options
    elif level == 0:
        pool = options[: max(1, (len(options) + 2) // 3)]
    else:
        pool = options[len(options) // 2 :]
    return pool[random.Random(seed).randrange(len(pool))]


def numeric_problem(
    *,
    concept_id: str,
    seed: int,
    difficulty: float,
    prompt: str,
    canonical: float,
    unit: str,
    facts: dict[str, object],
    tolerance: float = 0.02,
) -> Problem:
    published = dict(facts)
    published["asked_unit"] = unit
    return Problem(
        concept_id=concept_id,
        prompt=prompt,
        answer=format_number(display_amount(canonical, unit)),
        facts=published,
        unit=unit,
        tolerance=tolerance,
        instance_key=f"{concept_id}:{seed}:{band(difficulty)}",
        grading_type=GradingType.NUMERIC_UNITS,
    )


def text_problem(
    *,
    concept_id: str,
    seed: int,
    difficulty: float,
    prompt: str,
    answer: str,
    facts: dict[str, object],
    choices: tuple[tuple[str, str], ...] | None,
    grading_type: GradingType,
    order_threshold: float = 0.8,
) -> Problem:
    choice_models = None if choices is None else tuple(Choice(id=item_id, label=label) for item_id, label in choices)
    return Problem(
        concept_id=concept_id,
        prompt=prompt,
        answer=answer,
        facts=facts,
        choices=choice_models,
        unit=None,
        instance_key=f"{concept_id}:{seed}:{band(difficulty)}",
        order_threshold=order_threshold,
        grading_type=grading_type,
    )
