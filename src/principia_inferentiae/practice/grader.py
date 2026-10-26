"""Six graders. Intervene re-runs the simulator against the stored scenario."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from principia_inferentiae.practice.dag import GradingType
from principia_inferentiae.practice.sim_access import run, scenario_dict, separates
from principia_inferentiae.practice.state import Choice

_NUMBER = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([^\d\s][^\s]*)?\s*$"
)

# Unit token → (canonical dimension, multiplier into that dimension).
_UNITS: dict[str, tuple[str, float]] = {
    "B": ("B", 1.0),
    "byte": ("B", 1.0),
    "bytes": ("B", 1.0),
    "KiB": ("B", 1024.0),
    "MiB": ("B", 1024.0**2),
    "GiB": ("B", 1024.0**3),
    "TiB": ("B", 1024.0**4),
    "KB": ("B", 1_000.0),
    "MB": ("B", 1_000.0**2),
    "GB": ("B", 1_000.0**3),
    "TB": ("B", 1_000.0**4),
    "us": ("s", 1e-6),
    "µs": ("s", 1e-6),
    "μs": ("s", 1e-6),
    "ms": ("s", 1e-3),
    "s": ("s", 1.0),
    "TFLOP/s": ("FLOP/s", 1e12),
    "GFLOP/s": ("FLOP/s", 1e9),
    "FLOP/s": ("FLOP/s", 1.0),
    "GB/s": ("B/s", 1e9),
    "GiB/s": ("B/s", 1024.0**3),
    "B/s": ("B/s", 1.0),
    "tokens/s": ("tokens/s", 1.0),
    "tok/s": ("tokens/s", 1.0),
    "ms/token": ("s/token", 1e-3),
    "s/token": ("s/token", 1.0),
    "us/token": ("s/token", 1e-6),
    "tokens": ("tokens", 1.0),
    "token": ("tokens", 1.0),
    "requests/s": ("requests/s", 1.0),
    "GPU": ("GPU", 1.0),
    "FLOP/byte": ("FLOP/byte", 1.0),
    "fraction": ("fraction", 1.0),
    "shards": ("count", 1.0),
    "heads": ("count", 1.0),
    "layers": ("count", 1.0),
    "count": ("count", 1.0),
    "1": ("count", 1.0),
}


class Problem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_id: str
    prompt: str
    answer: str
    facts: dict[str, object]
    choices: tuple[Choice, ...] | None = None
    unit: str | None = None
    tolerance: float = 0.02
    instance_key: str
    order_threshold: float = 0.8
    grading_type: GradingType


class GradeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    correct: bool
    feedback: str


def format_number(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) <= 1e-6 * max(1.0, abs(value)):
        return str(int(rounded))
    return format(value, ".10g")


def to_canonical(number: float, unit: str) -> tuple[str, float]:
    if unit not in _UNITS:
        raise ValueError(f"unknown unit {unit}")
    dimension, multiplier = _UNITS[unit]
    return dimension, number * multiplier


def display_amount(canonical: float, unit: str) -> float:
    if unit not in _UNITS:
        raise ValueError(f"unknown unit {unit}")
    return canonical / _UNITS[unit][1]


def grade(problem: Problem, response: str) -> GradeResult:
    kind = problem.grading_type
    if kind is GradingType.NUMERIC_UNITS:
        return _grade_numeric(problem, response)
    if kind is GradingType.CHOICE or kind is GradingType.CLASSIFY:
        return _grade_exact(problem, response)
    if kind is GradingType.FORMULA:
        return _grade_formula(problem, response)
    if kind is GradingType.ORDER:
        return _grade_order(problem, response)
    if kind is GradingType.INTERVENE:
        return _grade_intervene(problem, response)
    raise ValueError(f"unsupported grading type {kind}")


def _fail(feedback: str) -> GradeResult:
    return GradeResult(correct=False, feedback=feedback)


def _ok(feedback: str = "Correct.") -> GradeResult:
    return GradeResult(correct=True, feedback=feedback)


def _parse_quantity(text: str, default_unit: str | None) -> tuple[float, str]:
    match = _NUMBER.fullmatch(text)
    if match is None:
        raise ValueError("Could not parse a number and unit.")
    number = float(match.group(1))
    unit = match.group(2) or default_unit
    if unit is None:
        raise ValueError("A unit is required.")
    return number, unit


def _grade_numeric(problem: Problem, response: str) -> GradeResult:
    try:
        expected_number, expected_unit = _parse_quantity(problem.answer, problem.unit)
        expected_dim, expected = to_canonical(expected_number, expected_unit)
        got_number, got_unit = _parse_quantity(response, problem.unit)
        got_dim, got = to_canonical(got_number, got_unit)
    except ValueError as error:
        return _fail(str(error))
    if expected_dim != got_dim:
        return _fail(f"Expected a {expected_dim} quantity.")
    if expected == 0:
        correct = got == 0
    else:
        correct = abs(got - expected) / abs(expected) <= problem.tolerance
    if correct:
        return _ok()
    unit = problem.unit or expected_unit
    return _fail(f"Expected {problem.answer} {unit}.")


def _choice_label(problem: Problem, choice_id: str) -> str:
    if problem.choices is None:
        return choice_id
    for choice in problem.choices:
        if choice.id == choice_id:
            return choice.label
    return choice_id


def _matches_choice(problem: Problem, response: str) -> bool:
    text = response.strip()
    if text == problem.answer:
        return True
    if problem.choices is None:
        return text.casefold() == problem.answer.casefold()
    for choice in problem.choices:
        if text == choice.id or text.casefold() == choice.label.casefold():
            return choice.id == problem.answer
    return False


def _grade_exact(problem: Problem, response: str) -> GradeResult:
    if _matches_choice(problem, response):
        return _ok()
    return _fail(f"Expected {_choice_label(problem, problem.answer)}.")


def _tokens(text: str) -> frozenset[str]:
    parts: list[str] = []
    current: list[str] = []
    for char in text.casefold():
        if char.isalnum() or char in {"λ", "Λ"}:
            current.append("λ" if char in {"λ", "Λ"} else char)
        elif current:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return frozenset(parts)


def _grade_formula(problem: Problem, response: str) -> GradeResult:
    if _matches_choice(problem, response):
        return _ok()
    if problem.choices is not None:
        label = _choice_label(problem, problem.answer)
        if _tokens(response) == _tokens(label) and _tokens(label):
            return _ok()
    return _fail(f"Expected {_choice_label(problem, problem.answer)}.")


def _parse_order(text: str) -> list[str] | None:
    if not text.strip():
        return None
    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        return None
    return parts


def _kendall_tau(expected: list[str], got: list[str]) -> float:
    if len(expected) != len(got) or set(expected) != set(got):
        return -1.0
    rank = {item: index for index, item in enumerate(expected)}
    pairs = 0
    concordant = 0
    for left in range(len(got)):
        for right in range(left + 1, len(got)):
            pairs += 1
            if rank[got[left]] < rank[got[right]]:
                concordant += 1
    if pairs == 0:
        return 1.0
    return (2 * concordant - pairs) / pairs


def _grade_order(problem: Problem, response: str) -> GradeResult:
    expected = _parse_order(problem.answer)
    got = _parse_order(response)
    if expected is None or got is None:
        return _fail(f"Expected order {problem.answer}.")
    if _kendall_tau(expected, got) + 1e-12 >= problem.order_threshold:
        return _ok()
    return _fail(f"Expected order {problem.answer}.")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("repairs must be a list")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("repair id must be a string")
        items.append(item)
    return items


def _grade_intervene(problem: Problem, response: str) -> GradeResult:
    repair = response.strip()
    repairs = _string_list(problem.facts.get("repairs"))
    scenario = scenario_dict(problem.facts.get("scenario"))
    if repair not in repairs:
        return _fail(f"Expected {_choice_label(problem, problem.answer)}.")
    baseline = run(scenario, None)
    repaired = run(scenario, repair)
    if separates(scenario, baseline, repaired):
        return _ok()
    return _fail(f"Expected {_choice_label(problem, problem.answer)}.")
