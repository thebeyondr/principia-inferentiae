"""Generator registry. Self-check is explicit so importing the CLI does not grade."""

from __future__ import annotations

from typing import Protocol

from principia_inferentiae.practice.grader import Problem, grade


class ConceptGenerator(Protocol):
    concept_id: str

    def generate(self, seed: int, difficulty: float) -> Problem: ...


GENERATORS: dict[str, ConceptGenerator] = {}
_CHECKED: set[str] = set()


def register(generator: ConceptGenerator) -> None:
    if generator.concept_id in GENERATORS:
        raise RuntimeError(f"duplicate generator {generator.concept_id}")
    GENERATORS[generator.concept_id] = generator


def check_generator(concept_id: str) -> None:
    """Seed 0 at difficulty 0.5 must grade itself correct and a blank incorrect."""

    if concept_id in _CHECKED:
        return
    generator = GENERATORS[concept_id]
    problem = generator.generate(seed=0, difficulty=0.5)
    if problem.concept_id != concept_id:
        raise RuntimeError(f"{concept_id} generated {problem.concept_id}")
    if not grade(problem, problem.answer).correct:
        raise RuntimeError(f"{concept_id} rejected its own answer")
    if grade(problem, "").correct:
        raise RuntimeError(f"{concept_id} accepted a blank answer")
    _CHECKED.add(concept_id)


def self_check() -> None:
    """Each registered generator's seed-0 item grades its own answer and rejects a blank."""

    for concept_id in sorted(GENERATORS):
        check_generator(concept_id)


def reset_self_check() -> None:
    _CHECKED.clear()
