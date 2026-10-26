"""SM-2 table, mastery transitions, decay, and weakness propagation."""

from __future__ import annotations

from datetime import timedelta

import pytest
from helpers import NOW, load_concept_dag

from principia_inferentiae.practice.dag import Status
from principia_inferentiae.practice.evidence import (
    MASTERY_EVIDENCE_FLOOR,
    MASTERY_THRESHOLD,
    NEUTRAL_DIFFICULTY,
)
from principia_inferentiae.practice.mastery import read_status, record_outcome
from principia_inferentiae.practice.scheduler import priority_score
from principia_inferentiae.practice.sm2 import SM2State, review
from principia_inferentiae.practice.state import ConceptProgress, StudentState


@pytest.mark.parametrize(
    ("qualities", "speed", "expected"),
    [
        ((5, 5, 5), 1.0, ((1, 2.6, 1), (2, 2.7, 6), (3, 2.8, 17))),
        ((4,), 1.0, ((1, 2.5, 1),)),
        ((3,), 1.0, ((1, 2.36, 1),)),
        ((2,), 1.0, ((0, 2.5, 1),)),
        ((0,), 1.0, ((0, 2.5, 1),)),
        ((5, 5, 5), 2.0, ((1, 2.6, 1), (2, 2.7, 6), (3, 2.8, 34))),
        ((5, 5, 5), 0.1, ((1, 2.6, 1), (2, 2.7, 6), (3, 2.8, 8))),
        ((5, 1), 1.0, ((1, 2.6, 1), (0, 2.6, 1))),
    ],
)
def test_sm2_table(
    qualities: tuple[int, ...],
    speed: float,
    expected: tuple[tuple[int, float, int], ...],
) -> None:
    state = SM2State()
    for quality, (repetitions, easiness, interval) in zip(qualities, expected, strict=True):
        state = review(state, quality, speed)
        assert state.repetitions == repetitions
        assert state.easiness == pytest.approx(easiness)
        assert state.interval_days == interval


def test_constants() -> None:
    assert MASTERY_THRESHOLD == 0.6
    assert MASTERY_EVIDENCE_FLOOR == 6
    assert NEUTRAL_DIFFICULTY == 0.55


def test_decay_is_read_time_only() -> None:
    progress = ConceptProgress(
        concept_id="units.bytes_prefixes",
        status=Status.MASTERED,
        last_seen=NOW - timedelta(days=14),
    )
    assert read_status(progress, NOW) is Status.DECAYING
    assert progress.status is Status.MASTERED
    recent = progress.model_copy(update={"last_seen": NOW - timedelta(days=13, hours=23)})
    assert read_status(recent, NOW) is Status.MASTERED


def test_status_does_not_move_backward() -> None:
    dag = load_concept_dag()
    state = StudentState()
    for index in range(6):
        state = record_outcome(
            state,
            dag,
            "units.bytes_prefixes",
            correct=True,
            difficulty=0.8,
            cold=True,
            seconds=5,
            instance_key=f"k{index}",
            now=NOW,
        )
    assert state.progress["units.bytes_prefixes"].status is Status.MASTERED
    state = record_outcome(
        state,
        dag,
        "units.bytes_prefixes",
        correct=False,
        difficulty=0.8,
        cold=False,
        seconds=5,
        instance_key="miss",
        now=NOW,
    )
    assert state.progress["units.bytes_prefixes"].status is Status.MASTERED


def test_mastery_requires_two_cold_correct_attempts() -> None:
    dag = load_concept_dag()
    state = StudentState()
    for index in range(6):
        state = record_outcome(
            state,
            dag,
            "units.us_ms_s",
            correct=True,
            difficulty=0.8,
            cold=index == 0,
            seconds=5,
            instance_key=f"k{index}",
            now=NOW,
        )
    assert state.progress["units.us_ms_s"].status is not Status.MASTERED


def test_weakness_propagates_to_dependents() -> None:
    dag = load_concept_dag()
    state = record_outcome(
        StudentState(),
        dag,
        "units.bytes_prefixes",
        correct=False,
        difficulty=0.5,
        cold=False,
        seconds=None,
        instance_key="miss",
        now=NOW,
    )
    assert state.progress["units.bytes_prefixes"].weakness == pytest.approx(0.4)
    assert state.progress["model.params_bytes"].weakness == pytest.approx(0.12)


def test_priority_formula() -> None:
    score = priority_score(
        days_since_seen=4,
        mastery=0.25,
        weakness=0.5,
        decaying=True,
        encompasses_weak=True,
        interference_recent=True,
    )
    assert score == pytest.approx(0.5 * 4 + 0.2 * 0.75 + 0.3 * 0.5 + 5 + 2 - 3)
