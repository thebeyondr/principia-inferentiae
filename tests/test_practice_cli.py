"""CLI JSON contract. The anchor lives in sqlite, not in the response."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import Result
from helpers import NOW, load_concept_dag
from typer.testing import CliRunner

from principia_inferentiae.cli import _save_state, app
from principia_inferentiae.practice.mastery import record_outcome
from principia_inferentiae.practice.state import StudentState

RUNNER = CliRunner()
NEXT_KEYS = {
    "attempt_id",
    "concept_id",
    "label",
    "domain",
    "grading_type",
    "prompt",
    "choices",
    "unit",
    "cold_offered",
}
ANSWER_KEYS = {"correct", "feedback", "status", "mastery_score", "concept_id"}


def _invoke(args: list[str]) -> Result:
    result = RUNNER.invoke(app, args)
    assert result.exit_code == 0, result.stdout + result.stderr
    return result


def test_next_and_answer_round_trip(tmp_path: Path) -> None:
    state = tmp_path / "practice.sqlite"
    now = "2026-06-01T00:00:00+00:00"
    first = _invoke(
        ["practice", "next", "--json", "--state", str(state), "--now", now, "--seed", "1"]
    )
    payload = json.loads(first.stdout)
    assert set(payload) == NEXT_KEYS
    assert payload["cold_offered"] is True
    connection = sqlite3.connect(state)
    row = connection.execute(
        "SELECT canonical_answer FROM attempts WHERE attempt_id = ?",
        (payload["attempt_id"],),
    ).fetchone()
    connection.close()
    assert row is not None
    wrong = _invoke(
        [
            "practice",
            "answer",
            payload["attempt_id"],
            "definitely-wrong",
            "--json",
            "--state",
            str(state),
            "--now",
            now,
        ]
    )
    verdict = json.loads(wrong.stdout)
    assert set(verdict) == ANSWER_KEYS
    assert verdict["correct"] is False
    assert verdict["concept_id"] == payload["concept_id"]
    again = _invoke(
        ["practice", "next", "--json", "--state", str(state), "--now", now, "--seed", "1"]
    )
    served = json.loads(again.stdout)
    connection = sqlite3.connect(state)
    canonical = connection.execute(
        "SELECT canonical_answer FROM attempts WHERE attempt_id = ?",
        (served["attempt_id"],),
    ).fetchone()
    connection.close()
    assert canonical is not None
    right = _invoke(
        [
            "practice",
            "answer",
            served["attempt_id"],
            str(canonical[0]),
            "--json",
            "--cold",
            "--seconds",
            "5",
            "--state",
            str(state),
            "--now",
            now,
        ]
    )
    graded = json.loads(right.stdout)
    assert graded["correct"] is True
    assert graded["status"] in {"LEARNING", "PRACTICING", "MASTERED", "DECAYING"}


def test_next_is_deterministic(tmp_path: Path) -> None:
    now = "2026-06-01T00:00:00+00:00"
    left = _invoke(
        [
            "practice",
            "next",
            "--json",
            "--state",
            str(tmp_path / "a.sqlite"),
            "--now",
            now,
            "--seed",
            "4",
        ]
    )
    right = _invoke(
        [
            "practice",
            "next",
            "--json",
            "--state",
            str(tmp_path / "b.sqlite"),
            "--now",
            now,
            "--seed",
            "4",
        ]
    )
    assert left.stdout == right.stdout


def test_status_and_export_graph(tmp_path: Path) -> None:
    state = tmp_path / "practice.sqlite"
    now = "2026-06-01T00:00:00+00:00"
    status = _invoke(
        ["practice", "status", "--json", "--state", str(state), "--now", now, "--domain", "units"]
    )
    payload = json.loads(status.stdout)
    assert set(payload) == {"domains"}
    assert len(payload["domains"]) == 1
    domain = payload["domains"][0]
    assert set(domain) == {"id", "label", "mastered", "total", "concepts"}
    assert domain["id"] == "units"
    assert domain["mastered"] == 0
    assert domain["total"] == 5
    concept = domain["concepts"][0]
    assert set(concept) == {"id", "label", "status", "mastery_score"}
    output = tmp_path / "graph.json"
    _invoke(["practice", "export-graph", "--output", str(output)])
    graph = json.loads(output.read_text(encoding="utf-8"))
    assert set(graph) == {"concepts"}
    assert len(graph["concepts"]) == 63
    assert set(graph["concepts"][0]) == {"id", "label", "domain", "prerequisites"}


def test_unknown_attempt_exits_nonzero(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "practice",
            "answer",
            "missing",
            "1",
            "--json",
            "--state",
            str(tmp_path / "practice.sqlite"),
            "--now",
            "2026-06-01T00:00:00+00:00",
        ],
    )
    assert result.exit_code != 0


def test_ledger_groups_mastered_concepts(tmp_path: Path) -> None:
    state_path = tmp_path / "practice.sqlite"
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
    _save_state(state_path, state, NOW)
    result = _invoke(
        [
            "practice",
            "ledger",
            "--state",
            str(state_path),
            "--now",
            "2026-06-01T00:00:00+00:00",
        ]
    )
    assert "unmapped" in result.stdout
    assert "units.bytes_prefixes" in result.stdout


def test_diagnose_json_shape(tmp_path: Path) -> None:
    result = _invoke(
        [
            "practice",
            "diagnose",
            "--json",
            "--state",
            str(tmp_path / "practice.sqlite"),
            "--now",
            "2026-06-01T00:00:00+00:00",
            "--seed",
            "0",
        ]
    )
    payload = json.loads(result.stdout)
    assert set(payload) == NEXT_KEYS
