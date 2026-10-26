"""Diagnosis generators. Rank, gate, and gaps come from ``diagnose_incident``."""

from __future__ import annotations

from principia_inferentiae.practice.dag import GradingType
from principia_inferentiae.practice.generators.base import text_problem
from principia_inferentiae.practice.generators.fixtures import (
    causal_incident,
    correlation_incident,
    diagnosis_of,
    missing_incident,
    rank_incident,
)
from principia_inferentiae.practice.generators.registry import register
from principia_inferentiae.practice.grader import Problem


class HypothesisRank:
    concept_id = "diagnosis.hypothesis_rank"

    def generate(self, seed: int, difficulty: float) -> Problem:
        diagnosis = diagnosis_of(rank_incident()).diagnosis
        order = ",".join(item.hypothesis_id for item in diagnosis.ranked_hypotheses)
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                "Three hypotheses: h-direct has a direct event, h-metric has a measured metric, "
                "h-corr has a correlation. Order them by evidence score, highest first, "
                "comma-separated."
            ),
            answer=order,
            facts={"oracle": "diagnosis_order"},
            choices=None,
            grading_type=GradingType.ORDER,
            order_threshold=1.0,
        )


class CausalGate:
    concept_id = "diagnosis.causal_gate"

    def generate(self, seed: int, difficulty: float) -> Problem:
        ranked = diagnosis_of(causal_incident()).diagnosis.ranked_hypotheses
        passed = tuple(item.hypothesis_id for item in ranked if item.causal_gate_passed)
        if len(passed) != 1:
            raise RuntimeError(passed)
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                "h-pass has one explicit hypothesis, a changed variable, the same workload hash, "
                "a different config hash, and a rerun whose metric moved in the predicted direction "
                "by more than twice the dispersion. h-miss has only a correlation. "
                "Which hypothesis passed the causal gate?"
            ),
            answer=passed[0],
            facts={"oracle": "diagnosis_gate"},
            choices=(
                ("h-pass", "h-pass"),
                ("h-miss", "h-miss"),
                ("neither", "neither"),
                ("both", "both"),
            ),
            grading_type=GradingType.CHOICE,
        )


class CorrelationVsCause:
    concept_id = "diagnosis.correlation_vs_cause"

    def generate(self, seed: int, difficulty: float) -> Problem:
        diagnosis = diagnosis_of(correlation_incident()).diagnosis
        top = diagnosis.ranked_hypotheses[0]
        if top.status != "correlated" or diagnosis.root_cause_hypothesis_id is not None:
            raise RuntimeError(top.status)
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                "The top hypothesis has supporting correlation and no controlled rerun. "
                "What may you call it?"
            ),
            answer="correlated",
            facts={"oracle": "diagnosis_status"},
            choices=(
                ("correlated", "Correlated, not a root cause"),
                ("confirmed", "Confirmed root cause"),
                ("unsupported", "Unsupported"),
                ("inconclusive", "Inconclusive because evidence conflicts"),
            ),
            grading_type=GradingType.CHOICE,
        )


class MissingTelemetry:
    concept_id = "diagnosis.missing_telemetry"

    def generate(self, seed: int, difficulty: float) -> Problem:
        diagnosis = diagnosis_of(missing_incident()).diagnosis
        if diagnosis.missing_telemetry != ("kv cache occupancy",):
            raise RuntimeError(diagnosis.missing_telemetry)
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                "Required telemetry is request latency and kv cache occupancy. "
                "The incident recorded request latency. Which telemetry is missing?"
            ),
            answer="kv cache occupancy",
            facts={"oracle": "diagnosis_missing"},
            choices=(
                ("kv cache occupancy", "kv cache occupancy"),
                ("request latency", "request latency"),
                ("ttft_ms", "ttft_ms"),
                ("none", "nothing is missing"),
            ),
            grading_type=GradingType.CHOICE,
        )


def _register_all() -> None:
    for generator in (
        HypothesisRank(),
        CausalGate(),
        CorrelationVsCause(),
        MissingTelemetry(),
    ):
        register(generator)


_register_all()
