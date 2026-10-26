"""Sizing generators. Counts come from ``size_deployment``."""

from __future__ import annotations

from principia_inferentiae.contracts import AnalysisStatus
from principia_inferentiae.practice.dag import GradingType
from principia_inferentiae.practice.generators.base import numeric_problem, steps_hidden, text_problem
from principia_inferentiae.practice.generators.fixtures import sizing_analysis
from principia_inferentiae.practice.generators.registry import register
from principia_inferentiae.practice.grader import Problem


class ServiceCurveIdentity:
    concept_id = "sizing.service_curve_identity"

    def generate(self, seed: int, difficulty: float) -> Problem:
        if steps_hidden(difficulty):
            mode = "wrong_scope"
            answer = "metric_scope"
            detail = "The measured curve's metric_scope is inference_only and the SLO is end_to_end."
        elif difficulty < 0.34:
            mode = "no_curve"
            answer = "missing_curve"
            detail = "No service curve was supplied."
        else:
            mode = "wrong_workload"
            answer = "workload_hash"
            detail = "The curve workload hash is not the profile workload hash."
        knobs: dict[str, object] = {"preset": "lower_bound", "curve": mode}
        analysis = sizing_analysis(knobs)
        if analysis.recommendation.recommended_gpu_count is not None:
            raise RuntimeError("identity failure still recommended a GPU count")
        if analysis.recommendation.status is not AnalysisStatus.INSUFFICIENT_EVIDENCE:
            raise RuntimeError(analysis.recommendation.status)
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"{detail} Which service-curve identity is missing?",
            answer=answer,
            facts={"oracle": "sizing_status", "knobs": knobs},
            choices=(
                ("missing_curve", "No workload- and config-matched measured curve"),
                ("workload_hash", "Workload hash does not match"),
                ("metric_scope", "Metric scope does not match"),
                ("quality", "The quality gate failed"),
            ),
            grading_type=GradingType.CHOICE,
        )


class MemoryFitLowerBound:
    concept_id = "sizing.memory_fit_lower_bound"

    def generate(self, seed: int, difficulty: float) -> Problem:
        knobs: dict[str, object] = {"preset": "lower_bound", "curve": "no_curve"}
        recommendation = sizing_analysis(knobs).recommendation
        if recommendation.memory_fit_lower_bound_gpus is None:
            raise RuntimeError("lower bound missing")
        hint = (
            ""
            if steps_hidden(difficulty)
            else " This is a memory-fit lower bound, not a SLO-goodput recommendation."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"How many GPUs is the memory-fit lower bound?{hint}",
            canonical=float(recommendation.memory_fit_lower_bound_gpus),
            unit="GPU",
            facts={
                "oracle": "sizing_field",
                "field": "memory_fit_lower_bound_gpus",
                "knobs": knobs,
            },
        )


class NullGpuCount:
    concept_id = "sizing.null_gpu_count"

    def generate(self, seed: int, difficulty: float) -> Problem:
        knobs: dict[str, object] = {"preset": "lower_bound", "curve": "no_curve"}
        recommendation = sizing_analysis(knobs).recommendation
        lower = recommendation.memory_fit_lower_bound_gpus
        if recommendation.recommended_gpu_count is not None or lower is None:
            raise RuntimeError("expected a null recommendation with a lower bound")
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"Memory-fit lower bound is {lower} GPU(s) and no matched service curve exists. "
                "What is recommended_gpu_count?"
            ),
            answer="null",
            facts={"oracle": "sizing_status", "knobs": knobs, "lower_bound": lower},
            choices=(
                ("null", "null"),
                (str(lower), f"{lower}, the memory-fit lower bound"),
                ("1", "1, one replica"),
                ("8", "8, one full node"),
            ),
            grading_type=GradingType.CHOICE,
        )


class BenchmarkPlan:
    concept_id = "sizing.benchmark_plan"

    def generate(self, seed: int, difficulty: float) -> Problem:
        matched = steps_hidden(difficulty)
        knobs: dict[str, object] = {
            "preset": "lower_bound",
            "curve": "matched" if matched else "no_curve",
        }
        analysis = sizing_analysis(knobs)
        status = analysis.benchmark_plan.status
        if matched and analysis.recommendation.status is not AnalysisStatus.SUPPORTED:
            raise RuntimeError(analysis.recommendation.status)
        if not matched and status != "required":
            raise RuntimeError(status)
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                "A matched measured curve already meets the SLO."
                if matched
                else "Memory fits, but there is no matched measured service curve."
            )
            + " What is the benchmark plan status?",
            answer=status,
            facts={"oracle": "benchmark_status", "knobs": knobs},
            choices=(
                ("required", "required"),
                ("not_required", "not_required"),
            ),
            grading_type=GradingType.CHOICE,
        )


def _register_all() -> None:
    for generator in (
        ServiceCurveIdentity(),
        MemoryFitLowerBound(),
        NullGpuCount(),
        BenchmarkPlan(),
    ):
        register(generator)


_register_all()
