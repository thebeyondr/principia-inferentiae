"""Metric generators. Percentiles and rates come from the workload and sim oracles."""

from __future__ import annotations

import random

from principia_inferentiae.practice.dag import GradingType
from principia_inferentiae.practice.generators.base import (
    numeric_problem,
    pick,
    steps_hidden,
    text_problem,
)
from principia_inferentiae.practice.generators.fixtures import traces_from, workload_profile
from principia_inferentiae.practice.generators.registry import register
from principia_inferentiae.practice.grader import Problem
from principia_inferentiae.practice.sim_access import metric, run


def _token_requests(seed: int, difficulty: float) -> list[object]:
    rng = random.Random(seed)
    width = pick(seed, difficulty, (4, 6, 8))
    prompts = [128, 256, 512, 1024, 2048, 4096, 8192]
    outputs = [8, 16, 32, 64, 128, 256]
    rows: list[object] = []
    for index in range(width):
        rows.append(
            {
                "request_id": f"m{index}",
                "arrival_time_ms": index * 10,
                "completed_time_ms": 1_000 + index * 10,
                "prompt_tokens": prompts[rng.randrange(len(prompts))],
                "output_tokens": outputs[rng.randrange(len(outputs))],
                "tenant_id": "a",
                "prefix_key": None,
                "shared_prefix_tokens": 0,
            }
        )
    return rows


class Boundaries:
    concept_id = "metrics.boundaries"

    def generate(self, seed: int, difficulty: float) -> Problem:
        scenario = {
            "kind": "admit",
            "prefill_us_per_token": 20,
            "decode_us_per_token": 2000,
            "kv_slots": 1,
            "broken": "none",
            "slo": {"ttft_ms": 1_000_000.0, "itl_ms": 1_000_000.0},
            "arrivals": [
                {
                    "request_id": "b1",
                    "arrival_time_ms": 0,
                    "prompt_tokens": 500,
                    "output_tokens": 8,
                }
            ],
        }
        ttft = metric(run(scenario), "ttft_p99_ms")
        itl = metric(run(scenario), "itl_p99_ms")
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"One request reports TTFT p99 {ttft} ms and ITL p99 {itl} ms. "
                "Which boundary is first-token time?"
            ),
            answer="ttft",
            facts={"oracle": "definition", "scenario": scenario},
            choices=(
                ("ttft", "TTFT, request boundary to first token"),
                ("itl", "ITL, the gap between later tokens"),
                ("tpot", "TPOT, an average over output tokens"),
                ("e2e", "E2E, request boundary to the last token"),
            ),
            grading_type=GradingType.CHOICE,
        )


class PerUserAggregate:
    concept_id = "metrics.per_user_vs_aggregate"

    def generate(self, seed: int, difficulty: float) -> Problem:
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                "A packed batch raises total tokens/s while one user's inter-token gap gets worse. "
                "Which quantity rose?"
            ),
            answer="aggregate",
            facts={"oracle": "definition"},
            choices=(
                ("aggregate", "Aggregate tokens/s"),
                ("per_user", "Per-user tokens/s"),
                ("goodput", "SLO goodput, by definition"),
                ("ttft", "TTFT, because batching removes queue wait"),
            ),
            grading_type=GradingType.CHOICE,
        )


class NearestRank:
    concept_id = "metrics.nearest_rank_percentile"

    def generate(self, seed: int, difficulty: float) -> Problem:
        percentile = "p50" if not steps_hidden(difficulty) else "p99"
        requests = _token_requests(seed, difficulty)
        profile = workload_profile(traces_from(requests))
        value = float(getattr(profile.prompt_tokens, percentile))
        shown = ", ".join(str(row["prompt_tokens"]) for row in requests if isinstance(row, dict))
        hint = (
            ""
            if steps_hidden(difficulty)
            else " Nearest-rank index is ceil(percentile × n) − 1 on the sorted sample."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"Prompt token counts: {shown}. What is nearest-rank {percentile}?{hint}",
            canonical=value,
            unit="tokens",
            facts={
                "oracle": "workload_field",
                "field": f"prompt_tokens.{percentile}",
                "requests": requests,
            },
        )


class P50VsP99:
    concept_id = "metrics.p50_vs_p99"

    def generate(self, seed: int, difficulty: float) -> Problem:
        requests = _token_requests(seed, difficulty)
        profile = workload_profile(traces_from(requests))
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"Prompt tokens nearest-rank p50={profile.prompt_tokens.p50:g}, "
                f"p99={profile.prompt_tokens.p99:g}. Which one gates a tail SLO?"
            ),
            answer="p99",
            facts={"oracle": "definition", "requests": requests},
            choices=(
                ("p99", "p99, the tail"),
                ("p50", "p50, because it is the typical request"),
                ("mean", "The arithmetic mean of p50 and p99"),
                ("min", "The minimum, so the SLO never misses"),
            ),
            grading_type=GradingType.CHOICE,
        )


class Goodput:
    concept_id = "metrics.goodput"

    def generate(self, seed: int, difficulty: float) -> Problem:
        itl_limit = pick(seed, difficulty, (5.0, 10.0, 50.0))
        scenario = {
            "kind": "admit",
            "prefill_us_per_token": 20,
            "decode_us_per_token": 2000,
            "kv_slots": 4,
            "broken": "none",
            "slo": {"ttft_ms": 10_000.0, "itl_ms": itl_limit},
            "arrivals": [
                {
                    "request_id": f"g{index}",
                    "arrival_time_ms": index * 10,
                    "prompt_tokens": 200,
                    "output_tokens": 16,
                }
                for index in range(4)
            ],
        }
        value = metric(run(scenario), "goodput")
        if value is None:
            raise RuntimeError("simulator returned no goodput")
        hint = (
            ""
            if steps_hidden(difficulty)
            else " Goodput is the fraction of completed requests inside every declared SLO clock."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"Four requests, decode_us_per_token=2000, kv_slots=4, "
                f"ITL SLO {itl_limit:g} ms, TTFT SLO 10000 ms. "
                f"What fraction of completed requests are in SLO?{hint}"
            ),
            canonical=value,
            unit="fraction",
            facts={"oracle": "sim_fraction", "field": "goodput", "scenario": scenario},
        )


class InferenceVsE2e:
    concept_id = "metrics.inference_vs_e2e"

    def generate(self, seed: int, difficulty: float) -> Problem:
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                "Queue wait is 400 ms and the engine's own time-to-first-token is 12 ms. "
                "What is the 412 ms number?"
            ),
            answer="end_to_end",
            facts={"oracle": "definition", "queue_ms": 400, "inference_ttft_ms": 12},
            choices=(
                ("end_to_end", "End-to-end TTFT, queue plus inference"),
                ("inference_only", "Inference-only TTFT"),
                ("itl", "ITL"),
                ("goodput", "Goodput"),
            ),
            grading_type=GradingType.CLASSIFY,
        )


class LittlesLaw:
    concept_id = "metrics.littles_law"

    def generate(self, seed: int, difficulty: float) -> Problem:
        requests: list[object] = [
            {
                "request_id": "l1",
                "arrival_time_ms": 0,
                "completed_time_ms": 200,
                "prompt_tokens": 128,
                "output_tokens": 16,
                "tenant_id": "a",
                "prefix_key": None,
                "shared_prefix_tokens": 0,
            },
            {
                "request_id": "l2",
                "arrival_time_ms": 0,
                "completed_time_ms": 200,
                "prompt_tokens": 128,
                "output_tokens": 16,
                "tenant_id": "a",
                "prefix_key": None,
                "shared_prefix_tokens": 0,
            },
            {
                "request_id": "l3",
                "arrival_time_ms": 5_000,
                "completed_time_ms": 5_200,
                "prompt_tokens": 128,
                "output_tokens": 16,
                "tenant_id": "b",
                "prefix_key": None,
                "shared_prefix_tokens": 0,
            },
        ]
        profile = workload_profile(traces_from(requests))
        concurrency = profile.peak_observed_concurrency
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"The trace's mean arrival rate is {profile.mean_arrival_rps:g} requests/s "
                f"and peak observed concurrency is {concurrency}. "
                "Which equation is Little's law?"
            ),
            answer="l_eq_lambda_w",
            facts={
                "oracle": "definition",
                "requests": requests,
                "mean_arrival_rps": profile.mean_arrival_rps,
                "peak_observed_concurrency": concurrency,
            },
            choices=(
                ("l_eq_lambda_w", "L = λW"),
                ("l_over_w", "L = λ / W"),
                ("w_eq_lambda_l", "W = λL"),
                ("l_eq_w", "L = W / λ"),
            ),
            grading_type=GradingType.FORMULA,
        )


def _register_all() -> None:
    for generator in (
        Boundaries(),
        PerUserAggregate(),
        NearestRank(),
        P50VsP99(),
        Goodput(),
        InferenceVsE2e(),
        LittlesLaw(),
    ):
        register(generator)


_register_all()
