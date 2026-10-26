"""Workload generators. Statistics come from ``characterize_workload``."""

from __future__ import annotations

import random

from principia_inferentiae.practice.generators.base import (
    numeric_problem,
    pick,
    steps_hidden,
)
from principia_inferentiae.practice.generators.fixtures import traces_from, workload_profile
from principia_inferentiae.practice.generators.registry import register
from principia_inferentiae.practice.grader import Problem


def _row(
    request_id: str,
    arrival: int,
    completed: int | None,
    prompt: int,
    output: int,
    tenant: str,
    prefix: str | None,
    shared: int,
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "arrival_time_ms": arrival,
        "completed_time_ms": completed,
        "prompt_tokens": prompt,
        "output_tokens": output,
        "tenant_id": tenant,
        "prefix_key": prefix,
        "shared_prefix_tokens": shared,
    }


def _prompt_rows(seed: int, difficulty: float) -> list[object]:
    rng = random.Random(seed)
    width = pick(seed, difficulty, (4, 6, 8))
    prompts = (128, 256, 512, 1024, 2048, 4096)
    outputs = (8, 32, 64, 128, 256)
    rows: list[object] = []
    for index in range(width):
        rows.append(
            _row(
                f"w{index}",
                index * 20,
                2_000 + index,
                prompts[rng.randrange(len(prompts))],
                outputs[rng.randrange(len(outputs))],
                "a",
                None,
                0,
            )
        )
    return rows


class PromptVsOutput:
    concept_id = "workload.prompt_vs_output"

    def generate(self, seed: int, difficulty: float) -> Problem:
        requests = _prompt_rows(seed, difficulty)
        profile = workload_profile(traces_from(requests))
        field = "output_tokens.p99" if steps_hidden(difficulty) else "prompt_tokens.p99"
        head, tail = field.split(".")
        value = float(getattr(getattr(profile, head), tail))
        kind = "output" if field.startswith("output") else "prompt"
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                "Prompt and output lengths stay separate. "
                f"Requests: {[(row['prompt_tokens'], row['output_tokens']) for row in requests if isinstance(row, dict)]}. "
                f"Nearest-rank p99 of {kind} tokens?"
            ),
            canonical=value,
            unit="tokens",
            facts={"oracle": "workload_field", "field": field, "requests": requests},
        )


class PeakConcurrency:
    concept_id = "workload.peak_concurrency"

    def generate(self, seed: int, difficulty: float) -> Problem:
        overlap = pick(seed, difficulty, (2, 3, 4))
        requests: list[object] = []
        for index in range(overlap):
            requests.append(_row(f"c{index}", index * 10, 5_000, 128, 16, "a", None, 0))
        requests.append(_row("later", 6_000, 7_000, 128, 16, "b", None, 0))
        profile = workload_profile(traces_from(requests))
        if profile.peak_observed_concurrency is None:
            raise RuntimeError("peak concurrency missing")
        intervals = [
            (row["arrival_time_ms"], row["completed_time_ms"])
            for row in requests
            if isinstance(row, dict)
        ]
        hint = (
            ""
            if steps_hidden(difficulty)
            else " Count overlapping half-open [arrival, completed) intervals; peak is the max active."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"Requests as [arrival_ms, completed_ms): {intervals}. "
                f"What is peak observed concurrency?{hint}"
            ),
            canonical=float(profile.peak_observed_concurrency),
            unit="count",
            facts={
                "oracle": "workload_field",
                "field": "peak_observed_concurrency",
                "requests": requests,
                "intervals": intervals,
            },
        )


class BurstVsSteady:
    concept_id = "workload.burst_vs_steady"

    def generate(self, seed: int, difficulty: float) -> Problem:
        burst = pick(seed, difficulty, (6, 8, 12))
        requests: list[object] = [
            _row(f"b{index}", 0, 100, 64, 8, "a", None, 0) for index in range(burst)
        ]
        requests.append(_row("tail", 5_000, 5_100, 64, 8, "a", None, 0))
        profile = workload_profile(traces_from(requests))
        window_ms = profile.observation_window_ms
        hint = (
            ""
            if steps_hidden(difficulty)
            else (
                " Burstiness is peak one-second arrivals divided by the mean arrival rate. "
                f"The observation window is last−first+1 ms (= {window_ms} ms here), "
                "so mean_rps = request_count / (window_ms/1000)."
            )
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"{burst} requests arrive at t=0 ms and one arrives at t=5000 ms "
                f"({burst + 1} requests, observation window {window_ms} ms). "
                f"What is burstiness (peak one-second / mean)?{hint}"
            ),
            canonical=float(profile.burstiness_peak_to_mean),
            unit="fraction",
            facts={
                "oracle": "workload_field",
                "field": "burstiness_peak_to_mean",
                "requests": requests,
                "observation_window_ms": window_ms,
            },
        )


class PrefixReuse:
    concept_id = "workload.prefix_reuse"

    def generate(self, seed: int, difficulty: float) -> Problem:
        shared = pick(seed, difficulty, (2, 3, 4))
        unique = pick(seed + 2, difficulty, (1, 2))
        requests: list[object] = []
        for index in range(shared):
            requests.append(_row(f"s{index}", index * 10, 1_000, 256, 16, "a", "sys", 128))
        for index in range(unique):
            requests.append(_row(f"u{index}", 100 + index, 1_000, 256, 16, "b", None, 0))
        profile = workload_profile(traces_from(requests))
        hint = (
            ""
            if steps_hidden(difficulty)
            else " Reuse fraction counts requests after the first one that shares a prefix key."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"{shared} requests share prefix key sys and {unique} are unique. "
                f"What is the prefix reuse fraction?{hint}"
            ),
            canonical=float(profile.prefix_reuse_fraction),
            unit="fraction",
            facts={"oracle": "workload_field", "field": "prefix_reuse_fraction", "requests": requests},
        )


class TenantMix:
    concept_id = "workload.tenant_mix"

    def generate(self, seed: int, difficulty: float) -> Problem:
        tenants = pick(seed, difficulty, ("a", "ab", "abc", "abcd"))
        requests: list[object] = [
            _row(f"t{index}", index * 10, 500, 64, 8, tenant, None, 0)
            for index, tenant in enumerate(tenants)
        ]
        profile = workload_profile(traces_from(requests))
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"Requests come from tenants {', '.join(tenants)}. How many tenants are in the trace?",
            canonical=float(profile.tenant_count),
            unit="count",
            facts={"oracle": "workload_field", "field": "tenant_count", "requests": requests},
        )


def _register_all() -> None:
    for generator in (
        PromptVsOutput(),
        PeakConcurrency(),
        BurstVsSteady(),
        PrefixReuse(),
        TenantMix(),
    ):
        register(generator)


_register_all()
