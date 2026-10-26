"""Phase generators. Numeric clocks come from ``simulate``."""

from __future__ import annotations

from principia_inferentiae.practice.dag import GradingType
from principia_inferentiae.practice.generators.base import (
    numeric_problem,
    pick,
    steps_hidden,
    text_problem,
)
from principia_inferentiae.practice.generators.registry import register
from principia_inferentiae.practice.grader import Problem
from principia_inferentiae.practice.sim_access import metric, run


def _scenario(
    prompt_tokens: int,
    prefill_us: int,
    decode_us: int,
    arrivals: int,
    kv_slots: int,
) -> dict[str, object]:
    rows: list[object] = []
    for index in range(arrivals):
        rows.append(
            {
                "request_id": f"p{index}",
                "arrival_time_ms": index * 15,
                "prompt_tokens": prompt_tokens if index == 0 else max(32, prompt_tokens // 10),
                "output_tokens": 8,
            }
        )
    return {
        "kind": "admit",
        "prefill_us_per_token": prefill_us,
        "decode_us_per_token": decode_us,
        "kv_slots": kv_slots,
        "broken": "none",
        "slo": {"ttft_ms": 1_000_000.0, "itl_ms": 1_000_000.0},
        "arrivals": rows,
    }


def _milliseconds(scenario: dict[str, object], field: str) -> float:
    value = metric(run(scenario), field)
    if value is None:
        raise RuntimeError(f"simulator returned no {field}")
    return value


class PrefillMs:
    concept_id = "phases.prefill_ms"

    def generate(self, seed: int, difficulty: float) -> Problem:
        tokens = pick(seed, difficulty, (128, 512, 2048, 8192))
        prefill_us = pick(seed + 1, difficulty, (20, 40, 80))
        scenario = _scenario(tokens, prefill_us, 0, 1, 4)
        # A zero decode gap keeps TTFT equal to prefill for a one-request admission.
        value_ms = _milliseconds(scenario, "ttft_p99_ms")
        hint = (
            ""
            if steps_hidden(difficulty)
            else " With one request and a zero decode gap, TTFT is prompt_tokens × prefill_us / 1000."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"One request, {tokens} prompt tokens, prefill_us_per_token={prefill_us}, "
                f"decode_us_per_token=0, kv_slots=4. TTFT p99 in ms?{hint}"
            ),
            canonical=value_ms / 1000.0,
            unit="ms",
            facts={"oracle": "sim_ms", "field": "ttft_p99_ms", "scenario": scenario},
        )


class DecodeGap:
    concept_id = "phases.decode_gap"

    def generate(self, seed: int, difficulty: float) -> Problem:
        decode_us = pick(seed, difficulty, (1000, 2000, 4000, 8000))
        scenario = _scenario(32, 20, decode_us, 1, 1)
        value_ms = _milliseconds(scenario, "itl_p99_ms")
        hint = (
            ""
            if steps_hidden(difficulty)
            else " One live request: ITL p99 is decode_us_per_token / 1000."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"One request, decode_us_per_token={decode_us}, kv_slots=1. "
                f"What is ITL p99 in ms?{hint}"
            ),
            canonical=value_ms / 1000.0,
            unit="ms",
            facts={"oracle": "sim_ms", "field": "itl_p99_ms", "scenario": scenario},
        )


class TtftQueue:
    concept_id = "phases.ttft_queue_prefill"

    def generate(self, seed: int, difficulty: float) -> Problem:
        tokens = pick(seed, difficulty, (200, 400, 800))
        scenario = _scenario(tokens, 50, 1000, 4, 1)
        value_ms = _milliseconds(scenario, "ttft_p99_ms")
        hint = (
            ""
            if steps_hidden(difficulty)
            else " kv_slots=1, so later requests wait. TTFT is queue wait plus prefill, not prefill alone."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"Four arrivals, first prompt {tokens} tokens, prefill_us_per_token=50, "
                f"decode_us_per_token=1000, kv_slots=1. TTFT p99 in ms?{hint}"
            ),
            canonical=value_ms / 1000.0,
            unit="ms",
            facts={"oracle": "sim_ms", "field": "ttft_p99_ms", "scenario": scenario},
        )


class ItlVsTpot:
    concept_id = "phases.itl_vs_tpot"

    def generate(self, seed: int, difficulty: float) -> Problem:
        scenario = _scenario(128, 20, 2000, 1, 1)
        itl_ms = _milliseconds(scenario, "itl_p99_ms")
        prompt = (
            f"A one-request run reports ITL p99 = {itl_ms:g} ms. "
            "Which statement keeps ITL and TPOT distinct?"
        )
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            answer="distribution",
            facts={"oracle": "definition", "itl_ms": itl_ms},
            choices=(
                ("distribution", "ITL is the distribution of token gaps; TPOT is one average"),
                ("same", "ITL and TPOT are the same sample"),
                ("ttft", "TPOT is time to first token"),
                ("queue", "ITL includes admission-queue wait by definition"),
            ),
            grading_type=GradingType.CHOICE,
        )


class ChunkedPrefill:
    concept_id = "phases.chunked_prefill"

    def generate(self, seed: int, difficulty: float) -> Problem:
        scenario = _scenario(32000, 20, 2000, 4, 8)
        scenario["kind"] = "chunk"
        scenario["chunk_tokens"] = 2048
        baseline = _milliseconds(scenario, "itl_p99_ms")
        hint = "" if steps_hidden(difficulty) else f" Baseline ITL p99 is {baseline:g} ms before the repair."
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                "One long prefill is blocking short decodes. "
                f"What does chunked prefill change?{hint}"
            ),
            answer="interleave",
            facts={"oracle": "definition", "scenario": scenario},
            choices=(
                ("interleave", "Same prefill work, but decode can run between chunks"),
                ("free", "It deletes the long prompt so shorts never wait"),
                ("itl_up", "It improves TTFT of the long request and slows every short ITL"),
                ("bytes", "It shrinks KV bytes per token"),
            ),
            grading_type=GradingType.CHOICE,
        )


class Disaggregation:
    concept_id = "phases.disaggregation"

    def generate(self, seed: int, difficulty: float) -> Problem:
        prefill = _scenario(8192, 80, 500, 1, 1)
        decode = _scenario(128, 20, 8000, 1, 8)
        prefill_ttft = _milliseconds(prefill, "ttft_p99_ms")
        decode_itl = _milliseconds(decode, "itl_p99_ms")
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"A long prompt shows TTFT p99 {prefill_ttft:g} ms. "
                f"A short-prompt batch shows ITL p99 {decode_itl:g} ms. "
                "When is disaggregated prefill/decode the right split?"
            ),
            answer="different_hardware",
            facts={"oracle": "definition", "prefill_ttft_ms": prefill_ttft, "decode_itl_ms": decode_itl},
            choices=(
                (
                    "different_hardware",
                    "When prefill and decode want different batch shapes or hardware",
                ),
                ("always", "Whenever TTFT and ITL are both measured"),
                ("never", "Only when the model has more than 8 KV heads"),
                ("kv", "When the KV cache does not fit, regardless of phase"),
            ),
            grading_type=GradingType.CHOICE,
        )


def _register_all() -> None:
    for generator in (
        PrefillMs(),
        DecodeGap(),
        TtftQueue(),
        ItlVsTpot(),
        ChunkedPrefill(),
        Disaggregation(),
    ):
        register(generator)


_register_all()
