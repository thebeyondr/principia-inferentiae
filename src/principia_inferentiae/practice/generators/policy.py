"""Policy generators. A repair is correct when the simulator separates the bad clock."""

from __future__ import annotations

import copy

from principia_inferentiae.practice.dag import GradingType
from principia_inferentiae.practice.generators.base import band, steps_hidden, text_problem
from principia_inferentiae.practice.generators.policy_data import SCENARIOS
from principia_inferentiae.practice.generators.registry import register
from principia_inferentiae.practice.grader import Problem
from principia_inferentiae.practice.sim_access import metric, separates
from principia_inferentiae.sim import simulate

_KINDS = (
    ("policy.admission", "admit"),
    ("policy.queue", "queue"),
    ("policy.chunked_prefill", "chunk"),
    ("policy.retry_storm", "retry"),
    ("policy.goodput", "goodput"),
    ("policy.prefix", "prefix"),
    ("policy.priority", "priority"),
    ("policy.sticky", "sticky"),
    ("policy.hog", "hog"),
    ("policy.frag", "frag"),
    ("policy.spec", "spec"),
    ("policy.evict", "evict"),
    ("policy.lora", "lora"),
    ("policy.scale", "scale"),
    ("policy.batch", "batch"),
)
_TIME_KEYS = ("timeout_ms", "swap_ms", "warm_ms", "baseline_ms")


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected an object")
    return {str(key): item for key, item in value.items()}


def _scale(base: dict[str, object], scale: int) -> dict[str, object]:
    scenario = copy.deepcopy(base)
    for key in ("prefill_us_per_token", "decode_us_per_token"):
        value = scenario.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        scenario[key] = int(value) * scale
    slo = scenario.get("slo")
    if isinstance(slo, dict):
        scaled: dict[str, object] = {}
        for key, value in slo.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                scaled[str(key)] = value
            else:
                scaled[str(key)] = float(value) * scale
        scenario["slo"] = scaled
    for key in _TIME_KEYS:
        value = scenario.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        scenario[key] = value * scale
    return scenario


def _winners(scenario: dict[str, object], repairs: list[str]) -> list[str]:
    baseline = simulate(scenario, None)
    return [
        repair
        for repair in repairs
        if separates(scenario, baseline, simulate(scenario, repair))
    ]


def _clocks(scenario: dict[str, object], repairs: list[str]) -> str:
    baseline = simulate(scenario, None)
    parts = [
        "baseline "
        f"ttft={metric(baseline, 'ttft_p99_ms')} itl={metric(baseline, 'itl_p99_ms')}"
    ]
    for repair in repairs:
        result = simulate(scenario, repair)
        parts.append(
            f"{repair} ttft={metric(result, 'ttft_p99_ms')} "
            f"itl={metric(result, 'itl_p99_ms')} goodput={metric(result, 'goodput')}"
        )
    return "; ".join(parts)


class PolicyGenerator:
    def __init__(self, concept_id: str, kind: str) -> None:
        self.concept_id = concept_id
        self.kind = kind

    def generate(self, seed: int, difficulty: float) -> Problem:
        spec = _mapping(SCENARIOS[self.kind])
        base = _mapping(spec["scenario"])
        choice_rows = spec["choices"]
        if not isinstance(choice_rows, list):
            raise TypeError("choices")
        choices: list[tuple[str, str]] = []
        for row in choice_rows:
            item = _mapping(row)
            choice_id = item["id"]
            label = item["label"]
            if not isinstance(choice_id, str) or not isinstance(label, str):
                raise TypeError("choice")
            choices.append((choice_id, label))
        repairs = [choice_id for choice_id, _label in choices]
        lab_answer = spec["lab_answer"]
        question = spec["question"]
        if not isinstance(lab_answer, str) or not isinstance(question, str):
            raise TypeError("scenario text")
        scale = (1, 2, 4)[band(difficulty)]
        scenario = _scale(base, scale)
        winners = _winners(scenario, repairs)
        if not winners and scale != 1:
            scenario = _scale(base, 1)
            winners = _winners(scenario, repairs)
        if not winners:
            raise RuntimeError(f"{self.concept_id} no separating repair: {_clocks(scenario, repairs)}")
        answer = lab_answer if lab_answer in winners else winners[0]
        clock = ""
        if not steps_hidden(difficulty):
            field = scenario.get("baseline_field")
            if isinstance(field, str):
                clock = f" The broken clock is {field}."
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"{question}{clock}",
            answer=answer,
            facts={"scenario": scenario, "repairs": repairs, "lab_answer": lab_answer},
            choices=tuple(choices),
            grading_type=GradingType.INTERVENE,
        )


def _register_all() -> None:
    for concept_id, kind in _KINDS:
        register(PolicyGenerator(concept_id, kind))


_register_all()
