"""Generator self-check and numeric recomputation. One loop, not one case per seed."""

from __future__ import annotations

import os

from principia_inferentiae.memory import (
    estimate_memory,
    kv_layout,
    parameter_bytes,
    weight_shards,
)
from principia_inferentiae.practice.dag import GradingType
from principia_inferentiae.practice.generators import GENERATORS, self_check
from principia_inferentiae.practice.generators.fixtures import (
    memory_case,
    roofline_of,
    sizing_analysis,
    traces_from,
    workload_profile,
)
from principia_inferentiae.practice.grader import Problem, grade, to_canonical
from principia_inferentiae.practice.sim_access import metric, run


def _profile_value(field: str, requests: object) -> float:
    if not isinstance(requests, list):
        raise TypeError("requests")
    profile = workload_profile(traces_from(requests))
    if "." in field:
        head, tail = field.split(".", 1)
        value = getattr(getattr(profile, head), tail)
    else:
        value = getattr(profile, field)
    if value is None:
        raise AssertionError(field)
    return float(value)


def recompute(facts: dict[str, object]) -> float:
    oracle = facts["oracle"]
    if oracle == "scale":
        return float(facts["count"]) * float(facts["multiplier"])
    if oracle == "flop_rate":
        return float(facts["flops"]) / (float(facts["duration_ms"]) / 1000.0)
    if oracle == "byte_rate":
        return float(facts["bytes"]) / (float(facts["duration_ms"]) / 1000.0)
    if oracle == "invert_to_tps":
        return 1000.0 / float(facts["ms_per_token"])
    if oracle == "invert_to_ms":
        return 1.0 / float(facts["tokens_per_s"])
    knobs = facts.get("knobs")
    if not isinstance(knobs, dict):
        knobs = {}
    typed = {str(key): value for key, value in knobs.items()}
    if oracle == "parameter_bytes":
        _run, model, _hardware, _config = memory_case(typed)
        return float(parameter_bytes(model.parameter_count, model))
    if oracle == "memory_field":
        run_meta, model, hardware, config = memory_case(typed)
        estimate = estimate_memory(run_meta, model, hardware, config).estimate
        return float(getattr(estimate, str(facts["field"])))
    if oracle == "weight_shards":
        _run, _model, _hardware, config = memory_case(typed)
        return float(weight_shards(config))
    if oracle == "kv_layout":
        _run, model, _hardware, config = memory_case(typed)
        layers, heads = kv_layout(model, config)
        return float(heads if facts["component"] == "heads" else layers)
    if oracle == "workload_field":
        return _profile_value(str(facts["field"]), facts["requests"])
    if oracle == "roofline_field":
        metrics = roofline_of(typed).metrics
        value = getattr(metrics, str(facts["field"]))
        if value is None:
            raise AssertionError(facts["field"])
        return float(value)
    if oracle == "sizing_field":
        recommendation = sizing_analysis(typed).recommendation
        value = getattr(recommendation, str(facts["field"]))
        if value is None:
            raise AssertionError(facts["field"])
        return float(value)
    if oracle == "sim_ms":
        scenario = facts["scenario"]
        if not isinstance(scenario, dict):
            raise TypeError("scenario")
        value = metric(run({str(key): item for key, item in scenario.items()}), str(facts["field"]))
        if value is None:
            raise AssertionError(facts["field"])
        return value / 1000.0
    if oracle == "sim_fraction":
        scenario = facts["scenario"]
        if not isinstance(scenario, dict):
            raise TypeError("scenario")
        value = metric(run({str(key): item for key, item in scenario.items()}), str(facts["field"]))
        if value is None:
            raise AssertionError(facts["field"])
        return value
    raise AssertionError(f"unknown oracle {oracle}")


def _assert_recomputed(problem: Problem) -> None:
    if problem.grading_type is not GradingType.NUMERIC_UNITS:
        return
    if problem.unit is None:
        raise AssertionError(problem.concept_id)
    _dimension, got = to_canonical(float(problem.answer), problem.unit)
    expected = recompute(problem.facts)
    if expected == 0:
        assert got == 0
        return
    assert abs(got - expected) / abs(expected) <= problem.tolerance


def test_self_check() -> None:
    self_check()
    for concept_id, generator in GENERATORS.items():
        problem = generator.generate(seed=0, difficulty=0.5)
        assert problem.concept_id == concept_id
        assert grade(problem, problem.answer).correct
        assert not grade(problem, "").correct
        _assert_recomputed(problem)


def test_fuzz_generators() -> None:
    seeds = int(os.environ.get("PRINCIPIA_FUZZ_SEEDS", "50"))
    assert seeds >= 1
    for generator in GENERATORS.values():
        for seed in range(seeds):
            difficulty = 0.5 if seed == 0 else (seed % 100) / 100
            problem = generator.generate(seed=seed, difficulty=difficulty)
            graded = grade(problem, problem.answer)
            assert graded.correct, problem.concept_id
            assert not grade(problem, "").correct
            _assert_recomputed(problem)


def test_registry_covers_the_graph() -> None:
    from helpers import EXPECTED_IDS

    assert set(GENERATORS) == EXPECTED_IDS
