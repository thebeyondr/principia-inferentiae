"""Model-memory generators. Answers come from ``memory.py``."""

from __future__ import annotations

from principia_inferentiae.practice.grader import Problem
from principia_inferentiae.memory import estimate_memory, parameter_bytes, weight_shards
from principia_inferentiae.practice.generators.base import (
    numeric_problem,
    pick,
    steps_hidden,
)
from principia_inferentiae.practice.generators.fixtures import memory_case
from principia_inferentiae.practice.generators.registry import register


def _bits_prompt(bits: float, hidden: bool) -> str:
    if hidden:
        name = {16.0: "BF16", 8.0: "FP8", 4.0: "INT4"}.get(bits, f"{bits:g}-bit")
        return f"stored as {name}"
    return f"at {bits / 8:g} bytes per parameter"


class ParamsBytes:
    concept_id = "model.params_bytes"

    def generate(self, seed: int, difficulty: float) -> Problem:
        count = pick(seed, difficulty, (1_000_000, 1_000_000_000, 8_000_000_000, 70_000_000_000))
        bits = pick(seed + 1, difficulty, (16.0, 8.0, 4.0))
        knobs: dict[str, object] = {
            "preset": "fit_small",
            "parameter_count": count,
            "bits": bits,
            "metadata": 0.0,
        }
        _run, model, _hardware, _config = memory_case(knobs)
        value = float(parameter_bytes(count, model))
        hidden = steps_hidden(difficulty)
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"{count} parameters {_bits_prompt(bits, hidden)}. "
                "How many weight bytes, with no quantization metadata?"
            ),
            canonical=value,
            unit="B",
            facts={"oracle": "parameter_bytes", "knobs": knobs},
        )


class QuantMetadata:
    concept_id = "model.quant_metadata"

    def generate(self, seed: int, difficulty: float) -> Problem:
        count = pick(seed, difficulty, (100_000, 1_000_000, 8_000_000))
        metadata = pick(seed + 2, difficulty, (0.125, 0.25, 0.5))
        knobs: dict[str, object] = {
            "preset": "fit_small",
            "parameter_count": count,
            "bits": 4.0,
            "metadata": metadata,
            "group_size": 64,
        }
        run, model, hardware, config = memory_case(knobs)
        value = float(estimate_memory(run, model, hardware, config).estimate.quantization_metadata_total_bytes)
        hint = (
            ""
            if steps_hidden(difficulty)
            else " Metadata bytes = ceil(parameters × metadata bytes per parameter)."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"{count} parameters, {metadata:g} metadata bytes per parameter, group size 64. "
                f"How many quantization-metadata bytes?{hint}"
            ),
            canonical=value,
            unit="B",
            facts={
                "oracle": "memory_field",
                "field": "quantization_metadata_total_bytes",
                "knobs": knobs,
            },
        )


class GroupSize:
    concept_id = "model.group_size_overhead"

    def generate(self, seed: int, difficulty: float) -> Problem:
        count = pick(seed, difficulty, (64_000, 1_000_000, 4_000_000))
        group = pick(seed + 3, difficulty, (32, 64, 128))
        scale = pick(seed + 4, difficulty, (4.0, 8.0))
        knobs: dict[str, object] = {
            "preset": "fit_small",
            "parameter_count": count,
            "bits": 4.0,
            "scale_bytes": scale,
            "group_size": group,
        }
        run, model, hardware, config = memory_case(knobs)
        value = float(estimate_memory(run, model, hardware, config).estimate.quantization_metadata_total_bytes)
        if steps_hidden(difficulty):
            detail = f"{scale:g} scale bytes per group of {group}."
        else:
            detail = (
                f"{scale:g} scale bytes per group of {group}, so metadata bytes per parameter "
                f"is {scale:g}/{group}."
            )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"{count} parameters. {detail} How many metadata bytes in total?",
            canonical=value,
            unit="B",
            facts={
                "oracle": "memory_field",
                "field": "quantization_metadata_total_bytes",
                "knobs": knobs,
            },
        )


class WeightShardsTp:
    concept_id = "model.weight_shards_tp"

    def generate(self, seed: int, difficulty: float) -> Problem:
        preset = pick(seed, difficulty, ("tp4", "pp2", "replicated", "shard_layers"))
        knobs: dict[str, object] = {"preset": preset}
        _run, _model, _hardware, config = memory_case(knobs)
        value = float(weight_shards(config))
        shown = (
            f"weight_sharding={config.weight_sharding}, "
            f"TP={config.tensor_parallel_size}, PP={config.pipeline_parallel_size}."
        )
        hint = (
            ""
            if steps_hidden(difficulty)
            else " tensor_and_pipeline shards by TP×PP; pipeline_only by PP; replicated is 1."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"How many weight shards? {shown}{hint}",
            canonical=value,
            unit="shards",
            facts={"oracle": "weight_shards", "knobs": knobs},
        )


class PerRankWeights:
    concept_id = "model.per_rank_weight_bytes"

    def generate(self, seed: int, difficulty: float) -> Problem:
        preset = pick(seed, difficulty, ("fit_small", "tp4", "imbalance", "pp2"))
        knobs: dict[str, object] = {"preset": preset}
        run, model, hardware, config = memory_case(knobs)
        value = float(estimate_memory(run, model, hardware, config).estimate.weights_per_rank_bytes)
        hint = (
            ""
            if steps_hidden(difficulty)
            else " Per-rank weights include the declared sharding imbalance."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"parameter_count={model.parameter_count}, bits={model.weight_format.bits_per_weight:g}, "
                f"TP={config.tensor_parallel_size}, PP={config.pipeline_parallel_size}, "
                f"weight_sharding={config.weight_sharding}, "
                f"imbalance={config.sharding_imbalance_fraction:g}. "
                f"How many weight bytes on one rank?{hint}"
            ),
            canonical=value,
            unit="B",
            facts={"oracle": "memory_field", "field": "weights_per_rank_bytes", "knobs": knobs},
        )


def _register_all() -> None:
    for generator in (
        ParamsBytes(),
        QuantMetadata(),
        GroupSize(),
        WeightShardsTp(),
        PerRankWeights(),
    ):
        register(generator)


_register_all()
