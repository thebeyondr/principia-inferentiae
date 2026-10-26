"""KV-cache generators. Numeric answers come from ``memory.py``."""

from __future__ import annotations

from principia_inferentiae.practice.grader import Problem
from principia_inferentiae.memory import estimate_memory, kv_layout
from principia_inferentiae.practice.dag import GradingType
from principia_inferentiae.practice.generators.base import (
    numeric_problem,
    pick,
    steps_hidden,
    text_problem,
)
from principia_inferentiae.practice.generators.fixtures import memory_case
from principia_inferentiae.practice.generators.registry import register

_LAYOUT_CHOICES = (
    ("replicate_heads", "shard_layers keeps every KV head on the rank and only splits layers"),
    ("split_heads", "shard_layers divides KV heads across TP and copies layers"),
    ("replicate_layers", "pipeline parallelism copies every layer onto every rank"),
    ("drop_heads", "MQA deletes KV heads when TP is greater than 1"),
)


class BytesPerToken:
    concept_id = "kv.bytes_per_token"

    def generate(self, seed: int, difficulty: float) -> Problem:
        preset = pick(seed, difficulty, ("fit_small", "gqa", "mha", "mqa"))
        knobs: dict[str, object] = {"preset": preset}
        run, model, hardware, config = memory_case(knobs)
        value = float(estimate_memory(run, model, hardware, config).estimate.kv_bytes_per_token_logical)
        if steps_hidden(difficulty):
            identity = (
                f"{model.attention_type} with {model.attention_heads} attention heads "
                f"and {model.kv_heads} KV heads"
            )
        else:
            identity = (
                f"layers={model.layer_count}, kv_heads={model.kv_heads}, "
                f"head_dim={model.head_dim}, kv_dtype_bytes={model.kv_dtype_bytes:g}. "
                "Logical bytes/token = 2 × layers × kv_heads × head_dim × dtype_bytes."
            )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"Logical KV bytes for one token. {identity}",
            canonical=value,
            unit="B",
            facts={"oracle": "memory_field", "field": "kv_bytes_per_token_logical", "knobs": knobs},
        )


class AttentionKind:
    concept_id = "kv.mha_gqa_mqa"

    def generate(self, seed: int, difficulty: float) -> Problem:
        preset = pick(seed, difficulty, ("mha", "gqa", "mqa"))
        knobs: dict[str, object] = {"preset": preset}
        _run, model, _hardware, _config = memory_case(knobs)
        if steps_hidden(difficulty):
            prompt = (
                f"attention_heads={model.attention_heads}, kv_heads={model.kv_heads}. "
                "Which attention type is this?"
            )
        else:
            prompt = (
                f"attention_heads={model.attention_heads}, kv_heads={model.kv_heads}. "
                "MHA means kv_heads = attention_heads, MQA means kv_heads = 1, "
                "otherwise GQA. Which type is this?"
            )
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            answer=model.attention_type,
            facts={"oracle": "attention_type", "knobs": knobs},
            choices=(
                ("MHA", "MHA"),
                ("GQA", "GQA"),
                ("MQA", "MQA"),
            ),
            grading_type=GradingType.CLASSIFY,
        )


class TpPpSplit:
    concept_id = "kv.tp_pp_split"

    def generate(self, seed: int, difficulty: float) -> Problem:
        preset = pick(seed, difficulty, ("tp4", "pp2", "gqa", "shard_layers"))
        component = "heads" if steps_hidden(difficulty) else "layers"
        knobs: dict[str, object] = {"preset": preset}
        _run, model, _hardware, config = memory_case(knobs)
        layers, heads = kv_layout(model, config)
        value = float(heads if component == "heads" else layers)
        if component == "heads":
            ask = "How many KV heads are stored on one rank?"
        else:
            ask = "How many layers are stored on one rank?"
        hint = (
            ""
            if steps_hidden(difficulty)
            else " PP divides layers. shard_kv_heads divides KV heads when they divide TP."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"{ask} layers={model.layer_count}, kv_heads={model.kv_heads}, "
                f"TP={config.tensor_parallel_size}, PP={config.pipeline_parallel_size}, "
                f"kv_sharding={config.kv_sharding}.{hint}"
            ),
            canonical=value,
            unit="count",
            facts={"oracle": "kv_layout", "component": component, "knobs": knobs},
        )


class ShardLayers:
    concept_id = "kv.shard_layers_replicate"

    def generate(self, seed: int, difficulty: float) -> Problem:
        sharded = memory_case({"preset": "tp4"})
        layered = memory_case({"preset": "shard_layers"})
        sharded_heads = kv_layout(sharded[1], sharded[3])[1]
        layered_heads = kv_layout(layered[1], layered[3])[1]
        prompt = (
            f"Same 32 layers and 8 KV heads, TP=4. "
            f"shard_kv_heads stores {sharded_heads} KV heads per rank. "
            f"shard_layers stores {layered_heads} KV heads per rank. "
            "What did shard_layers do?"
        )
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            answer="replicate_heads",
            facts={
                "oracle": "kv_layout_compare",
                "sharded_heads": sharded_heads,
                "layered_heads": layered_heads,
            },
            choices=_LAYOUT_CHOICES,
            grading_type=GradingType.CHOICE,
        )


class ActiveTokens:
    concept_id = "kv.active_tokens"

    def generate(self, seed: int, difficulty: float) -> Problem:
        sequences = pick(seed, difficulty, (2, 4, 8, 16))
        length = pick(seed + 6, difficulty, (1024, 2048, 4096, 8192))
        knobs: dict[str, object] = {
            "preset": "fit_small",
            "max_concurrent_sequences": sequences,
            "max_sequence_tokens": length,
        }
        run, model, hardware, config = memory_case(knobs)
        value = float(estimate_memory(run, model, hardware, config).estimate.active_tokens_configured)
        hint = (
            ""
            if steps_hidden(difficulty)
            else " Active tokens = max concurrent sequences × max sequence tokens."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"max_concurrent_sequences={sequences}, max_sequence_tokens={length}. "
                f"How many active tokens are configured?{hint}"
            ),
            canonical=value,
            unit="tokens",
            facts={"oracle": "memory_field", "field": "active_tokens_configured", "knobs": knobs},
        )


class PerRankFit:
    concept_id = "kv.per_rank_fit"

    def generate(self, seed: int, difficulty: float) -> Problem:
        preset = pick(seed, difficulty, ("fit_small", "gqa", "tp4"))
        knobs: dict[str, object] = {"preset": preset}
        run, model, hardware, config = memory_case(knobs)
        estimate = estimate_memory(run, model, hardware, config).estimate
        hint = (
            ""
            if steps_hidden(difficulty)
            else " Required bytes add weights, KV, runtime reserve, then fragmentation."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"Per-rank required bytes for preset inputs "
                f"(reserve={config.runtime_reserve.runtime_fixed_bytes_per_rank}, "
                f"fragmentation={config.runtime_reserve.fragmentation_fraction:g}, "
                f"safety_margin={config.runtime_reserve.safety_margin_fraction:g}, "
                f"device={hardware.device_memory_bytes}).{hint}"
            ),
            canonical=float(estimate.required_per_rank_bytes),
            unit="B",
            facts={"oracle": "memory_field", "field": "required_per_rank_bytes", "knobs": knobs},
        )


class RankOom:
    concept_id = "kv.rank_oom"

    def generate(self, seed: int, difficulty: float) -> Problem:
        knobs: dict[str, object] = {"preset": "oom"}
        run, model, hardware, config = memory_case(knobs)
        estimate = estimate_memory(run, model, hardware, config).estimate
        if estimate.fits_per_rank:
            raise RuntimeError("oom preset unexpectedly fits")
        prompt = (
            f"Weights alone are {estimate.weights_total_bytes} bytes and the device has "
            f"{hardware.device_memory_bytes} bytes before margin. "
            f"aggregate_capacity_lower_bound_gpus={estimate.aggregate_capacity_lower_bound_gpus}. "
            "What is the per-rank fit status?"
        )
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            answer=estimate.status.value,
            facts={"oracle": "memory_status", "knobs": knobs},
            choices=(
                ("supported", "The replica fits on every rank"),
                ("memory_not_fit", "At least one rank does not fit"),
                ("insufficient_evidence", "Fit cannot be decided from the inputs"),
                ("slo_not_met", "The SLO failed, so memory fit is unknown"),
            ),
            grading_type=GradingType.CLASSIFY,
        )


def _register_all() -> None:
    for generator in (
        BytesPerToken(),
        AttentionKind(),
        TpPpSplit(),
        ShardLayers(),
        ActiveTokens(),
        PerRankFit(),
        RankOom(),
    ):
        register(generator)


_register_all()
