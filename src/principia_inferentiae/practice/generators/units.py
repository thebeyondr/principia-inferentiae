"""Unit-algebra generators. These do not call the analysis core."""

from __future__ import annotations

from principia_inferentiae.practice.grader import Problem
from principia_inferentiae.practice.generators.base import (
    numeric_problem,
    pick,
    steps_hidden,
)
from principia_inferentiae.practice.generators.registry import register


class BytesPrefixes:
    concept_id = "units.bytes_prefixes"

    def generate(self, seed: int, difficulty: float) -> Problem:
        count = pick(seed, difficulty, (1, 2, 4, 8, 16))
        if steps_hidden(difficulty):
            multiplier = 1_000_000_000
            label = "GB (decimal SI, not GiB)"
            hint = "Decimal GB is 10^9 bytes. GiB is a different prefix."
        elif difficulty < 0.34:
            multiplier = 1024
            label = "KiB"
            hint = "1 KiB = 1024 bytes."
        else:
            multiplier = 1024**3
            label = "GiB"
            hint = "1 GiB = 1024^3 bytes."
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"How many bytes are in {count} {label}? {hint}",
            canonical=float(count * multiplier),
            unit="B",
            facts={"oracle": "scale", "count": count, "multiplier": multiplier},
        )


class Microseconds:
    concept_id = "units.us_ms_s"

    def generate(self, seed: int, difficulty: float) -> Problem:
        microseconds = pick(seed, difficulty, (250, 2_500, 250_000, 2_500_000))
        seconds = microseconds * 1e-6
        if steps_hidden(difficulty):
            unit = "s"
            hint = "Convert microseconds to seconds without stopping at milliseconds."
        else:
            unit = "ms"
            hint = "1000 µs = 1 ms."
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"{microseconds} µs is how many {unit}? {hint}",
            canonical=seconds,
            unit=unit,
            facts={"oracle": "scale", "count": microseconds, "multiplier": 1e-6},
        )


class Tflops:
    concept_id = "units.tflops"

    def generate(self, seed: int, difficulty: float) -> Problem:
        flops = pick(seed, difficulty, (1e12, 2e12, 5e12, 2e13))
        duration_ms = pick(seed + 3, difficulty, (5.0, 10.0, 20.0, 50.0))
        flop_per_s = flops / (duration_ms / 1000.0)
        hint = (
            "TFLOP/s = FLOPs / seconds / 10^12."
            if not steps_hidden(difficulty)
            else "Report TFLOP/s. Seconds are not the number written in milliseconds."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"A kernel does {flops:.0f} FLOPs in {duration_ms:g} ms. {hint}",
            canonical=flop_per_s,
            unit="TFLOP/s",
            facts={"oracle": "flop_rate", "flops": flops, "duration_ms": duration_ms},
        )


class Gbps:
    concept_id = "units.gbps"

    def generate(self, seed: int, difficulty: float) -> Problem:
        byte_count = pick(seed, difficulty, (1_000_000_000, 5_000_000_000, 8 * 1024**3))
        duration_ms = pick(seed + 5, difficulty, (100.0, 250.0, 500.0, 1000.0))
        bytes_per_s = byte_count / (duration_ms / 1000.0)
        hint = (
            "Decimal GB/s = bytes/second / 10^9. This is not GiB/s."
            if not steps_hidden(difficulty)
            else "The asked unit is decimal GB/s."
        )
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=f"{byte_count:.0f} bytes move in {duration_ms:g} ms. {hint}",
            canonical=bytes_per_s,
            unit="GB/s",
            facts={"oracle": "byte_rate", "bytes": byte_count, "duration_ms": duration_ms},
        )


class TokenInversion:
    concept_id = "units.tokens_per_s_inversion"

    def generate(self, seed: int, difficulty: float) -> Problem:
        if steps_hidden(difficulty):
            tokens_per_s = pick(seed, difficulty, (20.0, 25.0, 40.0, 50.0))
            canonical = (1000.0 / tokens_per_s) / 1000.0
            unit = "ms/token"
            prompt = (
                f"Steady generation is {tokens_per_s:g} tokens/s. "
                "How many milliseconds per token is that?"
            )
            facts: dict[str, object] = {
                "oracle": "invert_to_ms",
                "tokens_per_s": tokens_per_s,
            }
        else:
            ms_per_token = pick(seed, difficulty, (10.0, 20.0, 25.0, 40.0))
            canonical = 1000.0 / ms_per_token
            unit = "tokens/s"
            prompt = f"{ms_per_token:g} ms/token is how many tokens/s? tokens/s = 1000 / ms/token."
            facts = {"oracle": "invert_to_tps", "ms_per_token": ms_per_token}
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            canonical=canonical,
            unit=unit,
            facts=facts,
        )


def _register_all() -> None:
    for generator in (
        BytesPrefixes(),
        Microseconds(),
        Tflops(),
        Gbps(),
        TokenInversion(),
    ):
        register(generator)


_register_all()
