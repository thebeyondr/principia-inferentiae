"""Roofline generators. Classification comes from ``analyze_roofline``."""

from __future__ import annotations

from principia_inferentiae.contracts import BoundClassification
from principia_inferentiae.practice.dag import GradingType
from principia_inferentiae.practice.generators.base import numeric_problem, pick, text_problem
from principia_inferentiae.practice.generators.fixtures import roofline_of
from principia_inferentiae.practice.generators.registry import register
from principia_inferentiae.practice.grader import Problem

_COMPUTE: dict[str, object] = {"flops": 2e11, "hbm_bytes": 1e9, "duration_ms": 2.5}
_HBM: dict[str, object] = {"flops": 4e10, "hbm_bytes": 1e9, "duration_ms": 1.25}
_UNDER: dict[str, object] = {
    "flops": 2e11,
    "hbm_bytes": 1e9,
    "duration_ms": 20.0,
    "utilization": 0.99,
}
_MIXED: dict[str, object] = {"flops": 1e11, "hbm_bytes": 1e9, "duration_ms": 1.25}
_CLASSES = (
    ("compute_bound", "compute_bound"),
    ("hbm_bandwidth_bound", "hbm_bandwidth_bound"),
    ("underfilled", "underfilled"),
    ("mixed", "mixed"),
    ("insufficient_evidence", "insufficient_evidence"),
)


class ArithmeticIntensity:
    concept_id = "roofline.arithmetic_intensity"

    def generate(self, seed: int, difficulty: float) -> Problem:
        knobs = pick(seed, difficulty, (_COMPUTE, _HBM, _MIXED))
        metrics = roofline_of(knobs).metrics
        if metrics.arithmetic_intensity_flops_per_byte is None:
            raise RuntimeError("arithmetic intensity missing")
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"flops={knobs['flops']}, hbm_bytes={knobs['hbm_bytes']}. "
                "What is arithmetic intensity in FLOP/byte?"
            ),
            canonical=float(metrics.arithmetic_intensity_flops_per_byte),
            unit="FLOP/byte",
            facts={"oracle": "roofline_field", "field": "arithmetic_intensity_flops_per_byte", "knobs": knobs},
        )


class RidgePoint:
    concept_id = "roofline.ridge_point"

    def generate(self, seed: int, difficulty: float) -> Problem:
        tflops = pick(seed, difficulty, (50.0, 100.0, 200.0))
        bandwidth = pick(seed + 1, difficulty, (500.0, 1000.0, 2000.0))
        knobs: dict[str, object] = {**_COMPUTE, "empirical_tflops": tflops, "bandwidth_gbps": bandwidth}
        metrics = roofline_of(knobs).metrics
        if metrics.machine_balance_flops_per_byte is None:
            raise RuntimeError("ridge point missing")
        return numeric_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"Empirical compute ceiling {tflops:g} TFLOP/s, "
                f"HBM bandwidth {bandwidth:g} GB/s. "
                "What is the ridge point (machine balance) in FLOP/byte?"
            ),
            canonical=float(metrics.machine_balance_flops_per_byte),
            unit="FLOP/byte",
            facts={"oracle": "roofline_field", "field": "machine_balance_flops_per_byte", "knobs": knobs},
        )


class Classify:
    concept_id = "roofline.classify"

    def generate(self, seed: int, difficulty: float) -> Problem:
        knobs = pick(seed, difficulty, (_COMPUTE, _HBM, _UNDER, _MIXED))
        metrics = roofline_of(knobs).metrics
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"flops={knobs['flops']}, hbm_bytes={knobs['hbm_bytes']}, "
                f"duration_ms={knobs['duration_ms']}, ceiling 100 TFLOP/s, "
                "bandwidth 1000 GB/s. What is the roofline class?"
            ),
            answer=metrics.classification.value,
            facts={"oracle": "roofline_class", "knobs": knobs},
            choices=_CLASSES,
            grading_type=GradingType.CLASSIFY,
        )


class CapacityVsBandwidth:
    concept_id = "roofline.capacity_vs_bandwidth"

    def generate(self, seed: int, difficulty: float) -> Problem:
        knobs: dict[str, object] = {**_HBM, "capacity_used_bytes": 40 * 1024**3}
        metrics = roofline_of(knobs).metrics
        if metrics.classification is not BoundClassification.HBM_BANDWIDTH_BOUND:
            raise RuntimeError(metrics.classification)
        if metrics.capacity_pressure_fraction is None:
            raise RuntimeError("capacity pressure missing")
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                f"The measured kernel is {metrics.classification.value} at "
                f"{metrics.arithmetic_intensity_flops_per_byte} FLOP/byte. "
                f"Capacity pressure is {metrics.capacity_pressure_fraction:g}. "
                "Which statement is right?"
            ),
            answer="separate",
            facts={"oracle": "roofline_class", "knobs": knobs},
            choices=(
                ("separate", "Capacity pressure does not select the roofline class"),
                ("same", "High capacity pressure means the kernel is HBM-bandwidth bound"),
                ("gbps", "Device bytes and GB/s are the same resource"),
                ("util", "Capacity pressure above 0.5 forces underfilled"),
            ),
            grading_type=GradingType.CHOICE,
        )


class Utilization:
    concept_id = "roofline.utilization_not_saturation"

    def generate(self, seed: int, difficulty: float) -> Problem:
        metrics = roofline_of(_UNDER).metrics
        if metrics.classification is not BoundClassification.UNDERFILLED:
            raise RuntimeError(metrics.classification)
        return text_problem(
            concept_id=self.concept_id,
            seed=seed,
            difficulty=difficulty,
            prompt=(
                "Generic GPU utilization is 0.99. The kernel's achieved FLOP/s is "
                f"{metrics.achieved_tflops} TFLOP/s against its roofline. "
                "What is the class?"
            ),
            answer=metrics.classification.value,
            facts={"oracle": "roofline_class", "knobs": _UNDER},
            choices=_CLASSES,
            grading_type=GradingType.CLASSIFY,
        )


def _register_all() -> None:
    for generator in (
        ArithmeticIntensity(),
        RidgePoint(),
        Classify(),
        CapacityVsBandwidth(),
        Utilization(),
    ):
        register(generator)


_register_all()
