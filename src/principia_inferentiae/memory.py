"""Transparent model-weight, KV-cache, and per-rank capacity calculations."""

from __future__ import annotations

import math

from principia_inferentiae.contracts import (
    AnalysisStatus,
    Claim,
    Confidence,
    EvidenceKind,
    EvidenceRecord,
    HardwareSpec,
    MemoryAnalysis,
    MemoryEstimate,
    ModelSpec,
    RunMetadata,
    ServingConfig,
    UnitRecord,
)
from principia_inferentiae.io import digest_value

METHOD_ID = "memory-capacity/1.0.0"


def _exact_parts(count: int, parts: int, label: str) -> int:
    if count % parts:
        raise ValueError(f"{label} ({count}) must be divisible by parallel size {parts}")
    return count // parts


def _weight_shards(config: ServingConfig) -> int:
    if config.tensor_parallel_size > 1 and config.weight_sharding != "tensor_and_pipeline":
        raise ValueError("tensor parallelism requires tensor_and_pipeline weight sharding")
    if config.weight_sharding == "tensor_and_pipeline":
        return config.tensor_parallel_size * config.pipeline_parallel_size
    if config.weight_sharding == "pipeline_only":
        return config.pipeline_parallel_size
    return 1


def _kv_layout(model: ModelSpec, config: ServingConfig) -> tuple[int, int]:
    if model.attention_heads % config.tensor_parallel_size:
        raise ValueError("attention_heads must be divisible by tensor_parallel_size")
    layers = _exact_parts(model.layer_count, config.pipeline_parallel_size, "layer_count")
    if config.kv_sharding == "replicated":
        return layers, model.kv_heads
    if config.kv_sharding == "shard_layers":
        return layers, model.kv_heads
    if model.kv_heads % config.tensor_parallel_size == 0:
        return layers, model.kv_heads // config.tensor_parallel_size
    if config.tensor_parallel_size % model.kv_heads == 0:
        if model.kv_projection_parameter_count is None:
            raise ValueError(
                "TP greater than kv_heads replicates K/V projections; "
                "set kv_projection_parameter_count"
            )
        return layers, 1
    raise ValueError("kv_heads are incompatible with tensor_parallel_size")


def _parameter_bytes(parameter_count: int, model: ModelSpec) -> int:
    payload = math.ceil(parameter_count * model.weight_format.bits_per_weight / 8)
    metadata = math.ceil(parameter_count * model.weight_format.metadata_bytes_per_parameter)
    return payload + metadata


def _kv_bytes_per_token_per_rank(model: ModelSpec, config: ServingConfig) -> int:
    layers, kv_heads = _kv_layout(model, config)
    return math.ceil(2 * layers * kv_heads * model.head_dim * model.kv_dtype_bytes)


def estimate_memory(
    run: RunMetadata,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
) -> MemoryAnalysis:
    """Estimate conservative maximum-rank capacity for one replica."""

    if config.model_id != model.model_id or config.model_revision != model.revision:
        raise ValueError("serving config model identity does not match model spec")
    if config.hardware_id != hardware.hardware_id:
        raise ValueError("serving config hardware identity does not match hardware spec")
    if config.tensor_parallel_size not in hardware.supported_tensor_parallel_sizes:
        raise ValueError("tensor-parallel size is unsupported by the hardware contract")

    metadata_bytes = math.ceil(
        model.parameter_count * model.weight_format.metadata_bytes_per_parameter
    )
    weights_total_bytes = _parameter_bytes(model.parameter_count, model)
    weight_shards = _weight_shards(config)
    if (
        config.kv_sharding == "shard_kv_heads"
        and config.tensor_parallel_size > model.kv_heads
        and model.kv_projection_parameter_count is not None
    ):
        other_parameters = model.parameter_count - model.kv_projection_parameter_count
        other_bytes = math.ceil(
            _parameter_bytes(other_parameters, model)
            / weight_shards
            * (1 + config.sharding_imbalance_fraction)
        )
        weights_per_rank_bytes = other_bytes + _parameter_bytes(
            model.kv_projection_parameter_count,
            model,
        )
    else:
        weights_per_rank_bytes = math.ceil(
            weights_total_bytes / weight_shards * (1 + config.sharding_imbalance_fraction)
        )

    kv_bytes_per_token_logical = math.ceil(
        2 * model.layer_count * model.kv_heads * model.head_dim * model.kv_dtype_bytes
    )
    kv_bytes_per_token_per_rank = _kv_bytes_per_token_per_rank(model, config)
    active_tokens = config.max_concurrent_sequences * config.max_sequence_tokens
    kv_cache_per_rank_bytes = kv_bytes_per_token_per_rank * active_tokens

    reserve = config.runtime_reserve
    runtime_reserve_per_rank_bytes = (
        reserve.runtime_fixed_bytes_per_rank
        + reserve.activation_bytes_per_rank
        + reserve.graph_bytes_per_rank
        + reserve.workspace_bytes_per_rank
    )
    pre_fragmentation_bytes = (
        weights_per_rank_bytes + kv_cache_per_rank_bytes + runtime_reserve_per_rank_bytes
    )
    fragmentation_reserve_per_rank_bytes = math.ceil(
        pre_fragmentation_bytes * reserve.fragmentation_fraction
    )
    required_per_rank_bytes = pre_fragmentation_bytes + fragmentation_reserve_per_rank_bytes
    usable_device_bytes = math.floor(
        hardware.device_memory_bytes * (1 - reserve.safety_margin_fraction)
    )
    headroom_per_rank_bytes = usable_device_bytes - required_per_rank_bytes
    fits_per_rank = headroom_per_rank_bytes >= 0

    state_bytes = weights_total_bytes + kv_bytes_per_token_logical * active_tokens
    inflated_state_bytes = math.ceil(state_bytes * (1 + reserve.fragmentation_fraction))
    inflated_reserve_bytes = math.ceil(
        runtime_reserve_per_rank_bytes * (1 + reserve.fragmentation_fraction)
    )
    capacity_denominator = usable_device_bytes - inflated_reserve_bytes
    if not fits_per_rank or capacity_denominator <= 0:
        aggregate_capacity_lower_bound_gpus = None
    else:
        aggregate_capacity_lower_bound_gpus = math.ceil(inflated_state_bytes / capacity_denominator)
    status = AnalysisStatus.SUPPORTED if fits_per_rank else AnalysisStatus.MEMORY_NOT_FIT
    assumptions = (
        "Capacity uses configured maximum concurrent sequences and sequence length.",
        "The result is a maximum-rank estimate; aggregate capacity is only a lower bound.",
        "HBM capacity fit is separate from HBM bandwidth and compute throughput.",
        "Pipeline stages are approximated as equal partitions plus declared imbalance.",
    )
    estimate = MemoryEstimate(
        status=status,
        weights_total_bytes=weights_total_bytes,
        quantization_metadata_total_bytes=metadata_bytes,
        weights_per_rank_bytes=weights_per_rank_bytes,
        kv_bytes_per_token_logical=kv_bytes_per_token_logical,
        kv_bytes_per_token_per_rank=kv_bytes_per_token_per_rank,
        active_tokens_configured=active_tokens,
        kv_cache_per_rank_bytes=kv_cache_per_rank_bytes,
        runtime_reserve_per_rank_bytes=runtime_reserve_per_rank_bytes,
        fragmentation_reserve_per_rank_bytes=fragmentation_reserve_per_rank_bytes,
        required_per_rank_bytes=required_per_rank_bytes,
        usable_device_bytes=usable_device_bytes,
        headroom_per_rank_bytes=headroom_per_rank_bytes,
        fits_per_rank=fits_per_rank,
        aggregate_capacity_lower_bound_gpus=aggregate_capacity_lower_bound_gpus,
        model_digest=digest_value(model),
        hardware_digest=digest_value(hardware),
        config_hash=digest_value(config).value,
        assumptions=assumptions,
    )

    model_digest = estimate.model_digest
    hardware_digest = estimate.hardware_digest
    config_digest = digest_value(config)
    estimate_digest = digest_value(estimate)
    evidence = (
        EvidenceRecord(
            evidence_id="memory.model",
            kind=EvidenceKind.INPUT,
            description="Validated model architecture and precision contract",
            observed_value=model_digest.value,
            unit="sha256",
            source_digest=model_digest,
        ),
        EvidenceRecord(
            evidence_id="memory.hardware",
            kind=EvidenceKind.INPUT,
            description="Validated hardware capacity contract",
            observed_value=hardware_digest.value,
            unit="sha256",
            source_digest=hardware_digest,
        ),
        EvidenceRecord(
            evidence_id="memory.serving_config",
            kind=EvidenceKind.CONFIGURATION,
            description="Validated serving and sharding configuration",
            observed_value=config_digest.value,
            unit="sha256",
            source_digest=config_digest,
        ),
        EvidenceRecord(
            evidence_id="memory.estimate",
            kind=EvidenceKind.ESTIMATE,
            description="Deterministic maximum-rank capacity estimate",
            observed_value=str(required_per_rank_bytes),
            unit="bytes/rank",
            source_digest=estimate_digest,
        ),
    )
    lower_bound = (
        "null"
        if aggregate_capacity_lower_bound_gpus is None
        else str(aggregate_capacity_lower_bound_gpus)
    )
    claim = Claim(
        claim_id="memory.per_rank_fit",
        claim_type="memory_capacity_fit",
        statement=(
            f"Configured replica {'fits' if fits_per_rank else 'does not fit'} per rank: "
            f"{required_per_rank_bytes} required bytes versus {usable_device_bytes} usable bytes; "
            f"aggregate capacity lower bound is {lower_bound} GPU(s)."
        ),
        status=status,
        evidence_ids=tuple(item.evidence_id for item in evidence),
        supporting_evidence_ids=tuple(item.evidence_id for item in evidence),
        method_id=METHOD_ID,
        units=(
            UnitRecord(field="required_per_rank_bytes", unit="bytes/rank"),
            UnitRecord(field="usable_device_bytes", unit="bytes/device"),
            UnitRecord(field="kv_bytes_per_token_logical", unit="bytes/token"),
        ),
        assumptions=assumptions,
        limitations=(
            "This static estimate does not establish allocatability or SLO-goodput.",
            "Runtime reserve accuracy depends on the supplied measured or declared reserve.",
        ),
        confidence=(
            Confidence.MEDIUM
            if model.provenance.evidence_class == "measured"
            and hardware.provenance.evidence_class == "measured"
            and config.runtime_reserve.provenance.evidence_class == "measured"
            else Confidence.LOW
        ),
        missing_telemetry=(),
        next_discriminating_experiment=(
            "Run the pinned serving benchmark and record peak allocated HBM per rank."
        ),
    )
    return MemoryAnalysis(run=run, evidence=evidence, claims=(claim,), estimate=estimate)
