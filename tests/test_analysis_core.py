from __future__ import annotations

import pytest
from pydantic import ValidationError

from principia_inferentiae.contracts import (
    BoundClassification,
    Digest,
    HardwareSpec,
    ModelSpec,
    QuantizationSpec,
    RooflineMeasurement,
    RunMetadata,
    ServingConfig,
    SLOSpec,
    SourceProvenance,
    TraceRequest,
)
from principia_inferentiae.memory import estimate_memory
from principia_inferentiae.roofline import analyze_roofline
from principia_inferentiae.workload import (
    characterize_workload,
    prepare_vllm_timed_trace,
)


def _model_with(model: ModelSpec, **updates: object) -> ModelSpec:
    payload = model.model_dump()
    payload.update(updates)
    return ModelSpec.model_validate(payload)


def _config_with(config: ServingConfig, **updates: object) -> ServingConfig:
    payload = config.model_dump()
    payload.update(updates)
    return ServingConfig.model_validate(payload)


def _measurement(
    hardware: HardwareSpec,
    provenance: SourceProvenance,
    *,
    flops: float,
    hbm_bytes: float,
    duration_ms: float,
    utilization: float = 0.99,
) -> RooflineMeasurement:
    return RooflineMeasurement(
        measurement_id="m1",
        measurement_status="measured",
        scope_kind="kernel",
        scope_name="synthetic kernel",
        hardware_id=hardware.hardware_id,
        hardware_revision=hardware.revision,
        precision="bf16",
        flops=flops,
        hbm_bytes_moved=hbm_bytes,
        duration_ms=duration_ms,
        capacity_used_bytes=hardware.device_memory_bytes - 1,
        generic_gpu_utilization_fraction=utilization,
        raw_artifact_digest=Digest(algorithm="sha256", value="c" * 64),
        provenance=provenance,
    )


def test_workload_is_order_invariant(
    run: RunMetadata,
    trace: tuple[TraceRequest, ...],
    slo: SLOSpec,
) -> None:
    forward = characterize_workload(run, trace, slo)
    reverse = characterize_workload(run, tuple(reversed(trace)), slo)
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
    assert forward.profile.prompt_tokens.p99 == 8_000
    assert forward.profile.output_tokens.p99 == 400


def test_vllm_trace_adapter_preserves_shared_prefix_chunks() -> None:
    first = TraceRequest(
        request_id="a",
        arrival_time_ms=0,
        prompt_tokens=2_000,
        output_tokens=100,
        tenant_id="t",
        prefix_key="shared",
        shared_prefix_tokens=1_024,
    )
    second = TraceRequest(
        request_id="b",
        arrival_time_ms=1,
        prompt_tokens=2_000,
        output_tokens=100,
        tenant_id="t",
        prefix_key="shared",
        shared_prefix_tokens=1_024,
    )
    adapted = prepare_vllm_timed_trace((second, first), chunk_hash_size=512)
    assert adapted[0].timestamp == 0
    assert adapted[0].hash_ids[:2] == adapted[1].hash_ids[:2]
    assert adapted[0].hash_ids[2:] != adapted[1].hash_ids[2:]


def _measured_hardware(hardware: HardwareSpec) -> HardwareSpec:
    source = hardware.provenance.model_copy(
        update={"evidence_class": "measured", "usage_note": "Measured test ceiling."}
    )
    return hardware.model_copy(
        update={
            "provenance": source,
            "compute_ceilings": tuple(
                item.model_copy(update={"provenance": source}) for item in hardware.compute_ceilings
            ),
        }
    )


@pytest.mark.parametrize(
    ("attention_type", "kv_heads", "expected_bytes"),
    [
        ("MHA", 32, 524_288),
        ("GQA", 8, 131_072),
        ("MQA", 1, 16_384),
    ],
)
def test_kv_bytes_follow_attention_semantics(
    run: RunMetadata,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
    attention_type: str,
    kv_heads: int,
    expected_bytes: int,
) -> None:
    variant = _model_with(model, attention_type=attention_type, kv_heads=kv_heads)
    replicated = _config_with(config, tensor_parallel_size=1, kv_sharding="replicated")
    analysis = estimate_memory(run, variant, hardware, replicated)
    assert analysis.estimate.kv_bytes_per_token_logical == expected_bytes


def test_quantization_metadata_and_sharding_are_counted(
    run: RunMetadata,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
) -> None:
    quantized = _model_with(
        model,
        weight_format=QuantizationSpec(
            name="int4-grouped",
            bits_per_weight=4,
            metadata_bytes_per_parameter=0.125,
            group_size=64,
            metadata_semantics="scale and zero-point amortized per parameter",
        ),
    )
    analysis = estimate_memory(run, quantized, hardware, config)
    assert analysis.estimate.quantization_metadata_total_bytes == 1_000_000_000
    assert analysis.estimate.weights_total_bytes == 5_000_000_000
    assert analysis.estimate.weights_per_rank_bytes == 2_500_000_000


def test_memory_demand_is_monotonic_in_concurrency(
    run: RunMetadata,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
) -> None:
    low = estimate_memory(
        run,
        model,
        hardware,
        _config_with(config, max_concurrent_sequences=1),
    )
    high = estimate_memory(
        run,
        model,
        hardware,
        _config_with(config, max_concurrent_sequences=8),
    )
    assert high.estimate.required_per_rank_bytes > low.estimate.required_per_rank_bytes


@pytest.mark.parametrize(
    ("flops", "hbm_bytes", "duration_ms", "expected"),
    [
        (200e12, 1e12, 2_500, BoundClassification.COMPUTE_BOUND),
        (50e12, 1e12, 1_250, BoundClassification.HBM_BANDWIDTH_BOUND),
        (50e12, 1e12, 10_000, BoundClassification.UNDERFILLED),
        (100e12, 1e12, 1_250, BoundClassification.MIXED),
    ],
)
def test_roofline_classes_use_measured_scope(
    run: RunMetadata,
    hardware: HardwareSpec,
    flops: float,
    hbm_bytes: float,
    duration_ms: float,
    expected: BoundClassification,
) -> None:
    measured_hardware = _measured_hardware(hardware)
    measurement = _measurement(
        measured_hardware,
        measured_hardware.provenance,
        flops=flops,
        hbm_bytes=hbm_bytes,
        duration_ms=duration_ms,
    )
    result = analyze_roofline(run, measurement, measured_hardware)
    assert result.metrics.classification is expected


def test_generic_utilization_does_not_change_roofline_class(
    run: RunMetadata,
    hardware: HardwareSpec,
) -> None:
    measured_hardware = _measured_hardware(hardware)
    low = _measurement(
        measured_hardware,
        measured_hardware.provenance,
        flops=50e12,
        hbm_bytes=1e12,
        duration_ms=1_250,
        utilization=0.01,
    )
    high = _measurement(
        measured_hardware,
        measured_hardware.provenance,
        flops=50e12,
        hbm_bytes=1e12,
        duration_ms=1_250,
        utilization=0.99,
    )
    assert analyze_roofline(run, low, measured_hardware).metrics.classification == (
        analyze_roofline(run, high, measured_hardware).metrics.classification
    )
    assert (
        analyze_roofline(run, low, measured_hardware).metrics.classification
        is BoundClassification.HBM_BANDWIDTH_BOUND
    )


def test_malformed_units_and_impossible_slo_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RooflineMeasurement.model_validate(
            {
                "measurement_id": "bad",
                "measurement_status": "measured",
                "scope_kind": "kernel",
                "scope_name": "bad units",
                "precision": "bf16",
                "flops": 1,
                "hbm_bytes_moved": 1,
                "duration_s": 1,
                "raw_artifact_digest": {"algorithm": "sha256", "value": "d" * 64},
                "provenance": {
                    "source_uri": "fixture://bad",
                    "revision": "1",
                    "digest": {"algorithm": "sha256", "value": "e" * 64},
                    "evidence_class": "synthetic",
                    "retrieved_at_utc": "2026-01-01T00:00:00Z",
                    "usage_note": "fixture",
                },
            }
        )
    with pytest.raises(ValidationError, match="E2E p99 must exceed TTFT"):
        SLOSpec(
            max_ttft_ms_p99=100,
            max_e2e_ms_p99=100,
            min_goodput_fraction=0.9,
            metric_scope="end_to_end",
            queue_boundary="client",
        )


def test_shard_layers_does_not_divide_layers_by_tensor_parallel(
    run: RunMetadata,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
) -> None:
    variant = _model_with(model, attention_type="MQA", kv_heads=1)
    sharded = _config_with(config, tensor_parallel_size=8, kv_sharding="shard_layers")
    analysis = estimate_memory(run, variant, hardware, sharded)
    per_rank = analysis.estimate.kv_bytes_per_token_per_rank
    logical = analysis.estimate.kv_bytes_per_token_logical
    assert per_rank == logical


def test_indivisible_kv_heads_raise(
    run: RunMetadata,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
) -> None:
    variant = _model_with(model, attention_heads=12, kv_heads=6)
    sharded = _config_with(config, tensor_parallel_size=4)
    with pytest.raises(ValueError, match="kv_heads"):
        estimate_memory(run, variant, hardware, sharded)


def test_synthetic_roofline_ceiling_is_insufficient(
    run: RunMetadata,
    hardware: HardwareSpec,
) -> None:
    measurement = _measurement(
        hardware,
        hardware.provenance,
        flops=200e12,
        hbm_bytes=1e12,
        duration_ms=2_500,
    )
    result = analyze_roofline(run, measurement, hardware)
    assert result.metrics.classification is BoundClassification.INSUFFICIENT_EVIDENCE


def test_unknown_hardware_identity_fails_closed(
    run: RunMetadata,
    model: ModelSpec,
    hardware: HardwareSpec,
    config: ServingConfig,
) -> None:
    unknown = _config_with(config, hardware_id="unknown")
    with pytest.raises(ValueError, match="hardware identity"):
        estimate_memory(run, model, hardware, unknown)
