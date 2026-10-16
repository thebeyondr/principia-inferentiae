from __future__ import annotations

from datetime import datetime

import pytest

from principia_inferentiae.contracts import (
    ComputeCeiling,
    Digest,
    HardwareSpec,
    ModelSpec,
    QualityGate,
    QuantizationSpec,
    RunMetadata,
    RuntimeReserve,
    ServingConfig,
    SizingPolicy,
    SLOSpec,
    SourceProvenance,
    TraceRequest,
)


def digest(character: str = "a") -> Digest:
    return Digest(algorithm="sha256", value=character * 64)


def provenance(evidence_class: str = "synthetic") -> SourceProvenance:
    return SourceProvenance.model_validate(
        {
            "source_uri": "fixture://principia/tests",
            "revision": "1",
            "digest": digest(),
            "evidence_class": evidence_class,
            "retrieved_at_utc": "2026-01-01T00:00:00Z",
            "usage_note": "Synthetic test fixture; not a vendor performance claim.",
        }
    )


@pytest.fixture
def run() -> RunMetadata:
    return RunMetadata(
        schema_version="1.0.0",
        run_id="test-run",
        created_at_utc=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        package_version="0.1.0",
        code_revision="fixture-revision",
        reproducibility_tier="R1",
        method_revision="methods-1",
        random_seed=7,
        input_digests=(digest("1"),),
    )


@pytest.fixture
def model() -> ModelSpec:
    return ModelSpec(
        model_id="synthetic-gqa-8b",
        revision="fixture-1",
        parameter_count=8_000_000_000,
        layer_count=32,
        hidden_size=4096,
        attention_heads=32,
        kv_heads=8,
        head_dim=128,
        attention_type="GQA",
        weight_format=QuantizationSpec(
            name="bf16",
            bits_per_weight=16,
            metadata_bytes_per_parameter=0,
            metadata_semantics="none",
        ),
        kv_dtype_bytes=2,
        provenance=provenance(),
    )


@pytest.fixture
def hardware() -> HardwareSpec:
    source = provenance()
    return HardwareSpec(
        hardware_id="synthetic-80gb",
        revision="fixture-1",
        device_memory_bytes=80 * 1024**3,
        empirical_hbm_bandwidth_gbps=1_000,
        hbm_method_id="synthetic-ceiling",
        compute_ceilings=(
            ComputeCeiling(
                precision="bf16",
                empirical_tflops=100,
                method_id="synthetic-ceiling",
                provenance=source,
            ),
        ),
        supported_tensor_parallel_sizes=(1, 2, 4, 8),
        devices_per_node=8,
        interconnect="synthetic full mesh",
        provenance=source,
    )


@pytest.fixture
def config() -> ServingConfig:
    return ServingConfig(
        config_id="synthetic-vllm-tp2",
        model_id="synthetic-gqa-8b",
        model_revision="fixture-1",
        runtime_name="vllm",
        runtime_revision="0.11.0",
        precision="bf16",
        hardware_id="synthetic-80gb",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        max_concurrent_sequences=4,
        max_sequence_tokens=4096,
        max_batched_tokens=4096,
        weight_sharding="tensor_and_pipeline",
        kv_sharding="shard_kv_heads",
        sharding_imbalance_fraction=0,
        runtime_reserve=RuntimeReserve(
            runtime_fixed_bytes_per_rank=2 * 1024**3,
            activation_bytes_per_rank=1 * 1024**3,
            graph_bytes_per_rank=512 * 1024**2,
            workspace_bytes_per_rank=512 * 1024**2,
            fragmentation_fraction=0.05,
            safety_margin_fraction=0.10,
            provenance=provenance(),
        ),
        material_runtime_flags=("--disable-log-requests",),
    )


@pytest.fixture
def slo() -> SLOSpec:
    return SLOSpec(
        max_ttft_ms_p99=1_500,
        max_itl_ms_p99=80,
        max_tpot_ms_p99=80,
        max_e2e_ms_p99=10_000,
        min_per_user_tps=12,
        min_aggregate_tps=100,
        min_goodput_fraction=0.95,
        metric_scope="end_to_end",
        queue_boundary="client send to final byte",
    )


@pytest.fixture
def trace() -> tuple[TraceRequest, ...]:
    return (
        TraceRequest(
            request_id="r1",
            arrival_time_ms=0,
            completed_time_ms=2_000,
            prompt_tokens=1_000,
            output_tokens=100,
            tenant_id="a",
            prefix_key="p1",
        ),
        TraceRequest(
            request_id="r2",
            arrival_time_ms=100,
            completed_time_ms=2_200,
            prompt_tokens=2_000,
            output_tokens=200,
            tenant_id="a",
            prefix_key="p1",
        ),
        TraceRequest(
            request_id="r3",
            arrival_time_ms=1_100,
            completed_time_ms=3_000,
            prompt_tokens=4_000,
            output_tokens=300,
            tenant_id="b",
            prefix_key=None,
        ),
        TraceRequest(
            request_id="r4",
            arrival_time_ms=1_200,
            completed_time_ms=3_200,
            prompt_tokens=8_000,
            output_tokens=400,
            tenant_id="b",
            prefix_key="p2",
        ),
    )


@pytest.fixture
def quality_gate() -> QualityGate:
    return QualityGate(
        evaluation_id="quality-1",
        model_id="synthetic-gqa-8b",
        model_revision="fixture-1",
        precision="bf16",
        passed=True,
        metric_summary="Synthetic fixture quality gate.",
        artifact_digest=digest("b"),
    )


@pytest.fixture
def policy() -> SizingPolicy:
    return SizingPolicy(
        target_headroom_fraction=0.20,
        failure_reserve_replicas=1,
        rollout_reserve_replicas=1,
    )
