"""Golden clocks for the admission simulator, recorded from the node oracle."""

from __future__ import annotations

import pytest

from principia_inferentiae.sim import SimResult, simulate

# (request_id, arrival_time_ms, prompt_tokens, output_tokens) from labs/scenarios/today.json.
# Those bursts are seconds apart, so occupancy never reaches kv_slots and the
# admission knob does not move either clock. Expected clocks are from
# tests/fixtures/sim/admission_oracle.js (node v20.9.0 via nvm).
_TODAY_ROWS: tuple[tuple[str, int, int, int], ...] = (
    ("burst-a-01", 0, 20000, 512),
    ("burst-a-02", 40, 19800, 384),
    ("burst-a-03", 80, 19500, 256),
    ("burst-a-04", 120, 18000, 512),
    ("burst-a-05", 160, 17000, 128),
    ("burst-a-06", 200, 16000, 256),
    ("burst-b-01", 5000, 20000, 512),
    ("burst-b-02", 5050, 19000, 384),
    ("burst-b-03", 5100, 18500, 256),
    ("burst-b-04", 5150, 17500, 128),
    ("steady-01", 10000, 15000, 256),
    ("steady-02", 15000, 20000, 512),
)


def _admission_scenario(step_ms: int | None) -> dict[str, object]:
    arrivals: list[dict[str, object]] = []
    rows = enumerate(_TODAY_ROWS)
    for index, (request_id, arrival_time_ms, prompt_tokens, output_tokens) in rows:
        when = index * step_ms if step_ms is not None else arrival_time_ms
        arrivals.append(
            {
                "request_id": request_id,
                "arrival_time_ms": when,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
            }
        )
    return {
        "prefill_us_per_token": 20,
        "decode_us_per_token": 2000,
        "kv_slots": 8,
        "broken": "admission",
        "slo": {"ttft_ms": 500, "itl_ms": 20},
        "arrivals": arrivals,
    }


TODAY = _admission_scenario(None)
# Same prompts, every 20 ms: the schedule in labs/today.html, where the cap binds.
PACKED = _admission_scenario(20)


def _clocks(result: SimResult) -> tuple[float | None, float | None, float | None, int, int]:
    return (
        result.ttft_p99_ms,
        result.itl_p99_ms,
        result.goodput,
        result.rejects,
        result.admitted,
    )


@pytest.mark.parametrize(
    ("repair", "expected"),
    [
        (None, (402.0, 12.0, 1.0, 0, 12)),
        ("admission", (402.0, 12.0, 1.0, 0, 12)),
        ("prefill", (202.0, 12.0, 1.0, 0, 12)),
        ("decode", (401.0, 6.0, 1.0, 0, 12)),
        ("kv", (402.0, 12.0, 1.0, 0, 12)),
    ],
)
def test_admission_p99_matches_node_oracle(
    repair: str | None,
    expected: tuple[float, float, float, int, int],
) -> None:
    result = simulate(TODAY, repair)
    assert result.ttft_p99_ms == expected[0]
    assert result.itl_p99_ms == expected[1]
    assert _clocks(result) == expected


def test_admission_repair_clears_itl_wrong_repair_does_not() -> None:
    # broken=admission clears `enforce`, so every arrival shares decode.
    # ITL step is decode_ms * decoding.length: 2 * 12 = 24, above the 20 ms SLO.
    # repair "admission" holds the cap at kv_slots (8), so the step falls to 16 ms.
    # repair "prefill" only sets aMul = 0.5. TTFT drops; the cap stays off, so ITL stays 24.
    baseline = simulate(PACKED, None)
    fixed = simulate(PACKED, "admission")
    wrong = simulate(PACKED, "prefill")

    assert _clocks(baseline) == (632.0, 24.0, 0.0, 0, 12)
    assert baseline.itl_p99_ms is not None and baseline.itl_p99_ms > 20
    assert _clocks(fixed) == (422.0, 16.0, 1.0, 4, 8)
    assert fixed.itl_p99_ms is not None and fixed.itl_p99_ms <= 20
    assert fixed.ttft_p99_ms is not None and fixed.ttft_p99_ms <= 500
    assert _clocks(wrong) == (255.0, 24.0, 0.0, 0, 12)
    assert wrong.itl_p99_ms == baseline.itl_p99_ms
    assert wrong.rejects == 0


def _scenario(kind: str, arrivals: list[dict[str, object]], **extra: object) -> dict[str, object]:
    scenario: dict[str, object] = {
        "kind": kind,
        "prefill_us_per_token": 20,
        "decode_us_per_token": 2000,
        "kv_slots": 8,
        "slo": {"ttft_ms": 500, "itl_ms": 20},
        "arrivals": arrivals,
    }
    scenario.update(extra)
    return scenario


def _arrival(request_id: str, arrival: int, prompt: int, output: int) -> dict[str, object]:
    return {
        "request_id": request_id,
        "arrival_time_ms": arrival,
        "prompt_tokens": prompt,
        "output_tokens": output,
    }


def test_day3_chunk_and_queue_match_node_oracle() -> None:
    # Recorded from tests/fixtures/sim/kernels_oracle.js.
    chunk_rows = [
        _arrival("long", 0, 4096, 16),
        _arrival("s1", 5, 100, 8),
        _arrival("s2", 6, 120, 8),
    ]
    chunk = _scenario("chunk", chunk_rows, chunk_tokens=2048)
    assert _clocks(simulate(chunk, None)) == (86.32000000000001, 6.0, 1.0, 0, 3)
    chunked = simulate(chunk, "chunk")
    assert _clocks(chunked) == (89.92, 2.0, 1.0, 0, 3)
    alias = _scenario("chunked_prefill", chunk_rows, chunk_tokens=2048)
    assert simulate(alias, "chunked_prefill") == chunked
    with_effects = _scenario("chunk", chunk_rows, chunk_tokens=2048, effects={"c": "chunk"})
    assert simulate(with_effects, "c") == chunked

    queue_rows = [
        _arrival("q1", 0, 500, 16),
        _arrival("q2", 10, 500, 16),
        _arrival("q3", 20, 800, 32),
    ]
    queue = _scenario("queue", queue_rows)
    assert _clocks(simulate(queue, None)) == (82.0, 2.0, 1.0, 0, 3)
    assert _clocks(simulate(queue, "add_engines")) == (28.0, 2.0, 1.0, 0, 3)


def test_frag_paged_blocks_admits_the_rest() -> None:
    rows = [_arrival(f"f{index}", index * 10, 400, 24) for index in range(8)]
    frag = _scenario("frag", rows)
    assert _clocks(simulate(frag, None)) == (16.0, 8.0, 1.0, 4, 4)
    assert _clocks(simulate(frag, "paged_blocks")) == (24.0, 16.0, 1.0, 0, 8)
