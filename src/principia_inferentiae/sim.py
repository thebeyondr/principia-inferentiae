"""Ports of the laptop serving simulators.

``simulate`` is deterministic: it does not read a clock, the filesystem, or the network.
``repair=None`` runs the injected fault. On the admission kernel that fault is
``scenario["broken"]`` (``admission`` leaves the KV cap unenforced, so decode
concurrency and therefore inter-token latency grow with every extra admit).
Otherwise the fault is the baseline choice of the kernel named by ``scenario["kind"]``.

Repair names select a kernel when ``kind`` is omitted, and name the page move when
``kind`` is set. Admission moves are ``admission``, ``prefill``, ``decode``, and ``kv``.
Day-3 kernels are ``queue``, ``chunk`` / ``chunked_prefill``, ``retry`` / ``retry_storm``,
``goodput``, and ``admit`` (choice letters ``a``-``d``, not ``effects``). Day-4 kernels
are ``prefix``, ``priority``, ``sticky``, ``hog``, ``frag``, ``spec``, ``evict``,
``lora``, ``scale``, and ``batch``. A choice letter is resolved through
``scenario["effects"]`` the way the page ``act`` helper does, and ``chunked_prefill``
is the chunk move. Day-3 moves include ``faster_prefill``, ``faster_decode``,
``add_engines``, ``report_e2e``, ``chunk``, ``reject_shorts``, ``stop_retries``,
``cut_timeout``, ``admit_cap``, ``half_retries``, ``cap4``, ``pack12``,
``faster_decode_pack``, and ``admit6``. Day-4 moves include ``enable_prefix``,
``shorter_prompts``, ``more_engines``, ``preempt_batch``, ``admit_fewer``, ``more_kv``,
``sticky_route``, ``pin_user``, ``bigger_cache``, ``cap_output``, ``preempt_long``,
``more_capacity``, ``paged_blocks``, ``buy_memory``, ``lower_batch``, ``raise_accept``,
``bigger_k``, ``faster_draft``, ``disable_spec``, ``grow_cache``, ``pin_hot``,
``more_replicas``, ``queue_by_adapter``, ``merge_one``, ``pin_adapters``, ``prewarm``,
``faster_scale``, ``admit_shed``, ``bigger_machine``, ``continuous``, ``smaller_static``,
and ``bigger_static``. ``kind`` ``device`` and ``cold``, and a ``prefix`` or ``spec``
scenario that still carries the earlier page fields, use ``labs/sim.js``.

Day 4 kernels read the fields those pages already embed: ``prefix_tokens`` (prefix,
sticky, evict), ``replicas`` (sticky), ``max_output`` (hog), ``cache_blocks`` (evict),
``draft_k`` and ``accept`` (spec), ``swap_ms`` and ``unique_adapters`` plus per-arrival
``adapter`` (lora), ``warm_ms`` (scale), per-arrival ``priority`` (priority) and
``user_id`` (sticky). Omitted knobs use the page defaults: ``prefix_tokens`` 4000,
3000, or 2000 by kernel, ``replicas`` 3, ``max_output`` 128, ``cache_blocks`` 4,
``draft_k`` 4, ``accept`` 0.2, ``swap_ms`` 40, ``unique_adapters`` 4, ``warm_ms`` 3000.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from principia_inferentiae.contracts import FrozenModel

__all__ = ["SimResult", "simulate"]

_ADMISSION_KNOBS = frozenset({"admission", "prefill", "decode", "kv"})
_EFFECT_KERNELS = frozenset(
    {
        "queue",
        "chunk",
        "retry",
        "goodput",
        "day4_prefix",
        "priority",
        "sticky",
        "hog",
        "frag",
        "day4_spec",
        "evict",
        "lora",
        "scale",
        "batch",
    }
)
_KIND = {
    "admission": "admission",
    "admit": "admit",
    "queue": "queue",
    "chunk": "chunk",
    "chunked_prefill": "chunk",
    "retry": "retry",
    "retry_storm": "retry",
    "goodput": "goodput",
    "priority": "priority",
    "sticky": "sticky",
    "hog": "hog",
    "frag": "frag",
    "evict": "evict",
    "lora": "lora",
    "scale": "scale",
    "batch": "batch",
    "device": "device",
    "cold": "cold",
}
_REPAIR_KIND = {
    "queue": "queue",
    "chunk": "chunk",
    "chunked_prefill": "chunk",
    "retry": "retry",
    "retry_storm": "retry",
    "goodput": "goodput",
    "prefix": "prefix",
    "priority": "priority",
    "sticky": "sticky",
    "hog": "hog",
    "frag": "frag",
    "spec": "spec",
    "evict": "evict",
    "lora": "lora",
    "scale": "scale",
    "batch": "batch",
    "device": "device",
    "cold": "cold",
    "admit": "admit",
}


class SimResult(FrozenModel):
    """Clocks for one simulator run. ``admitted`` counts finished requests, not rejects."""

    ttft_p99_ms: float | None
    itl_p99_ms: float | None
    goodput: float | None
    rejects: int
    admitted: int


@dataclass(slots=True)
class _Job:
    phase: str
    ttft: float | None
    itls: list[float] = field(default_factory=list)
    arrival: float = 0.0
    prompt: float = 0.0
    output: float = 0.0
    tokens_left: float = 0.0


@dataclass(slots=True)
class _Event:
    t: float
    kind: str
    job: _Job | None = None
    gen: int = 0
    dt: float = 0.0


def _as_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    raw = cast(dict[object, object], value)
    mapped: dict[str, object] = {}
    for key, item in raw.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} keys must be strings")
        mapped[key] = item
    return mapped


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a number")
    return float(value)


def _req_float(mapping: dict[str, object], key: str) -> float:
    if key not in mapping:
        raise ValueError(f"missing {key}")
    return _as_float(mapping[key], key)


def _opt_float(mapping: dict[str, object], key: str, default: float) -> float:
    """Match JavaScript ``value || default`` for missing, zero, and empty values."""

    if key not in mapping:
        return default
    value = mapping[key]
    if value is None or value == "":
        return default
    number = _as_float(value, key)
    if number == 0:
        return default
    return number


def _rows(scenario: dict[str, object]) -> list[dict[str, object]]:
    if "arrivals" not in scenario:
        raise ValueError("missing arrivals")
    raw = scenario["arrivals"]
    if not isinstance(raw, list):
        raise TypeError("arrivals must be a list")
    return [_as_dict(item, "arrival") for item in cast(list[object], raw)]


def _require_rows(scenario: dict[str, object]) -> list[dict[str, object]]:
    rows = _rows(scenario)
    if not rows:
        raise ValueError("arrivals must be non-empty")
    return rows


def _slo(scenario: dict[str, object]) -> tuple[float, float]:
    slo = _as_dict(scenario.get("slo"), "slo")
    return _req_float(slo, "ttft_ms"), _req_float(slo, "itl_ms")


def _p99(values: Sequence[float | None]) -> float | None:
    """Nearest-rank p99 used by the labs: ``min(n - 1, ceil(0.99 * n) - 1)``."""

    if not values:
        return None
    ordered = sorted(values, key=lambda value: 0.0 if value is None else value)
    index = min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)
    return ordered[index]


def _js_le(left: float | None, right: float) -> bool:
    """JavaScript ``null <= number`` is true because null coerces to 0."""

    if left is None:
        return True
    return left <= right


def _fill_itl(job: _Job, gap: float, count: float) -> None:
    if count < 0:
        raise ValueError("output_tokens must be non-negative")
    job.itls = [gap] * math.trunc(count)
    job.phase = "done"


def _done(ttft: float, gap: float, output: float) -> _Job:
    job = _Job(phase="pending", ttft=ttft)
    _fill_itl(job, gap, output)
    return job


def _rejected() -> _Job:
    return _Job(phase="rejected", ttft=None)


def _finish(jobs: list[_Job], scenario: dict[str, object], *, drop_null_ttft: bool) -> SimResult:
    ttft_slo, itl_slo = _slo(scenario)
    done = [job for job in jobs if job.phase == "done"]
    good = sum(1 for job in done if _js_le(job.ttft, ttft_slo) and _js_le(_p99(job.itls), itl_slo))
    if drop_null_ttft:
        ttfts: list[float | None] = [job.ttft for job in done if job.ttft is not None]
    else:
        ttfts = [job.ttft for job in done]
    admitted = len(done)
    return SimResult(
        ttft_p99_ms=_p99(ttfts),
        itl_p99_ms=_p99([_p99(job.itls) for job in done]),
        goodput=(good / admitted) if admitted else 0.0,
        rejects=sum(1 for job in jobs if job.phase == "rejected"),
        admitted=admitted,
    )


def _act(scenario: dict[str, object], choice: str | None) -> str | None:
    if not choice:
        return None
    resolved = choice
    effects = scenario.get("effects")
    if isinstance(effects, dict):
        mapped = cast(dict[object, object], effects).get(choice)
        if isinstance(mapped, str) and mapped:
            resolved = mapped
    if resolved == "chunked_prefill":
        return "chunk"
    return resolved


def _rates(scenario: dict[str, object]) -> tuple[float, float]:
    return (
        _req_float(scenario, "prefill_us_per_token") / 1000,
        _req_float(scenario, "decode_us_per_token") / 1000,
    )


def _has_arrival_field(scenario: dict[str, object], key: str) -> bool:
    return any(key in row for row in _rows(scenario))


def _prefix_kernel(scenario: dict[str, object]) -> str:
    if "prefix_tokens" in scenario:
        return "day4_prefix"
    if _has_arrival_field(scenario, "shared_prefix_tokens"):
        return "labs_prefix"
    return "day4_prefix"


def _spec_kernel(scenario: dict[str, object]) -> str:
    if "draft_k" in scenario or "accept" in scenario:
        return "day4_spec"
    labs_spec_keys = ("verify_us", "draft_len", "accept_len", "draft_us_per_token")
    if any(key in scenario for key in labs_spec_keys):
        return "labs_spec"
    return "day4_spec"


def _kernel_from_kind(kind: str, scenario: dict[str, object]) -> str:
    if kind == "prefix":
        return _prefix_kernel(scenario)
    if kind == "spec":
        return _spec_kernel(scenario)
    mapped = _KIND.get(kind)
    if mapped is None:
        raise ValueError(f"unknown scenario kind {kind!r}")
    return mapped


def _kernel(scenario: dict[str, object], repair: str | None) -> str:
    if "kind" in scenario and scenario["kind"] is not None:
        kind = scenario["kind"]
        if not isinstance(kind, str):
            raise TypeError("kind must be a string")
        return _kernel_from_kind(kind, scenario)
    if repair is None or repair in _ADMISSION_KNOBS:
        return "admission"
    mapped = _REPAIR_KIND.get(repair)
    if mapped is None:
        raise ValueError(f"unknown repair {repair!r}")
    return _kernel_from_kind(mapped, scenario)


def _choice(kernel: str, scenario: dict[str, object], repair: str | None) -> str | None:
    if kernel in _EFFECT_KERNELS:
        return _act(scenario, repair)
    return repair


def _admission(scenario: dict[str, object], repair: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    kv_slots = _req_float(scenario, "kv_slots")
    cap = kv_slots
    engines = kv_slots
    enforce = scenario.get("broken") != "admission"
    a_mul = 1.0
    b_mul = 1.0
    if repair == "admission":
        enforce = True
    if repair == "prefill":
        a_mul = 0.5
    if repair == "decode":
        b_mul = 0.5
    if repair == "kv":
        cap = kv_slots * 2
        engines = cap
        enforce = True

    jobs = [
        _Job(
            phase="pending",
            ttft=None,
            arrival=_req_float(row, "arrival_time_ms"),
            prompt=_req_float(row, "prompt_tokens"),
            output=_req_float(row, "output_tokens"),
            tokens_left=_req_float(row, "output_tokens"),
        )
        for row in _rows(scenario)
    ]
    now = 0.0
    free = engines
    occupied = 0.0
    waiting: list[_Job] = []
    decoding: list[_Job] = []
    gen = 0
    events = [_Event(t=job.arrival, kind="arrive", job=job) for job in jobs]

    def schedule_decode() -> None:
        nonlocal gen
        if not decoding:
            return
        gen += 1
        dt = b_ms * b_mul * len(decoding)
        events.append(_Event(t=now + dt, kind="decode", gen=gen, dt=dt))

    def pump() -> None:
        nonlocal free
        while waiting and free > 0:
            job = waiting.pop(0)
            job.phase = "prefill"
            free -= 1
            events.append(_Event(t=now + job.prompt * a_ms * a_mul, kind="prefill", job=job))

    while events:
        events.sort(key=lambda event: event.t)
        event = events.pop(0)
        now = event.t
        if event.kind == "arrive":
            if event.job is None:
                raise RuntimeError("arrive event missing job")
            if enforce and occupied >= cap:
                event.job.phase = "rejected"
            else:
                occupied += 1
                event.job.phase = "waiting"
                waiting.append(event.job)
                pump()
        elif event.kind == "prefill":
            if event.job is None:
                raise RuntimeError("prefill event missing job")
            event.job.phase = "decode"
            free += 1
            decoding.append(event.job)
            pump()
            schedule_decode()
        elif event.kind == "decode" and event.gen == gen:
            for job in decoding:
                job.itls.append(event.dt)
                if job.ttft is None:
                    job.ttft = now - job.arrival
                job.tokens_left -= 1
            index = len(decoding) - 1
            while index >= 0:
                if decoding[index].tokens_left <= 0:
                    decoding[index].phase = "done"
                    del decoding[index]
                    occupied -= 1
                index -= 1
            schedule_decode()
    return _finish(jobs, scenario, drop_null_ttft=False)


def _admit(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    capacity = _req_float(scenario, "kv_slots")
    admit_cap = math.inf
    engines = 1.0
    if choice == "a":
        admit_cap = capacity
    if choice == "b":
        engines = 3.0
    if choice == "d":
        admit_cap = 8.0
    jobs: list[_Job] = []
    live = 0.0
    for row in _rows(scenario):
        output = _req_float(row, "output_tokens")
        if choice == "c":
            output = float(math.ceil(output / 2))
        if live >= admit_cap:
            jobs.append(_rejected())
            continue
        live += 1
        gap = b_ms * math.ceil(live / engines)
        prompt = _req_float(row, "prompt_tokens")
        jobs.append(_done(prompt * a_ms + gap, gap, output))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _queue(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    faster_prefill = move == "faster_prefill"
    faster_decode = move == "faster_decode"
    more_engines = move == "add_engines"
    hide_queue = faster_prefill or faster_decode
    engines = 4 if more_engines else 1
    server_free = 0.0
    jobs: list[_Job] = []
    for index, row in enumerate(_rows(scenario)):
        prompt = _req_float(row, "prompt_tokens")
        output = _req_float(row, "output_tokens")
        arrival = _req_float(row, "arrival_time_ms")
        prefill = prompt * a_ms * (0.5 if faster_prefill else 1.0)
        gap = b_ms * (0.5 if faster_decode else 1.0)
        offset = index % engines * 5 if more_engines else 0.0
        start = arrival + offset if more_engines else max(arrival, server_free)
        queue_wait = start - arrival
        infer_ttft = prefill + gap
        if not more_engines:
            server_free = start + prefill + output * gap
        ttft = infer_ttft if hide_queue else queue_wait + infer_ttft
        jobs.append(_done(ttft, gap, output))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _chunk(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    rows = _require_rows(scenario)
    long = rows[0]
    shorts = rows[1:]
    long_prompt = _req_float(long, "prompt_tokens")
    long_output = _req_float(long, "output_tokens")
    jobs: list[_Job] = []
    if move == "chunk":
        chunk_size = _opt_float(scenario, "chunk_tokens", 2048)
        if chunk_size <= 0:
            raise ValueError("chunk_tokens must be positive")
        clock = 0.0
        remaining = long_prompt
        while remaining > 0:
            piece = min(chunk_size, remaining)
            clock += piece * a_ms
            remaining -= piece
            for _short in shorts:
                clock += b_ms
        jobs.append(_done(clock, b_ms, long_output))
        for row in shorts:
            prompt = _req_float(row, "prompt_tokens")
            jobs.append(_done(prompt * a_ms + b_ms, b_ms, _req_float(row, "output_tokens")))
        return _finish(jobs, scenario, drop_null_ttft=True)

    pre_mul = 0.5 if move == "faster_prefill" else 1.0
    dec_mul = 0.5 if move == "faster_decode" else 1.0
    long_prefill = long_prompt * a_ms * pre_mul
    jobs.append(_done(long_prefill + b_ms * dec_mul, b_ms * dec_mul, long_output))
    for row in shorts:
        if move == "reject_shorts":
            jobs.append(_rejected())
            continue
        prompt = _req_float(row, "prompt_tokens")
        output = _req_float(row, "output_tokens")
        ttft = long_prefill + prompt * a_ms * pre_mul + b_ms * dec_mul
        gap = b_ms * dec_mul * (1 + len(shorts))
        jobs.append(_done(ttft, gap, output))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _retry(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    multiplier = _opt_float(scenario, "retry_mul", 3)
    timeout_ms = _opt_float(scenario, "timeout_ms", 200)
    admit_cap = math.inf
    if move == "stop_retries":
        multiplier = 1
    if move == "cut_timeout":
        timeout_ms = 50
    if move == "admit_cap":
        admit_cap = _req_float(scenario, "kv_slots")
    if move == "half_retries":
        multiplier = 2
    traffic: list[tuple[float, float, float]] = []
    for row in _rows(scenario):
        attempt = 0.0
        arrival = _req_float(row, "arrival_time_ms")
        prompt = _req_float(row, "prompt_tokens")
        output = _req_float(row, "output_tokens")
        while attempt < multiplier:
            traffic.append((arrival + attempt * 30, prompt, output))
            attempt += 1
    jobs: list[_Job] = []
    live = 0.0
    for _arrival, prompt, output in traffic:
        if live >= admit_cap:
            jobs.append(_rejected())
            continue
        live += 1
        gap = b_ms * live
        ttft = prompt * a_ms + gap
        if ttft > timeout_ms and move != "stop_retries":
            jobs.append(_rejected())
            live -= 1
            continue
        jobs.append(_done(ttft, gap, output))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _goodput(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    rows = _rows(scenario)
    batch = float(len(rows))
    step = b_ms
    slots = _req_float(scenario, "kv_slots")
    if move == "cap4":
        batch = slots
    if move == "pack12":
        batch = float(len(rows))
    if move == "faster_decode_pack":
        step = b_ms * 0.5
    if move == "admit6":
        batch = float(math.ceil(len(rows) / 2))
    gap = step * batch
    jobs: list[_Job] = []
    for index, row in enumerate(rows):
        if (move == "cap4" and index >= slots) or (move == "admit6" and index >= batch):
            jobs.append(_rejected())
            continue
        prompt = _req_float(row, "prompt_tokens")
        jobs.append(_done(prompt * a_ms + gap, gap, _req_float(row, "output_tokens")))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _day4_prefix(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    prefix_tokens = _opt_float(scenario, "prefix_tokens", 4000)
    jobs: list[_Job] = []
    for index, row in enumerate(_rows(scenario)):
        prompt = _req_float(row, "prompt_tokens")
        if move == "shorter_prompts":
            prompt = float(math.ceil(prompt * 0.25))
        prefill_tokens = prompt
        if move == "enable_prefix" and index > 0:
            prefill_tokens = max(0.0, prompt - prefix_tokens)
        if move == "more_engines" and index > 0:
            prefill_tokens = prompt
        gap = b_ms * (0.5 if move == "faster_decode" else 1.0) * min(index + 1, 4)
        jobs.append(_done(prefill_tokens * a_ms + gap, gap, _req_float(row, "output_tokens")))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _priority(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    preempt = move == "preempt_batch"
    fast = move == "faster_decode"
    admit_cap = 2.0 if move == "admit_fewer" else math.inf
    jobs: list[_Job] = []
    live = 0
    for row in _rows(scenario):
        interactive = row.get("priority") == "interactive"
        if live >= admit_cap:
            jobs.append(_rejected())
            continue
        live += 1
        concurrent = float(live)
        if preempt and interactive:
            concurrent = float(max(1, live - 1))
        if move == "more_kv":
            concurrent = float(live)
        if not preempt and not interactive:
            steal = 1.0
        elif not preempt and interactive:
            steal = float(live)
        else:
            steal = concurrent
        factor = 0.5 if fast else 1.0
        real_gap = b_ms * factor * (1.0 if preempt and interactive else steal)
        prompt = _req_float(row, "prompt_tokens")
        jobs.append(_done(prompt * a_ms + real_gap, real_gap, _req_float(row, "output_tokens")))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _sticky(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    prefix_tokens = _opt_float(scenario, "prefix_tokens", 3000)
    seen: dict[object, bool] = {}
    jobs: list[_Job] = []
    for index, row in enumerate(_rows(scenario)):
        prompt = _req_float(row, "prompt_tokens")
        prefill_tokens = prompt
        if move == "sticky_route" and index > 0:
            prefill_tokens = max(0.0, prompt - prefix_tokens)
        elif move == "pin_user":
            user = row.get("user_id")
            if user in seen:
                prefill_tokens = max(0.0, prompt - prefix_tokens)
            seen[user] = True
        elif move == "faster_prefill":
            prefill_tokens = float(math.ceil(prompt * 0.5))
        jobs.append(_done(prefill_tokens * a_ms + b_ms, b_ms, _req_float(row, "output_tokens")))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _hog(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    cap_out = move == "cap_output"
    preempt = move == "preempt_long"
    fast = move == "faster_decode"
    max_output = _opt_float(scenario, "max_output", 128)
    width = float(len(_rows(scenario)))
    jobs: list[_Job] = []
    for row in _rows(scenario):
        output = _req_float(row, "output_tokens")
        if move == "reject_shorts" and output <= 64:
            jobs.append(_rejected())
            continue
        limited = output
        if cap_out and limited > max_output:
            limited = max_output
        concurrent = width
        if preempt and output <= 64:
            concurrent = 2
        if cap_out:
            concurrent = float(min(width, 4))
        if move == "more_capacity":
            concurrent = 2
        gap = b_ms * (0.5 if fast else 1.0) * concurrent
        prompt = _req_float(row, "prompt_tokens")
        jobs.append(_done(prompt * a_ms + gap, gap, limited))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _frag(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    paged = move == "paged_blocks"
    buy_mem = move == "buy_memory"
    lower_batch = move == "lower_batch"
    slots = _req_float(scenario, "kv_slots")
    hole = math.floor(slots * 0.5)
    fragmented = not paged and not buy_mem
    jobs: list[_Job] = []
    live = 0.0
    for row in _rows(scenario):
        free_slots = slots * 2 if buy_mem else slots
        if paged:
            usable = free_slots - live
        elif fragmented and live >= hole:
            usable = 0.0
        else:
            usable = free_slots - live
        if usable < 1 or (lower_batch and live >= 2):
            if lower_batch and live >= 2:
                jobs.append(_rejected())
                continue
            if not paged and not buy_mem and live >= hole:
                jobs.append(_rejected())
                continue
        if live >= free_slots:
            jobs.append(_rejected())
            continue
        live += 1
        gap = b_ms * live
        prompt = _req_float(row, "prompt_tokens")
        jobs.append(_done(prompt * a_ms + gap, gap, _req_float(row, "output_tokens")))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _day4_spec(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, base_gap = _rates(scenario)
    move = choice
    draft_k = _opt_float(scenario, "draft_k", 4)
    alpha = _opt_float(scenario, "accept", 0.2)
    if move == "raise_accept":
        alpha = 0.8
    if move == "bigger_k":
        draft_k = 16
    if move == "faster_draft":
        alpha = 0.2
    if move == "disable_spec":
        draft_k = 1
        alpha = 1
    draft_cost = base_gap * 0.25 if move == "faster_draft" else base_gap * 0.5
    gained = 1 + alpha * (draft_k - 1)
    if move == "disable_spec":
        gap = base_gap
    else:
        gap = (draft_cost * draft_k + base_gap) / max(gained, 0.01)
    row = _require_rows(scenario)[0]
    prompt = _req_float(row, "prompt_tokens")
    job = _done(prompt * a_ms + gap, gap, _req_float(row, "output_tokens"))
    return _finish([job], scenario, drop_null_ttft=True)


def _evict(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    rows = _rows(scenario)
    if move == "grow_cache":
        hit_rate = 0.85
    elif move == "pin_hot":
        hit_rate = 0.75
    elif move == "faster_prefill":
        hit_rate = 0.1
    elif move == "more_replicas":
        hit_rate = 0.15
    else:
        hit_rate = _opt_float(scenario, "cache_blocks", 4) / max(len(rows), 1)
    prefix_tokens = _opt_float(scenario, "prefix_tokens", 2000)
    hits_allowed = math.floor((len(rows) - 1) * hit_rate)
    jobs: list[_Job] = []
    for index, row in enumerate(rows):
        prompt = _req_float(row, "prompt_tokens")
        if index > 0 and index <= hits_allowed:
            prefill_tokens = max(0.0, prompt - prefix_tokens)
        elif move == "faster_prefill":
            prefill_tokens = float(math.ceil(prompt * 0.5))
        else:
            prefill_tokens = prompt
        jobs.append(_done(prefill_tokens * a_ms + b_ms, b_ms, _req_float(row, "output_tokens")))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _lora(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    swap_ms = _opt_float(scenario, "swap_ms", 40)
    rows = _rows(scenario)
    if move == "queue_by_adapter":
        ordered = sorted(rows, key=lambda row: _req_float(row, "adapter"))
        jobs: list[_Job] = []
        for index, row in enumerate(ordered):
            adapter = row.get("adapter")
            first = next(i for i, item in enumerate(ordered) if item.get("adapter") == adapter)
            swap = swap_ms if first == index else 0.0
            prompt = _req_float(row, "prompt_tokens")
            jobs.append(_done(swap + prompt * a_ms + b_ms, b_ms, _req_float(row, "output_tokens")))
        return _finish(jobs, scenario, drop_null_ttft=True)

    resident = _opt_float(scenario, "unique_adapters", 4)
    loaded: dict[object, None] = {}
    jobs = []
    for index, row in enumerate(rows):
        adapter = row.get("adapter")
        swap = 0.0
        if move == "merge_one":
            swap = 0.0
        elif move == "pin_adapters":
            if adapter not in loaded and len(loaded) >= resident:
                del loaded[next(iter(loaded))]
            if adapter not in loaded:
                swap = swap_ms
                loaded[adapter] = None
        else:
            prev = None if index == 0 else rows[index - 1].get("adapter")
            if adapter != prev:
                swap = swap_ms
        gap = b_ms * (0.5 if move == "faster_decode" else 1.0)
        prompt = _req_float(row, "prompt_tokens")
        jobs.append(_done(swap + prompt * a_ms + gap, gap, _req_float(row, "output_tokens")))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _scale(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    rows = _rows(scenario)
    slots = _req_float(scenario, "kv_slots")
    engines = 1.0
    if move == "prewarm":
        engines = 3.0
    if move == "faster_scale":
        engines = 3.0
    if move == "admit_shed":
        engines = 1.0
    if move == "bigger_machine":
        engines = 1.0
    warm = _opt_float(scenario, "warm_ms", 3000)
    if move == "prewarm":
        ready_at = 0.0
    elif move == "faster_scale":
        ready_at = 500.0
    else:
        ready_at = warm
    jobs: list[_Job] = []
    for index, row in enumerate(rows):
        if move == "admit_shed" and index >= slots:
            jobs.append(_rejected())
            continue
        arrival = _req_float(row, "arrival_time_ms")
        if move == "prewarm" or (move == "faster_scale" and arrival >= ready_at):
            available = engines
        else:
            available = 1.0
        if move == "bigger_machine":
            conc = float(min(len(rows), 8))
            gap_cap = 8.0
        else:
            conc = float(math.ceil((index + 1) / max(available, 1.0)))
            gap_cap = 4.0
        queue = 0.0 if move == "prewarm" else max(0.0, (index - available + 1) * 40)
        gap = b_ms * min(conc, gap_cap)
        prompt = _req_float(row, "prompt_tokens")
        jobs.append(_done(queue + prompt * a_ms + gap, gap, _req_float(row, "output_tokens")))
    return _finish(jobs, scenario, drop_null_ttft=True)


def _batch(scenario: dict[str, object], choice: str | None) -> SimResult:
    a_ms, b_ms = _rates(scenario)
    move = choice
    rows = _rows(scenario)
    continuous = move == "continuous"
    if move == "bigger_static":
        fill_target = 8
    elif move == "smaller_static":
        fill_target = 2
    else:
        fill_target = 4
    gap_mul = 0.5 if move == "faster_decode" else 1.0
    jobs: list[_Job] = []
    for index, row in enumerate(rows):
        arrival = _req_float(row, "arrival_time_ms")
        prompt = _req_float(row, "prompt_tokens")
        if continuous:
            start = arrival
            gap = b_ms * gap_mul * min(index + 1, 4)
        else:
            idx_in_batch = index % fill_target
            batch_ready = index - idx_in_batch + fill_target - 1
            last = rows[min(batch_ready, len(rows) - 1)]
            start = _req_float(last, "arrival_time_ms")
            gap = b_ms * gap_mul * fill_target
        jobs.append(
            _done(start - arrival + prompt * a_ms + gap, gap, _req_float(row, "output_tokens"))
        )
    return _finish(jobs, scenario, drop_null_ttft=True)


def _labs_prefix(scenario: dict[str, object], repair: str | None) -> SimResult:
    a_ms = _req_float(scenario, "prefill_us_per_token") / 1000
    if repair == "prefill":
        a_ms *= 0.5
    rows = _rows(scenario)
    gap = _req_float(scenario, "decode_us_per_token") / 1000 * len(rows)
    if repair == "decode":
        gap *= 0.5
    jobs: list[_Job] = []
    for row in rows:
        prompt = _req_float(row, "prompt_tokens")
        if repair == "prefix":
            prompt -= _req_float(row, "shared_prefix_tokens")
        jobs.append(_done(prompt * a_ms + gap, gap, _req_float(row, "output_tokens")))
    return _finish(jobs, scenario, drop_null_ttft=False)


def _labs_spec(scenario: dict[str, object], repair: str | None) -> SimResult:
    row = _require_rows(scenario)[0]
    prompt = _req_float(row, "prompt_tokens")
    output = _req_float(row, "output_tokens")
    prefill = prompt * (_req_float(scenario, "prefill_us_per_token") / 1000)
    verify_us = _req_float(scenario, "verify_us")
    if repair == "off":
        target_ms = verify_us / 1000
        job = _done(prefill + target_ms, target_ms, output)
        return _finish([job], scenario, drop_null_ttft=False)
    draft = _req_float(scenario, "draft_len")
    draft_us = _req_float(scenario, "draft_us_per_token")
    accept_len = _req_float(scenario, "accept_len")
    if repair == "match":
        draft = accept_len
    if repair == "faster":
        draft_us *= 0.5
    committed = min(accept_len, draft)
    if committed <= 0:
        raise ValueError("spec committed length must be positive")
    round_ms = (draft * draft_us + verify_us) / 1000
    gap = round_ms / committed
    return _finish([_done(prefill + round_ms, gap, output)], scenario, drop_null_ttft=False)


def _device(scenario: dict[str, object], repair: str | None) -> SimResult:
    row = _require_rows(scenario)[0]
    compute = _req_float(scenario, "compute_ms")
    link = _req_float(scenario, "link_ms")
    if repair == "device":
        compute = _req_float(scenario, "local_ms")
        link = 0.0
    if repair == "decode":
        compute *= 0.5
    if repair == "link":
        link *= 0.5
    gap = compute + link
    prefill_us = _req_float(scenario, "prefill_us_per_token")
    prefill = _req_float(row, "prompt_tokens") * (prefill_us / 1000)
    job = _done(prefill + gap, gap, _req_float(row, "output_tokens"))
    return _finish([job], scenario, drop_null_ttft=False)


def _cold(scenario: dict[str, object], repair: str | None) -> SimResult:
    a_ms = _req_float(scenario, "prefill_us_per_token") / 1000
    if repair == "prefill":
        a_ms *= 0.5
    gap = _req_float(scenario, "decode_us_per_token") / 1000
    replicas = 2 if repair == "scale" else 1
    load = 0.0 if repair == "warm" else _req_float(scenario, "cold_ms")
    rows = _rows(scenario)
    cap = math.ceil(len(rows) / 2) if repair == "admission" else len(rows)
    lanes = [load] * replicas
    jobs: list[_Job] = []
    done_count = 0
    for index, row in enumerate(rows):
        if done_count >= cap and repair == "admission":
            jobs.append(_rejected())
            continue
        arrival = _req_float(row, "arrival_time_ms")
        prompt = _req_float(row, "prompt_tokens")
        lane = index % replicas
        start = max(lanes[lane], arrival)
        done_at = start + prompt * a_ms
        lanes[lane] = done_at
        jobs.append(_done(done_at + gap - arrival, gap, _req_float(row, "output_tokens")))
        done_count += 1
    return _finish(jobs, scenario, drop_null_ttft=False)


_KERNELS = {
    "admission": _admission,
    "admit": _admit,
    "queue": _queue,
    "chunk": _chunk,
    "retry": _retry,
    "goodput": _goodput,
    "day4_prefix": _day4_prefix,
    "priority": _priority,
    "sticky": _sticky,
    "hog": _hog,
    "frag": _frag,
    "day4_spec": _day4_spec,
    "evict": _evict,
    "lora": _lora,
    "scale": _scale,
    "batch": _batch,
    "labs_prefix": _labs_prefix,
    "labs_spec": _labs_spec,
    "device": _device,
    "cold": _cold,
}


def simulate(scenario: dict[str, object], repair: str | None = None) -> SimResult:
    """Run one laptop serving scenario under ``repair`` (``None`` keeps the fault)."""

    if not isinstance(scenario, dict):
        raise TypeError("scenario must be an object")
    kernel = _kernel(scenario, repair)
    runner = _KERNELS[kernel]
    return runner(scenario, _choice(kernel, scenario, repair))
