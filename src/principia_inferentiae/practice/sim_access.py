"""Read scheduler results without redefining ``simulate``."""

from __future__ import annotations

_CLOCK_FIELD = {"ttft": "ttft_p99_ms", "itl": "itl_p99_ms"}


def run(scenario: dict[str, object], repair: str | None = None) -> object:
    # Imported on use so the analysis CLI can load before sim.py exists.
    from principia_inferentiae.sim import simulate

    return simulate(scenario, repair)


def metric(result: object, name: str) -> float | None:
    value: object
    if isinstance(result, dict):
        if name not in result:
            return None
        value = result[name]
    elif hasattr(result, name):
        value = getattr(result, name)
    else:
        return None
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def scenario_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("scenario must be an object")
    return {str(key): item for key, item in value.items()}


def _slo_limit(scenario: dict[str, object], clock: str) -> float:
    slo = scenario.get("slo")
    if not isinstance(slo, dict):
        raise TypeError("scenario.slo must be an object")
    key = "ttft_ms" if clock == "ttft" else "itl_ms"
    limit = slo.get(key)
    if isinstance(limit, bool) or not isinstance(limit, int | float):
        raise TypeError(f"scenario.slo.{key} must be numeric")
    return float(limit)


def bad_clock(scenario: dict[str, object], baseline: object) -> str:
    ttft = metric(baseline, "ttft_p99_ms")
    itl = metric(baseline, "itl_p99_ms")
    ttft_bad = ttft is None or ttft > _slo_limit(scenario, "ttft")
    itl_bad = itl is None or itl > _slo_limit(scenario, "itl")
    if itl_bad and not ttft_bad:
        return "itl"
    if ttft_bad and not itl_bad:
        return "ttft"
    declared = scenario.get("baseline_field")
    if declared in _CLOCK_FIELD:
        return str(declared)
    return "itl"


def _other_clock_ok(
    scenario: dict[str, object], baseline: object, repaired: object, other: str
) -> bool:
    before_other = metric(baseline, _CLOCK_FIELD[other])
    after_other = metric(repaired, _CLOCK_FIELD[other])
    if before_other is None or after_other is None:
        return False
    if after_other > before_other * 1.02 + 0.5:
        return False
    other_limit = _slo_limit(scenario, other)
    return not (before_other <= other_limit < after_other)


def separates(scenario: dict[str, object], baseline: object, repaired: object) -> bool:
    """Bad clock is inside its SLO and the other clock does not regress.

    Prefix / sticky / evict leave one cold first request, so nearest-rank p99 can
    stay out of SLO while goodput jumps. Those pages separate on goodput when the
    repaired share in SLO clears a high bar and neither clock regresses.
    """

    target = bad_clock(scenario, baseline)
    other = "itl" if target == "ttft" else "ttft"
    after = metric(repaired, _CLOCK_FIELD[target])
    before = metric(baseline, _CLOCK_FIELD[target])
    if after is not None and after <= _slo_limit(scenario, target):
        return _other_clock_ok(scenario, baseline, repaired, other)
    if not _other_clock_ok(scenario, baseline, repaired, other):
        return False
    if before is None or after is None or after > before * 1.02 + 0.5:
        return False
    base_gp = metric(baseline, "goodput")
    after_gp = metric(repaired, "goodput")
    if base_gp is None or after_gp is None:
        return False
    return base_gp < 0.5 and after_gp >= 0.75 and (after_gp - base_gp) >= 0.25
