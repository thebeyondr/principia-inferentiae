"""Importing this package registers one generator per concept."""

from . import diagnosis as diagnosis
from . import kv_cache as kv_cache
from . import metrics as metrics
from . import model_memory as model_memory
from . import phases as phases
from . import policy as policy
from . import roofline as roofline
from . import sizing as sizing
from . import units as units
from . import workload as workload
from .registry import GENERATORS, self_check

__all__ = ["GENERATORS", "self_check"]

_REGISTERED = (
    diagnosis,
    kv_cache,
    metrics,
    model_memory,
    phases,
    policy,
    roofline,
    sizing,
    units,
    workload,
)
