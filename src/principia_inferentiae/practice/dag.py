"""Concept DAG: validation, topological order, and dependency indexes."""

from __future__ import annotations

from collections import deque
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DOMAIN_ORDER: tuple[str, ...] = (
    "units",
    "model_memory",
    "kv_cache",
    "phases",
    "metrics",
    "workload",
    "roofline",
    "sizing",
    "diagnosis",
    "policy",
)

DOMAIN_LABELS: dict[str, str] = {
    "units": "Units",
    "model_memory": "Model memory",
    "kv_cache": "KV cache",
    "phases": "Prefill and decode",
    "metrics": "Serving metrics",
    "workload": "Workload",
    "roofline": "Roofline",
    "sizing": "Sizing",
    "diagnosis": "Diagnosis",
    "policy": "Scheduling policy",
}

# Serving-critical competency ledger ids. The engine never writes the ledger.
LEDGER_EVIDENCE_IDS: frozenset[str] = frozenset(
    {
        "j-http-serve",
        "j-docker-port",
        "j-k8s-deploy-service",
        "j-load-generation",
        "j-serving-observability",
        "j-failure-recovery",
        "j-baseline-results",
        "e-runtime-metrics",
        "e-decode-bandwidth",
        "e-kv-prefix",
        "e-continuous-batching",
        "p-nsys-kernels",
    }
)


class GradingType(StrEnum):
    NUMERIC_UNITS = "numeric_units"
    CHOICE = "choice"
    CLASSIFY = "classify"
    ORDER = "order"
    INTERVENE = "intervene"
    FORMULA = "formula"


class Status(StrEnum):
    UNSEEN = "UNSEEN"
    LEARNING = "LEARNING"
    PRACTICING = "PRACTICING"
    MASTERED = "MASTERED"
    DECAYING = "DECAYING"


class GraphError(ValueError):
    """Structural concept-graph failures, joined for the graph test."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


class MasteryThresholdSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_floor: int = Field(default=6, ge=1)
    streak: int = Field(default=5, ge=1)
    avg_time_seconds: float | None = Field(default=None, gt=0)


class Concept(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
    label: str = Field(min_length=1)
    domain: str
    subdomain: str = Field(min_length=1)
    grading_type: GradingType
    prerequisites: tuple[str, ...] = ()
    key_prerequisites: tuple[str, ...] = ()
    encompasses: tuple[str, ...] = ()
    interference_group: str | None = None
    mastery_threshold: MasteryThresholdSpec = Field(default_factory=MasteryThresholdSpec)
    ledger_evidence_id: str | None = None
    knowledge_page: str = Field(min_length=1)
    tolerance: float = Field(default=0.02, ge=0)

    @model_validator(mode="after")
    def key_prereqs_are_prereqs(self) -> Self:
        unknown = set(self.key_prerequisites) - set(self.prerequisites)
        if unknown:
            raise ValueError(f"{self.id} key_prerequisites not in prerequisites: {sorted(unknown)}")
        return self


class ConceptDAG:
    """Acyclic concept graph. Construction fails closed on structural errors."""

    def __init__(self, concepts: tuple[Concept, ...]) -> None:
        errors: list[str] = []
        by_id: dict[str, Concept] = {}
        for concept in concepts:
            if concept.id in by_id:
                errors.append(f"duplicate concept id {concept.id}")
                continue
            by_id[concept.id] = concept
            if concept.domain not in DOMAIN_LABELS:
                errors.append(f"{concept.id} has unknown domain {concept.domain}")
            if concept.ledger_evidence_id is not None and (
                concept.ledger_evidence_id not in LEDGER_EVIDENCE_IDS
            ):
                errors.append(
                    f"{concept.id} ledger_evidence_id {concept.ledger_evidence_id} "
                    "is not in the allowlist"
                )
        for concept in by_id.values():
            for prereq in concept.prerequisites:
                if prereq not in by_id:
                    errors.append(f"{concept.id} prerequisite {prereq} does not exist")
            for covered in concept.encompasses:
                if covered not in by_id:
                    errors.append(f"{concept.id} encompasses unknown concept {covered}")
                if covered == concept.id:
                    errors.append(f"{concept.id} encompasses itself")
        if errors:
            raise GraphError(tuple(errors))
        self.by_id = by_id
        self._order = _topological_order(by_id)
        dependents: dict[str, tuple[str, ...]] = {concept_id: () for concept_id in by_id}
        buckets: dict[str, list[str]] = {concept_id: [] for concept_id in by_id}
        for concept in by_id.values():
            for prereq in concept.prerequisites:
                buckets[prereq].append(concept.id)
        for concept_id, children in buckets.items():
            dependents[concept_id] = tuple(sorted(children))
        self.dependents = dependents
        self._depths = _depths(self._order, by_id)

    @classmethod
    def from_documents(cls, documents: tuple[object, ...] | list[object]) -> ConceptDAG:
        concepts: list[Concept] = []
        for item in documents:
            concepts.append(Concept.model_validate(item))
        return cls(tuple(concepts))

    def __len__(self) -> int:
        return len(self.by_id)

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self.by_id

    def get(self, concept_id: str) -> Concept:
        return self.by_id[concept_id]

    def topological(self) -> tuple[Concept, ...]:
        return tuple(self.by_id[concept_id] for concept_id in self._order)

    def topological_ids(self) -> tuple[str, ...]:
        return self._order

    def depth(self, concept_id: str) -> int:
        return self._depths[concept_id]

    def in_domain(self, domain: str) -> tuple[Concept, ...]:
        return tuple(concept for concept in self.topological() if concept.domain == domain)


def _topological_order(by_id: dict[str, Concept]) -> tuple[str, ...]:
    indegree = {concept_id: 0 for concept_id in by_id}
    children: dict[str, list[str]] = {concept_id: [] for concept_id in by_id}
    for concept in by_id.values():
        for prereq in concept.prerequisites:
            if prereq not in by_id:
                continue
            children[prereq].append(concept.id)
            indegree[concept.id] += 1
    ready = deque(sorted(concept_id for concept_id, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while ready:
        concept_id = ready.popleft()
        order.append(concept_id)
        for child in sorted(children[concept_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(by_id):
        remaining = sorted(concept_id for concept_id in by_id if concept_id not in order)
        raise GraphError((f"cycle involving {', '.join(remaining)}",))
    return tuple(order)


def _depths(order: tuple[str, ...], by_id: dict[str, Concept]) -> dict[str, int]:
    depths: dict[str, int] = {}
    for concept_id in order:
        prereqs = by_id[concept_id].prerequisites
        depths[concept_id] = 0 if not prereqs else 1 + max(depths[prereq] for prereq in prereqs)
    return depths
