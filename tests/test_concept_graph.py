"""Concept-graph structure, anchors, and clock purity."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from helpers import EXPECTED_IDS, REPO, concept_documents, load_concept_dag

from principia_inferentiae.practice.dag import (
    LEDGER_EVIDENCE_IDS,
    ConceptDAG,
    GradingType,
    GraphError,
)

PRACTICE = REPO / "src" / "principia_inferentiae" / "practice"


def _slug(heading: str) -> str:
    cleaned = heading.strip().lower()
    chars: list[str] = []
    for char in cleaned:
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "-"}:
            chars.append("-")
    slug = "".join(chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def test_exact_concept_ids_and_domains() -> None:
    dag = load_concept_dag()
    assert {concept.id for concept in dag.topological()} == EXPECTED_IDS
    assert len(dag) == 63
    for concept in dag.topological():
        assert (
            concept.domain == concept.id.split(".", 1)[0]
            or (concept.domain == "model_memory" and concept.id.startswith("model."))
            or (concept.domain == "kv_cache" and concept.id.startswith("kv."))
        )


def test_graph_is_acyclic_and_prerequisites_resolve() -> None:
    dag = load_concept_dag()
    order = dag.topological_ids()
    position = {concept_id: index for index, concept_id in enumerate(order)}
    for concept in dag.topological():
        for prereq in concept.prerequisites:
            assert prereq in dag
            assert position[prereq] < position[concept.id]
        assert set(concept.key_prerequisites) <= set(concept.prerequisites)
        for covered in concept.encompasses:
            assert covered in dag
            assert covered != concept.id


def test_policy_concepts_are_intervene() -> None:
    dag = load_concept_dag()
    for concept in dag.topological():
        if concept.id.startswith("policy."):
            assert concept.grading_type is GradingType.INTERVENE


def test_ledger_ids_are_allowlisted() -> None:
    dag = load_concept_dag()
    for concept in dag.topological():
        if concept.ledger_evidence_id is not None:
            assert concept.ledger_evidence_id in LEDGER_EVIDENCE_IDS


def test_knowledge_page_anchors_exist() -> None:
    dag = load_concept_dag()
    for concept in dag.topological():
        page, separator, anchor = concept.knowledge_page.partition("#")
        assert separator == "#"
        text = (REPO / page).read_text(encoding="utf-8")
        headings = re.findall(r"^#{1,6} +(.+)$", text, flags=re.MULTILINE)
        assert anchor in {_slug(heading) for heading in headings}


def test_cycle_is_rejected() -> None:
    documents = concept_documents()
    documents.append(
        {
            "id": "units.bytes_prefixes",
            "label": "duplicate",
            "domain": "units",
            "subdomain": "units.prefixes",
            "grading_type": "numeric_units",
            "prerequisites": ["units.us_ms_s"],
            "knowledge_page": "knowledge/serving-metrics.md#theory-and-units",
        }
    )
    with pytest.raises(GraphError):
        ConceptDAG.from_documents(documents)


def test_practice_package_has_no_clock_or_filesystem() -> None:
    import principia_inferentiae.practice as practice_pkg

    assert Path(practice_pkg.__file__).resolve().parent == PRACTICE
    forbidden = ("datetime.now", "time.time", "os.environ", "open(")
    for path in PRACTICE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} contains {token}"
