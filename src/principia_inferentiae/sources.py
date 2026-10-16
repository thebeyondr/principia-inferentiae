"""Deterministic source-catalog ingestion and contradiction flagging."""

from __future__ import annotations

from pydantic import Field, model_validator

from principia_inferentiae.contracts import Digest, FrozenModel


class SourceFact(FrozenModel):
    fact_key: str
    value: str
    unit: str
    scope: str


class SourceEntry(FrozenModel):
    source_id: str
    title: str
    url: str
    authors: tuple[str, ...]
    publication_or_version_date: str
    source_type: str
    license_or_usage_note: str
    topics: tuple[str, ...]
    last_verified_date: str
    content_digest: Digest
    content_digest_semantics: str
    facts: tuple[SourceFact, ...] = ()


class SourceCatalog(FrozenModel):
    schema_version: str
    entries: tuple[SourceEntry, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> SourceCatalog:
        ids = [entry.source_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("source catalog IDs must be unique")
        return self


class FactConflict(FrozenModel):
    fact_key: str
    existing_source_id: str
    new_source_id: str
    existing_value: str
    new_value: str
    unit: str
    scope: str


class SourceIngestResult(FrozenModel):
    catalog: SourceCatalog
    contradictions: tuple[FactConflict, ...]
    duplicate_urls: tuple[str, ...]
    retrieval_checks: tuple[str, ...] = Field(min_length=1)


def ingest_source(catalog: SourceCatalog, entry: SourceEntry) -> SourceIngestResult:
    """Merge one entry without overwriting and surface review obligations."""

    if any(existing.source_id == entry.source_id for existing in catalog.entries):
        raise ValueError(f"source_id already exists: {entry.source_id}")

    duplicate_urls = tuple(
        sorted(existing.source_id for existing in catalog.entries if existing.url == entry.url)
    )
    conflicts: list[FactConflict] = []
    for new_fact in entry.facts:
        for existing in catalog.entries:
            for existing_fact in existing.facts:
                same_key = (
                    existing_fact.fact_key == new_fact.fact_key
                    and existing_fact.unit == new_fact.unit
                    and existing_fact.scope == new_fact.scope
                )
                if same_key and existing_fact.value != new_fact.value:
                    conflicts.append(
                        FactConflict(
                            fact_key=new_fact.fact_key,
                            existing_source_id=existing.source_id,
                            new_source_id=entry.source_id,
                            existing_value=existing_fact.value,
                            new_value=new_fact.value,
                            unit=new_fact.unit,
                            scope=new_fact.scope,
                        )
                    )
    entries = tuple(sorted((*catalog.entries, entry), key=lambda item: item.source_id))
    return SourceIngestResult(
        catalog=SourceCatalog(schema_version=catalog.schema_version, entries=entries),
        contradictions=tuple(
            sorted(conflicts, key=lambda item: (item.fact_key, item.existing_source_id))
        ),
        duplicate_urls=duplicate_urls,
        retrieval_checks=(
            "Write original synthesis; do not copy source prose.",
            "Add or update knowledge-page citations and retrieval questions.",
            "Review every surfaced contradiction before promoting a fact.",
            "Append the human review outcome to the project activity log.",
        ),
    )
