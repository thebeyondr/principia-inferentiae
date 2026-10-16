---
name: ingest-inference-source
description: Register one inference source deterministically, preserve provenance, and surface duplicate URLs or contradictory scoped facts for human review. Use when adding a paper, doc, or note to the knowledge base.
license: Apache-2.0
compatibility: Requires Python 3.11+ and the principia CLI.
---

# Ingest inference source

Use when adding one authoritative source or versioned fact record.

1. Prepare a validated `SourceEntry` YAML/JSON with locator-digest semantics.
2. Run:

   `principia ingest-source --catalog sources/catalog.yaml --entry <entry.yaml> --catalog-output <catalog.yaml> --result-output <review.json>`

3. Inspect `contradictions` and `duplicate_urls`; do not resolve conflicts by deletion.
4. Write original synthesis, citations, cross-links, and retrieval checks in `knowledge/`.
5. Human-review the diff before replacing the canonical catalog.

Never copy article prose or invent a source checksum. The CLI owns merge and conflict logic.

