# Contribution contract

## Authority boundaries

1. Markdown explains and cites; it never becomes an alternate calculator.
2. `sources/` and `data/catalogs/` record versioned facts with units and provenance.
3. `src/principia_inferentiae/` is the only authority that computes claims.
4. `.agents/skills/` orchestrates public CLI commands only. Keep formulas out of prompts.
5. Generated reports are views over validated JSON and may not add conclusions.

## Evidence rules

- Fail closed on missing, stale, mismatched, or contradictory evidence.
- Preserve nulls and `insufficient_evidence`; never replace them with an agent estimate.
- Distinguish HBM capacity from HBM bandwidth.
- Apply roofline only to a measured kernel or named phase with compatible empirical ceilings.
- Keep TTFT, ITL, TPOT, E2E, per-user TPS, aggregate TPS, and goodput separate.
- Keep quality selection separate from deployment sizing.
- Root-cause claims require the complete controlled-intervention causal gate.
- All time, version, environment, and provenance data must arrive through inputs.

## Knowledge-page convention

Every focused page contains:

1. mental model;
2. theory and equations with units;
3. what must be measured;
4. assumptions and failure modes;
5. discriminating experiments;
6. retrieval checks;
7. authoritative source links.

Write original concise synthesis. Do not mirror Outcome School or any other article.

## Source ingestion

Use `.agents/skills/ingest-inference-source/`. Record a canonical locator, revision,
verification date, usage note, and digest semantics. A locator digest is not a content
digest; label it exactly. Conflicting scoped facts remain side by side until a human
reviews revision, units, scope, and measurement method. Never silently overwrite.

## Catalog changes

- Units belong in field names or typed unit fields.
- Vendor facts require a cited revision and source digest semantics.
- Prefer clearly labeled synthetic fixtures when authoritative values are unavailable.
- Changing a fact revision requires tests or regenerated case-study outputs as applicable.

## Determinism and review

Before handoff:

```bash
ruff check .
ruff format --check .
mypy src tests
pytest
principia analyze --help
```

Generated output must be byte-stable for reordered semantically identical inputs. Do not
read implicit wall-clock time, Git state, host hardware, environment variables, or the
network in analysis code.

## Activity log

`knowledge/activity-log.md` is append-only. Add a dated entry only when the date is an
explicit human input. Record changed source IDs, fact revisions, method revisions,
generated artifacts, contradiction decisions, and reviewer identity. Never rewrite
earlier entries; append a superseding entry.

