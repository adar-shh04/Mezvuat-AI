# 0010. Temporal metadata model

Proposed · 2026-11-08 · Spec §1.3A, §1.3E, §1.3G (temporal accuracy)

## Context

Publication, effective, revision and ingestion dates are distinct, and the spec forbids treating them as interchangeable. The system must not claim temporal correctness when metadata is missing.

Later capabilities in scope:
- What changed between versions.
- Which source is more recent.
- Whether a statement held at a given date.
- Which documents contradict each other.

## Options considered

1. **Logical document → versions, with typed dates, each carrying provenance and precision, plus a relationship table.** Chosen.
2. A flat document table with a single `date` column. This conflates the four kinds of date and violates the spec.
3. A full bitemporal model (valid time × transaction time for every row). Overkill. Ingestion timestamps already give us transaction time.

## Decision

### Tables (Postgres, Alembic-managed)

**`document`** is the logical work, for example arXiv `2401.01234`.

| Column | Notes |
|---|---|
| `id` | |
| `workspace_id` | |
| `canonical_key` | Nullable, e.g. `arxiv:2401.01234` or a normalized URL |
| `title`, `authors[]`, `issuing_org` | Each is nullable |
| `created_at` | |

**`document_version`** is one concrete content state.

| Column | Notes |
|---|---|
| `id`, `document_id` | |
| `version_label` | Nullable, e.g. `v2` |
| `version_ordinal` | Nullable |
| `raw_sha256`, `text_sha256` | |
| `source_uri` | |
| `license` | Nullable |
| `ingested_at` | System clock, always known |
| `ingestion_status`, `error` | |
| `extractor_version`, `chunker_version` | |

**`temporal_fact`** has one row per (version, kind).

| Column | Notes |
|---|---|
| `document_version_id` | |
| `kind` | `published`, `revised`, `effective_from`, `effective_to` |
| `value` | `date` |
| `precision` | `day`, `month` or `year` |
| `source` | `source_metadata` (e.g. arXiv API), `extracted_text`, `pdf_info`, `user` |
| `confidence` | `high`, `medium` or `low` |
| `evidence` | Short text or field name explaining where the value came from |

- **A missing row means "unknown".** No NULL-as-default, and no fallback from one kind of date to another.
- Disagreeing sources are kept as multiple rows, and the API reports the conflict.

**`document_relation`**

| Column | Notes |
|---|---|
| `from_version_id`, `to_version_id` | `to_document_id` when the target is not ingested |
| `type` | `version_of`, `supersedes`, `cites`, `amends`, `contradicts` |
| `source` | `source_metadata`, `extracted`, `user`, `analysis` |
| `evidence` | JSON |

- `contradicts` with source `analysis` is only ever written from verified, cited claim pairs (ADR-0009).

**Derived values:**
- `is_latest_version` is set per document by `version_ordinal`, or by `revised` date when the ordinal is unknown. **If neither is known, it is unknown** and `is_latest_version` = NULL.
- It is mirrored into the Qdrant payload and the BM25 filter metadata.

### Semantics

- **"Latest".** Determined by `version_ordinal`, then by the `revised`/`published` date. `ingested_at` is **never** used.
- **"As of date D".**
  1. Pick the version with the greatest `revised` (or `published`) date ≤ D, among versions whose relevant date precision allows a decision.
  2. If the precision is too coarse (year-only, with D in the same year) or the date is unknown, the version is **excluded from as-of claims** and listed as "temporally unresolved".
  3. If `effective_from`/`effective_to` exist (for example in regulatory documents), they take precedence for applicability questions. Papers usually lack them, so they stay unknown.
- **Answers:**
  - Every cited source shows its known dates, labelled by kind.
  - When a temporal question depends on unknown dates, the limitations section says so explicitly.
- **Version diff** (later milestone): text diff between two versions' chunks, aligned by section path. The LLM summary of the diff is labelled as interpretation and cites both versions.

### Extraction sources, in priority order

1. Structured source metadata, such as arXiv OAI-PMH `arXivRaw` version dates (ADR-0013).
2. User-entered values.
3. PDF `/Info` dates, low confidence, since they often reflect file creation.
4. Dates in the text, such as "Published: …" on the title page, medium confidence, with the evidence quote stored.

## Amendment, 2026-11-14: as-of and latest in the research workflow (M5, part 1)

- `mezvuat.temporal` holds the rules and depends only on `core`. `ingested_at` is not a `DateFact` kind, so it cannot reach any temporal decision.
- **Ordering date.** A version is ordered by its best `revised` fact, else its best `published` fact. The best fact is chosen by source (`source_metadata` > `user` > `extracted_text` > `pdf_info`), then confidence.
- **Precision.** A date with year precision cannot decide a comparison within its year, and one with month precision cannot decide it within its month. Two candidates with the same date are both unresolved.
- **In the graph.** The `as_of` date comes from the run request, or else from the query analysis, which is then disclosed under limitations. Per document, assessment and synthesis see only the evidence from the applicable version. Unresolved versions, and documents with no known dates, are excluded and named under limitations. Versions that did not exist yet are excluded silently.
- **Latest.** When the question asks for the latest version, or the request sets `latest_only`, a document's latest flag decides which evidence is used. The flag comes from the version table, falling back to the chunk payload. A document whose latest version is unknown is kept, and the limitation says that the latest version is unknown.
- **Answers.** Every citation carries its version's dates. The answer ends with a deterministic `Sources` section that shows each cited version's dates by kind, or "dates unknown". The evidence headers in prompts show the same labels.
- **Not done yet.** The temporal question set of 30 or more questions with gold answers is still to build. It needs either generation on the public corpus (Groq quota) or the product owner's time.

## Consequences

- Queries must join `temporal_fact`. That is fine at this scale; a materialized view per version can be added if needed.
- Many documents will honestly show "unknown" dates. The UI must make that look normal, not broken.

## Verification plan

- **Unit tests:**
  - `ingested_at` never influences `is_latest_version` or as-of resolution.
  - Year-precision dates do not resolve a same-year as-of query.
  - Conflicting date sources are both returned.
- **Temporal eval set** (M5/M6): 30 or more questions over multi-version arXiv papers, with gold answers naming the correct version. Each is answerable only from a specific version, or requires saying "unknown".
  - `temporal_accuracy` = the fraction of answers that cite the correct version, or correctly abstain.
