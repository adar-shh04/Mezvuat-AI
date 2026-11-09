# 0013. Demo corpus and domain

Proposed · 2026-11-08 · Spec §1.2, §1.3E

## Context

The demo needs a focused collection of a few hundred documents. It must be legally usable in a public portfolio, and it should exercise the hard features:
- multi-document comparison,
- versions and dates,
- technical terminology for BM25,
- conflicting findings.

The architecture stays domain-independent.

## Options considered

| Option | Temporal richness | Licensing | Metadata quality | Verdict |
|---|---|---|---|---|
| **arXiv papers, filtered to CC BY / CC BY-SA / CC0, with ≥ 2 versions** | Native v1…vN with dates for each version | Paper license is exposed in OAI-PMH metadata. Metadata is CC0. | Excellent: OAI-PMH `arXivRaw` lists every version date | **Chosen** |
| EU legislation (EUR-Lex): consolidated versions, amendments | Excellent: effective dates and amendments | EU reuse policy (UNVERIFIED for this use) | Good | Strong future second domain; proves domain independence |
| Public institutional reports (IPCC, central banks) | Annual editions | Varies by publisher | Mixed | Later |

## Decision

- **Domain:** arXiv `cs.IR` + `cs.CL` papers about retrieval-augmented generation, dense retrieval and reranking. The topic is focused, has plenty of terminology, and has genuine disagreements.
  - Bonus: the corpus is about the methods this system uses.
- **Selection:**
  - License ∈ {CC BY 4.0, CC BY-SA 4.0, CC0}, read from the OAI-PMH metadata.
  - ≥ 40% of papers with ≥ 2 versions.
  - Target **100–150 papers, about 250–400 versions**. That is about 10–20k chunks (EST), which fits ADR-0004 and ADR-0005.
  - A small set of "citing" papers that reference earlier ones in the corpus gives cross-document comparison.
- **Acquisition script** (M1/M3):
  1. Fetch metadata through OAI-PMH (`https://oaipmh.arxiv.org/oai`, `arXivRaw`) to get license and per-version dates.
  2. Fetch the PDF of each version from arXiv.
  3. Respect the arXiv terms: **at most 1 request every 3 s on a single connection**, and no bulk crawling of the full corpus.
  4. Cache results.
- **What is committed:** only the **manifest** (ids, versions, license, SHA-256). PDFs live in git-ignored `data/`. This follows arXiv's guidance not to serve e-prints yourself unless the license permits; CC licenses do permit it, but we still avoid committing binaries.
- **Temporal mapping (ADR-0010):**
  - v1 submission date → `published` (`source=source_metadata`).
  - vN date → `revised`.
  - No effective dates (unknown).
  - Journal publication date is unknown unless the user enters it.

## Amendment, 2026-11-09: metadata source

The OAI-PMH plan could not be used: both `ListRecords` and `GetRecord` on `oaipmh.arxiv.org` timed out during the build (HTTP 503 after about 120 s, 000 after 25 s). Candidates now come from the arXiv search API (titles, authors, categories, abstracts), and licence and version history from each paper abstract page, at one request every 3 s on one connection. Parsed in `mezvuat.ingestion.corpus.oai`. The selection rules and the D1 allowlist are unchanged: the allowlist is keyed by the SHA-256 of downloaded bytes, so the metadata source cannot make a document public.

## Consequences

- Real versioning comes free, and every temporal claim can be checked against arXiv metadata.
- English-only.
- Filtering to CC-licensed papers shrinks the pool (most arXiv papers use the non-exclusive license). Whether 100–150 suitable papers exist in this topic is verified when the manifest is built. If not, we widen to adjacent cs categories.
- **Before any download, the user is asked to approve the total size**, about 0.5–1.5 GB (EST).

## Verification plan

- **Manifest builder test** using recorded OAI-PMH fixtures: licenses are parsed, non-CC papers are excluded, and version dates are extracted.
- **Rate-limiter test:** at least 3 s between requests.

## Sources (accessed 2026-11-08)

- info.arxiv.org/help/api/tou.html
- info.arxiv.org/help/bulk_data.html
- info.arxiv.org/help/license/index.html
- info.arxiv.org/help/oa/index.html
- Sample record: oaipmh.arxiv.org GetRecord for 1706.03762 in arXivRaw format
