# 0003. Document parsing, storage, dedupe and chunking

Accepted with evidence · 2026-11-09 (proposed 2026-11-08) · Spec §1.3A, §5.3, §8 item 9

## Context

Spec §1.3A requires two things:
- **Traceability:** every chunk must trace back to its document and its location in it.
- **Idempotent re-ingestion:** re-ingesting a document must not create duplicates.

The inputs are PDF, TXT/Markdown and public URLs. The demo corpus is scientific papers (ADR-0013), so section structure matters. The parser must be license-compatible with an open portfolio repo and light enough for a 16 GB laptop.

## Options considered: PDF extraction

All entries checked 2026-11-08.

| Library | License | Page / heading / offsets | Weight | Assessment |
|---|---|---|---|---|
| PyMuPDF 2.1 (+ pymupdf4llm) | **AGPL-3.0** or commercial | Fast; header detection; boxes | No ML | The AGPL network clause would cover the whole service. Rejected as the default. |
| **pdfplumber 0.11.10** (pdfminer.six) | **MIT** | Per-page text, with per-character position and font size | Light, no torch | Rich enough to build page-accurate offsets and a font-size heading heuristic. **Default.** |
| pypdf 7.0 | BSD-3 | Page text only | Pure Python | Used for PDF metadata and validation: encryption, page count, `/Info`. |
| docling 2.148 | MIT (code), separate model licenses | Excellent provenance (`page_no`, bbox, charspan, section levels) | Pulls torch, layout models and OCR | Too heavy for the default path. Possible future `Parser` plugin. |
| marker-pdf 2.1 | Code Apache-2.0; **weights under modified OpenRAIL-M** with commercial limits | Good | torch + surya | Rejected because of the weights license and weight. |
| unstructured OSS 0.27 | Apache-2.0 | Element metadata | spacy, numba, poppler, tesseract | The docs say it is "not designed for production". Rejected. |
| GROBID 0.9.1 (`-crf` image, ~500 MB) | Apache-2.0 (UNVERIFIED on a primary page) | TEI XML: title, authors, sections, references, for scholarly PDFs | Java service, about 4 GB RAM for full text | **Optional enricher**, behind the `grobid` compose profile. It supplies the section tree and bibliographic metadata for papers. |

## Decision

### Parsing

- **Parser protocol.** Every parser implements `Parser.parse(raw: bytes, hint) -> ParsedDocument`.
- **What `ParsedDocument` holds:**
  - The normalized full text.
  - A list of `pages` as char ranges into that text.
  - A `sections` tree, each section with a title, level and char range.
  - The extracted metadata.
  - An `extractor` name and version.
- **PDF:**
  - pdfplumber extracts the text. It is processed page by page, and page boundaries are kept as char offsets into the normalized text.
  - pypdf handles metadata and validation.
  - Headings come from a font-size/weight heuristic.
  - If the `grobid` profile is enabled, GROBID's section tree and header metadata (title, authors, dates) override the heuristic, and the source of each field is recorded.
- **TXT / Markdown:**
  - Markdown headings become sections.
  - A "page" is the whole file (`page = null`). Locations are reported as section plus char offsets.
- **URL:**
  - Extracted with **trafilatura 2.3** (Apache-2.0 since 1.8).
  - The fetch runs through an SSRF guard, following OWASP:
    - Only `http` and `https`.
    - DNS is resolved once, and loopback, RFC 1918, link-local, ULA and metadata IPs are rejected for IPv4 and IPv6.
    - The connection goes to the validated IP.
    - Each redirect hop is re-validated, with at most 3 hops.
    - Timeout 15 s and a size cap of 25 MB.
    - Content-type allowlist: `text/html`, `application/pdf`, `text/plain`. A PDF response is routed to the PDF parser.
- **Validation:**
  - Magic bytes must match the declared type.
  - Size cap is 50 MB.
  - Page cap is 1,000.
  - Encrypted PDFs are rejected with a clear error.
  - Files are never executed or rendered server-side beyond text extraction.
  - The extracted text is NFC-normalized, and control characters are stripped.
- **No OCR in v1.** A scanned PDF with no text layer fails with `no_text_layer`; we do not ingest it silently as empty.

### Raw file storage

- **Store.** A **content-addressed local volume**, `blobs/sha256/ab/cd/<sha256>`, behind a `BlobStore` protocol (`put`, `get`, `exists`, `delete`).
- **Why not MinIO.** It would add a service, and we have only a single node and no multi-host need. An S3/MinIO implementation of the same protocol can be added when we deploy to the cloud (ADR-0014).
- **Retention.** Deleting a workspace deletes its rows, Qdrant points and BM25 index entries. A blob is garbage-collected once no workspace references it.

### Identity and dedupe

- **`raw_sha256`** is the hash of the uploaded bytes. Within a workspace, uploading the same bytes again is a no-op that returns the existing `document_version`. The status is `duplicate`, and nothing is re-chunked.
- **`text_sha256`** is the hash of the normalized text. If the bytes differ but the text is the same (for example, the same paper re-saved), the user sees "probable duplicate of X". The system does **not** auto-merge.
- **Re-processing.** A new chunker or embedder version re-processes existing versions under a new `index_version`. Old points are deleted only after the new index is complete, with an atomic switch per workspace.
- **Versions.** The `document` vs `document_version` split is defined in ADR-0010.

### Chunking (structure-aware, provenance-first)

1. **Split on section boundaries first.** A chunk never crosses a section boundary.
2. **Window inside a section** with a recursive splitter: paragraph, then sentence, then token window. The target is **~350 tokens** (counted with the embedder's tokenizer), with **~50 tokens of overlap** and a hard maximum of 512.
   - The 512 cap matches the input limit of the shortlisted encoders (ADR-0006).
   - Size and overlap are configuration values and are swept in M2/M3. Only the default is fixed here.
3. **Prefix each chunk with context for embedding only:** `"{doc title} > {section path}\n"`. The stored text and its offsets refer to the original span; the prefix is never cited.
4. **Provenance record per chunk:**
   - `chunk_id` = `sha256(document_version_id, chunker_version, ordinal)`, which is deterministic.
   - `document_id` and `document_version_id`.
   - `char_start` and `char_end` in the normalized text.
   - `page_start` and `page_end`, derived from the page char ranges.
   - `section_path`, for example `["3 Method", "3.2 Training"]`.
   - `token_count`.
   - `text` (stored in Postgres).
   - `text_sha256`.
   - `chunker_version`.
5. **References and bibliography sections** are tagged `section_kind=references` and excluded from retrieval by default. They are noisy for both BM25 and dense retrieval. This is configurable.

## Amendment, 2026-11-09: PDF extractor decided with evidence (decisions D6 and D7)

D6 allows AGPL and asks for this choice to be measured, keeping the simpler option unless the difference is measurable. **PyMuPDF replaces pdfplumber as the extractor.** Measured by `experiments/spikes/pdf_extractors.py` on the latest version of 30 papers from the demo corpus, against arXiv-rendered HTML of the same version as an independent reference. Raw per-paper results and CIs: `docs/eval/pdf_extractors.json`.

| Extractor | Reference 5-gram recall (95% CI) | Extracted 5-gram precision (95% CI) | Seconds per page |
|---|---|---|---|
| **PyMuPDF 1.28** | **0.838** (0.816, 0.859) | **0.654** (0.619, 0.687) | 0.004 |
| pdfplumber, tuned (column cropping, `x_tolerance=1`) | 0.754 (0.723, 0.785) | 0.580 (0.539, 0.620) | 0.195 |
| pdfplumber, column cropping only | 0.231 (0.173, 0.294) | 0.253 (0.199, 0.313) | 0.238 |
| pdfplumber, default | 0.180 (0.131, 0.238) | 0.198 (0.151, 0.251) | 0.181 |

The paired difference against the best pdfplumber arm is +0.084 recall (CI 0.064, 0.105) and +0.074 precision (CI 0.058, 0.090). PyMuPDF is also about 40 times faster, and it reads two-column papers correctly without a column-cropping heuristic. Default-configuration pdfplumber glues words together on these LaTeX PDFs, so the tuned arm is the fair comparison.

**Version.** `pymupdf>=1.28.2,<2`. PyMuPDF 2.x pulls in onnxruntime, and placeholder `pymupdf4llm` and `pymupdf-layout` packages whose metadata carries no licence, for layout code Mezvuat does not use. The earlier 1.x `pymupdf-layout` was listed as PolyForm Noncommercial, which is incompatible with this repository (secondary sources; not confirmed from a licence file). PyMuPDF itself is AGPL-3.0, which matches the repository licence (D7).

**Limitations.** 30 papers, one domain. Precision is capped well below 1 because the reference omits figures, footers and references that PDF extraction keeps. Trafilatura extraction of the reference may itself lose text. Heading detection is a font-size and weight heuristic: it was checked on generated PDFs and one real PDF, not measured across the corpus.

`pdfplumber` is now a development-only dependency used by this spike.

## Consequences

- No AGPL code ships in the default build.
- The heading heuristic on raw pdfplumber output is imperfect. GROBID fixes this for papers, at a RAM cost.
- Tables and figures come out as plain text with no structure. This is a known limitation, documented in the UI.
- With content-addressed blobs, identical files across workspaces share storage, while access control stays per workspace in Postgres.

## Verification plan

- **Golden tests on 3–5 committed, permissively licensed sample PDFs:**
  - For every chunk, `normalized_text[char_start:char_end] == chunk.text`.
  - The page of a known sentence matches a hand-checked page number.
  - Sections are detected for a known paper.
- **Idempotency test:** ingesting the same file twice yields one version, unchanged chunk counts, and `duplicate` status on the second attempt.
- **SSRF tests:** `http://127.0.0.1`, `http://169.254.169.254`, `http://[::1]`, a redirect to a private IP, and a DNS name resolving to a private IP are all rejected.
- **M2 comparison:** BEIR SciFact nDCG@10 for chunk sizes 256, 350 and 512 (corpus documents are abstracts, so this is mostly a sanity check). On the domain set (M3), the sweep picks the default using the validation split only.

## Sources (accessed 2026-11-08)

- PyPI JSON for pymupdf, pymupdf4llm, pdfplumber, pypdf, docling, docling-slim, marker-pdf, unstructured and trafilatura.
- docling-project.github.io/docling/concepts/docling_document/
- github.com/datalab-to/marker
- grobid.readthedocs.io/en/latest/Grobid-docker/
- OWASP SSRF Prevention Cheat Sheet.
