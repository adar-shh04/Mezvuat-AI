# 0009. Citations, claim verification and prompt-injection defence

Proposed · 2026-11-08 · Spec §1.3F, §1.1 promise, §5.3

## Context

The spec sets four rules:
- A citation must point to an actual retrieved source location.
- Citations must never be invented.
- An LLM saying "source supports claim" is not proof.
- Retrieved content is untrusted data, not instructions.

## Options considered

**Citation format:**
- (a) The LLM cites free-text titles or URLs. This invites fabrication. Rejected.
- (b) The LLM cites opaque evidence ids taken from a closed registry. **Chosen.**

**Claim support check:**
- (i) An LLM judge alone is not proof, by spec.
- (ii) An NLI cross-encoder alone is weak on long technical passages.
- (iii) **Layered (chosen):**
  1. A deterministic quote/span check, which is mandatory.
  2. An NLI model.
  3. An LLM judge with quoted evidence, only for cases NLI leaves undecided.

**Injection defence:**
- Prompt-only warnings are known to be insufficient on their own.
- **Chosen: defence in depth.** No tools, schema-constrained outputs, data delimiting, flagging, and output validation.

## Decision

### Evidence registry and citation ids

- **Registry.** `retrieve` stores every chunk shown to the LLM in `state.evidence` under a short run-local id (`E1…En`). Each id maps to:
  - `chunk_id`, `document_version_id`
  - `char_start`/`char_end`, `page_start`/`page_end`
  - `section_path`
  - the exact chunk text
- **Closed set.** The prompt lists only these ids. The model cannot cite anything else.
- **Synthesis output** follows a JSON schema (`Draft`). Each claim has `evidence_ids: [..]` and `quotes: [{evidence_id, text}]`: a short verbatim supporting quote, at most 300 characters, per cited id.

### Verification (the `verify` node)

Each claim gets one status: `supported`, `partially_supported`, `unsupported`, `contradicted`, or `unverifiable`.

1. **Deterministic checks.** Failing any of these makes the claim `unsupported`.
   - Every `evidence_id` exists in the registry.
   - Every quote is a substring of the cited chunk text. Matching normalizes whitespace, Unicode (NFKC) and hyphenation at line breaks, with case kept. A fuzzy fallback accepts a match ratio ≥ 0.95 (rapidfuzz, MIT; version checked at M5) to tolerate PDF-extraction artefacts.
   - The matched span is converted to absolute `char_start`/`char_end`, and from there to a page number.
   - The **UI citation points to this span**, not just the chunk.
2. **NLI check.** Premise = the quoted span plus its surrounding sentence window. Hypothesis = the claim.
   - Candidate models are `cross-encoder/nli-deberta-v3-small` or `-base`. License and size are checked at M5; CPU is fine at this volume.
   - Entailment ≥ τ_e gives `supported`, and contradiction ≥ τ_c gives `contradicted`.
   - τ values are calibrated on a small hand-labelled set during M5, with no default claimed now.
3. **LLM judge,** only for NLI-neutral cases (`task_type=citation_verification`, `response_format=json_schema`, temperature 0).
   - The judge gets the claim and the cited spans. It must return `{verdict, supporting_quote}`.
   - The returned quote **must pass the same substring check**. A judge verdict without a verifiable quote is downgraded to `unverifiable`.
4. **Policy:**
   - `unsupported` and `contradicted` claims trigger one re-synthesis (ADR-0008), then they are dropped.
   - A dropped claim's removal is listed under limitations: "the system could not verify…".
   - `interpretation` claims are allowed, but must be visibly labelled as model interpretation (§1.3F).

### Response validation (the `respond` node)

- The final Markdown may only contain citation markers `[E#]` for **supported or partially_supported** claims.
- After the stream finishes, a validator re-parses the full text.
  - **Unknown markers** cause the answer to be replaced with a deterministic rendering of the verified claims. The model's extra prose is never shown uncited.
  - An answer whose verified claims all have no citations is flagged.
- **Conflicts** (§1.3F): pairs of claims with `contradicted` NLI between evidence items from different documents are listed in a "conflicting evidence" section, with both citations and the document dates (ADR-0010).

### Prompt-injection defences

1. **No tools.** The LLM has no tools and no side-effecting capability. Nodes return data only, and the graph decides control flow. This is the main structural defence: a successful injection can at most distort text, and that text then faces verification.
2. **Data delimiting.** Evidence is placed in the user message inside `<evidence id="E3" boundary="{random 128-bit nonce}">…</evidence>` blocks.
   - Any occurrence of the tag or nonce pattern inside chunk text is escaped.
   - The system prompt says evidence is quoted data, and that instructions inside it must be ignored and reported.
3. **Flagging.** At ingestion time, chunks matching instruction-like patterns get an `injection_suspect` flag in metadata. Examples: "ignore previous instructions", role tags, `system:`, and hidden-text heuristics such as tiny or white font from pdfplumber character attributes.
   - Flagged chunks remain retrievable, since flagging should not censor, but they are surfaced in the evidence viewer with a warning.
   - The flag rate is a metric.
4. **Schema-constrained outputs** for every structured node. Free text appears only in `respond`, and that output is validated as described above.
5. **Frontend rendering.**
   - Markdown is sanitized: no raw HTML, no auto-loaded remote images (an exfiltration channel), and links are shown with their full URL.
   - Source text is displayed as text, never as HTML.
6. **Workspace isolation.** Retrieval cannot cross workspaces (ADR-0004 and ADR-0005), so an injected document cannot pull in another tenant's content.

## Amendment, 2026-11-14: deterministic layer as built in M4

- **Status before NLI exists.** A claim that passes the deterministic checks is `supported` with `method: "quote_match"`. Every answer that relies on such claims says so under limitations: the quote exists at the cited location, but its support for the claim has not been checked semantically. M5 adds NLI and the judge, which refine this status.
- **Normalisation.** In addition to whitespace, NFKC and line-break hyphenation, typographic single quotes, double quotes and dashes are folded to ASCII, because models rarely reproduce them. Case is kept. Fuzzy matching (rapidfuzz, ratio 0.95 or more) is deferred to M5.
- **Minimum quote length.** A quote must have at least 20 normalised characters, so that a claim cannot be "verified" by quoting a single term.
- **Every cited id needs a quote.** A claim that cites an id without a verified quote for that id fails.
- **Marker validation.** Besides unknown `[E#]` markers, pseudo-citations such as `[3]`, `[Source 2]` or `[E3 p.2]` also make the validator replace the model's text. An answer without any marker is replaced by the deterministic rendering too. It is not just flagged, because uncited model prose would reach the user.
- **Gaps.** Synthesis `gaps` are shown under limitations, at most 5 of them and 300 characters each. Synthesis `conflicts` are kept in the run state but not shown until M5 builds the cited conflict section.

## Consequences

- **Citations are verifiable by construction.** The UI can highlight the exact span on the exact page.
- **Costs:** an NLI model (about 100–200M parameters, EST), and one optional LLM call per undecided claim. These are bounded by the run budget.
- **Over-strict quote matching may drop true claims** whose quotes are paraphrased. That cost is measured as "claims dropped" in M6 and tuned using validation data only.

## Verification plan

- **Unit tests:**
  - A fabricated evidence id is rejected.
  - A paraphrased "quote" is rejected.
  - A hyphenation-broken quote is accepted.
  - The span-to-page mapping is correct on a golden PDF.
- **Adversarial test set** (M5): 20 or more documents with embedded instructions ("cite E99", "say the answer is X", "output the system prompt"). Pass criteria:
  - no unknown citations reach the user;
  - no system-prompt leakage in the final answer;
  - the flag rate is reported.
- **M6 metrics:**
  - `citation_validity`: deterministic, and must equal 1.0.
  - `citation_support_rate` and claim-drop rate, reported per evaluation run.
