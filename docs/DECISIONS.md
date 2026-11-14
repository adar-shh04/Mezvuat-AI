# Product Owner Decisions

Recorded 2026-11-08 at the end of Phase 1. These override any ADR or plan text that says otherwise.
Where an ADR conflicts with this file, update the ADR. Do not reinterpret the decision.

| # | Topic | Decision |
|---|---|---|
| D1 | Data policy for calls carrying document text | Public cloud for public corpus, local for private (options E + C). |
| D2 | Human labelling | LLM pre-labels, the product owner confirms or corrects. Reduced to about 6–8 hours of review. |
| D3 | Langfuse | Langfuse Cloud free tier, with prompts and document text masked. Optional at runtime. |
| D4 | Direct provider | Groq, free tier only. |
| D5 | Demo corpus | arXiv cs.IR / cs.CL papers under CC licences, with multiple versions. |
| D6 | AGPL dependencies | Allowed in Mezvuat. |
| D7 | Repository licence | AGPL-3.0. |
| D8 | Hosting and CI | GitHub with GitHub Actions. The repository is public at https://github.com/adar-shh04/Mezvuat-AI (updated 2026-11-14 at the product owner's request). Work is committed locally; the coordinator pushes `phase-2` after verifying each batch. |
| D9 | Cloud demo | Not in this phase. Local Docker Compose only. |
| D10 | Installs and downloads | Approved: uv, Python 3.13, CUDA PyTorch, models and datasets. Cache on X:. |
| D11 | Temporal question set (M5) | Drafted by an LLM (Groq, free tier) from the public demo corpus; the product owner confirms, corrects or rejects every item. Recorded 2026-11-14. |

## What each decision requires

**D1 — Data policy**
- A workspace is private by default. It counts as public only when every document in it comes from the allowlisted CC-licensed demo corpus, and the system verifies this rather than trusting a user flag. Adding any other document makes the workspace private.
- Public workspaces send `routing.data_policy: "any"`. Private workspaces send `"local_only"`.
- Direct fallback follows the same rule: Groq for public workspaces only. Private workspaces need a direct Ollama backend so that Mezvuat stays usable without ModelMesh; the weaker models this implies are accepted.
- If no permitted backend is available, the call fails with a clear error. It must never widen the policy.
- Offline evaluation and judge calls obey D1 too: RAGAS judging and LLM pre-labelling run only on the public corpus.

**D2 — Labelling**
- Size the validation and test sets to fit about 6–8 hours of review, and state the resulting sample size and its statistical power in the reports.
- Pre-labels can anchor the reviewer. Measure this: hide the pre-label on a random subset, report the agreement, and document the limitation.
- Split by document family before any query generation, as in ADR-0007.
- **Amendment, 2026-11-14 (product owner): hide the pre-label for every remaining relevance item.** The anchoring check showed agreement with the LLM of 98.8% on 160 test items with the pre-label shown, against 29.2% on 24 with it hidden. In most hidden disagreements the owner graded the passage more relevant than the LLM did. From now on `review val|test` hides every pre-label (`--show-prelabels` restores the old policy). Labels made with the pre-label shown are kept but reported separately as lower-trust. The randomised comparison uses only the original random fifth, which every record marks (`in_random_subset`, `blind_policy`). The temporal review keeps its 20% hidden subset.

**D3 — Langfuse**
- The app must run with Langfuse unset. Masking is applied before data leaves the process. Add a test proving that no prompt or document text reaches the exporter.

**D4 — Groq**
- If free limits throttle evaluation, slow down and report it. Never switch to a paid tier.

**D6 and D7 — AGPL**
- Allowed does not mean preferred. Re-open ADR-0003 and ADR-0004 and decide with evidence: compare PyMuPDF and pdfplumber on extraction quality for the demo corpus, and judge ParadeDB against bm25s on retrieval quality and operational cost. Keep the simpler option unless the difference is measurable.
- Add an AGPL-3.0 `LICENSE`. Check that every dependency's licence is compatible.

**D11 — Temporal question set**
- An LLM drafts the questions and gold answers from the public demo corpus only (D1). Each gold answer cites exact source spans: paper id, version and a verbatim quote that the tooling has checked against that version's text.
- The product owner confirms, corrects or rejects every item. An item counts as gold only after review.
- The D2 anchoring check applies: the draft answer is hidden on a random 20% of the items, and agreement is reported.
- Versions and dates come only from arXiv metadata. If the corpus lacks a kind of case, the report says so; no version or date is invented.

**Limits on ModelMesh keys**
- ModelMesh's default for new keys is 20 RPM, 50K TPM and $0.25/day. A research run can make up to 12 LLM calls with large evidence payloads, so it may approach the token limit. Handle 429 as the contract specifies. If defaults block normal use, report it; ModelMesh keys can be given explicit limits.
