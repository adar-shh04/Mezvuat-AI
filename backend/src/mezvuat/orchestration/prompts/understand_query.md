version: 1
---
Analyse the user's research question. Do not answer it.

- objective: what the user wants to know, in one sentence.
- subject: the main topic.
- constraints: explicit constraints (time, method, dataset, document), or an empty list.
- expected_answer_type: one of factoid, explanation, comparison, list, other.
- temporal_scope: as_of is an ISO date (YYYY-MM-DD) only if the question names a point in time, otherwise null. wants_latest is true if the user asks for the latest or current version. compare_versions is true if the user asks what changed between versions.
- needs_multi_doc: true if the answer probably needs more than one document.
- single_hop: true if a single search for the question itself is likely enough.

Question:
{{question}}
