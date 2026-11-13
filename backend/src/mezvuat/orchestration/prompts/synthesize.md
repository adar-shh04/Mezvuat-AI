version: 1
---
Answer the question using only the evidence blocks below.

Write the answer as a list of short claims. For each claim:
- id: C1, C2, ...
- text: one sentence.
- evidence_ids: the ids of the evidence blocks that support it.
- quotes: for every id in evidence_ids, one quote copied exactly, character for character, from that evidence block. A quote is between 20 and 300 characters. Do not paraphrase, shorten or fix the quote.
- kind: "fact" if the evidence states it, "interpretation" if it is your inference from the evidence.

conflicts: statements where evidence blocks from different documents disagree, or an empty list.
gaps: parts of the question the evidence does not answer, or an empty list.
If the evidence does not answer the question, return no claims and explain in gaps.
{{feedback}}
Question:
{{question}}

{{evidence}}
