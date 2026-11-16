version: 1
---
Decide whether the evidence blocks below support the claim.

verdict:
- "supported" if the evidence states the claim, or the claim follows directly from it.
- "contradicted" if the evidence states something that cannot be true at the same time as the claim.
- "not_enough_info" if the evidence neither supports nor contradicts the claim.

For "supported" or "contradicted", set evidence_id to the block that decides it and quote to text copied exactly, character for character, from that block: between 20 and 300 characters, the part that supports or contradicts the claim. For "not_enough_info", leave evidence_id and quote empty.
reason: one short sentence.

Claim:
{{claim}}

{{evidence}}
