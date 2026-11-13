"""The eight research nodes and their routing (ADR-0008).

Nodes return partial state updates. The LLM has no tools: nodes return data and the routing functions decide the
control flow. When a bound is hit the graph moves forward, towards an answer that states the gap; it never loops
further and never fabricates. Fatal errors (retrieval or inference unavailable, data policy) go to `fail`.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from langgraph.config import get_stream_writer
from pydantic import BaseModel

from mezvuat.citations import EvidenceItem, PageSpan, QuoteInput, check_claim, chunk_index, scan_markers
from mezvuat.core.errors import MezvuatError
from mezvuat.inference import (
    FATAL_INFERENCE_CODES,
    ChatMessage,
    Correlation,
    InferenceError,
    LLMAttempt,
    LLMRequest,
    LLMResult,
    StreamDelta,
    StreamDone,
    StreamIncompleteError,
    TaskType,
)
from mezvuat.orchestration.deps import WorkflowDeps
from mezvuat.orchestration.models import (
    MAX_NEW_QUERIES,
    MAX_SUBQUESTIONS,
    Citation,
    ClaimVerdict,
    Coverage,
    Draft,
    FinalAnswer,
    QueryAnalysis,
    Reformulation,
    ResearchPlan,
    SubQuestion,
    Sufficiency,
)
from mezvuat.orchestration.prompts import Prompt, evidence_section, load_prompt, new_boundary
from mezvuat.orchestration.state import (
    MAX_EVIDENCE_FOR_SYNTHESIS,
    MAX_ITER,
    MAX_RESYNTH,
    RESERVED_CALLS,
    Budget,
    LLMCallRecord,
    PendingQuery,
    ResearchState,
    RetrievalCall,
    RunError,
)
from mezvuat.retrieval.types import RankedEvidence

log = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)
Update = dict[str, Any]

QUOTE_ONLY_LIMITATION = (
    "Citations were checked for verbatim quotes at the cited location only. Whether each quote fully supports its "
    "claim has not been checked by a semantic model yet."
)
ABSTAIN_TEXT = "I could not find evidence in this workspace that answers the question, so I am not giving an answer."


@dataclass
class _Call:
    """Outcome of one LLM step: parsed output or a reason it is missing, plus the state bookkeeping."""

    parsed: BaseModel | None = None
    result: LLMResult | None = None
    skipped: str | None = None
    error: InferenceError | None = None
    updates: Update = field(default_factory=dict)

    @property
    def fatal(self) -> bool:
        return self.error is not None and self.error.code in FATAL_INFERENCE_CODES

    def failure(self, node: str) -> Update:
        error = self.error or InferenceError("inference failed")
        return _fail(node, error.code, error.message)


def _fail(node: str, code: str, message: str) -> Update:
    return {"status": "failed", "error": RunError(code=code, message=message, node=node)}


def _json(model: BaseModel | None) -> str:
    return "null" if model is None else json.dumps(model.model_dump(mode="json"), ensure_ascii=False)


class ResearchNodes:
    def __init__(self, deps: WorkflowDeps) -> None:
        self.deps = deps
        self.system = load_prompt("system")

    # ----- budget -------------------------------------------------------------------------------------------------

    def remaining_s(self, state: ResearchState) -> float:
        return state.budget.deadline - self.deps.clock()

    def can_call(self, state: ResearchState, calls: int = 1) -> bool:
        budget = state.budget
        return budget.llm_calls_left >= calls and budget.tokens_left > 0 and self.remaining_s(state) > 1.0

    def _records(
        self, state: ResearchState, node: str, task_type: str, prompt: Prompt, attempts: Sequence[LLMAttempt], mode: str
    ) -> list[LLMCallRecord]:
        return [
            LLMCallRecord(
                client_request_id=a.client_request_id,
                run_id=state.run_id,
                workflow_node=node,
                task_type=task_type,
                backend="direct",
                provider=a.provider,
                model=a.model,
                prompt_version=prompt.version_id,
                iteration=state.iteration,
                status=a.status,
                error_code=a.error_code,
                http_status=a.http_status,
                provider_request_id=a.provider_request_id,
                prompt_tokens=a.usage.prompt_tokens,
                completion_tokens=a.usage.completion_tokens,
                usage_source=a.usage.source,
                latency_ms=a.latency_ms,
                finish_reason=a.finish_reason,
                structured_mode=mode,
            )
            for a in attempts
        ]

    def _account(
        self,
        state: ResearchState,
        node: str,
        task_type: str,
        prompt: Prompt,
        attempts: Sequence[LLMAttempt],
        mode: str,
    ) -> Update:
        records = self._records(state, node, task_type, prompt, attempts, mode)
        if records and self.deps.recorder is not None:
            try:
                self.deps.recorder.record(records)
            except Exception:  # Recording must never break a user's run; the state still carries the rows.
                log.exception("llm_call recording failed", extra={"run_id": state.run_id})
        tokens = sum((r.prompt_tokens or 0) + (r.completion_tokens or 0) for r in records)
        budget = state.budget
        return {
            "llm_calls": records,
            "budget": Budget(
                llm_calls_left=budget.llm_calls_left - max(len(records), 1),
                tokens_left=budget.tokens_left - tokens,
                deadline=budget.deadline,
            ),
        }

    def _request(
        self,
        state: ResearchState,
        node: str,
        task_type: TaskType,
        prompt: Prompt,
        user: str,
        *,
        schema: type[BaseModel] | None,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> LLMRequest:
        return LLMRequest(
            task_type=task_type,
            messages=[ChatMessage("system", self.system.template), ChatMessage("user", user)],
            correlation=Correlation(state.run_id, node, prompt.version_id, state.iteration),
            data_policy=state.data_policy,
            max_tokens=max_tokens,
            response_schema=schema,
            temperature=temperature,
            seed=0,
            timeout_s=max(min(self.deps.max_call_timeout_s, self.remaining_s(state)), 1.0),
            repair_attempts=1 if state.budget.llm_calls_left >= 2 else 0,
        )

    def structured(
        self,
        state: ResearchState,
        node: str,
        task_type: TaskType,
        prompt_name: str,
        values: dict[str, str],
        schema: type[M],
        max_tokens: int,
    ) -> _Call:
        prompt = load_prompt(prompt_name)
        if not self.can_call(state):
            return _Call(skipped="the run's LLM call, token or time budget was exhausted")
        request = self._request(
            state, node, task_type, prompt, prompt.render(values), schema=schema, max_tokens=max_tokens
        )
        try:
            result = self.deps.llm.complete(request)
        except InferenceError as exc:
            log.warning("llm step failed", extra={"node": node, "code": exc.code, "run_id": state.run_id})
            updates = self._account(state, node, task_type, prompt, exc.attempts, "json_schema_strict")
            return _Call(error=exc, updates=updates)
        updates = self._account(state, node, task_type, prompt, result.attempts, result.structured_mode)
        if not isinstance(result.parsed, schema):
            return _Call(error=InferenceError("backend returned no parsed output"), updates=updates)
        return _Call(parsed=result.parsed, result=result, updates=updates)

    @staticmethod
    def _degraded(call: _Call, what: str) -> list[str]:
        reason = call.skipped or (call.error.code if call.error else "unknown")
        return [f"{what} was skipped ({reason})."]

    # ----- 1. understand_query ------------------------------------------------------------------------------------

    def understand_query(self, state: ResearchState) -> Update:
        call = self.structured(
            state,
            "understand_query",
            "query_classification",
            "understand_query",
            {"question": state.question},
            QueryAnalysis,
            700,
        )
        if call.fatal:
            return {**call.updates, **call.failure("understand_query")}
        updates = {**call.updates, "analysis": call.parsed}
        if call.parsed is None:
            updates["limitations"] = self._degraded(call, "Query analysis")
        return updates

    # ----- 2. plan_research ---------------------------------------------------------------------------------------

    def plan_research(self, state: ResearchState) -> Update:
        single = ResearchPlan(
            subquestions=[SubQuestion(id="S1", question=state.question)],
            success_criteria="Evidence that directly answers the question.",
        )
        analysis = state.analysis
        if analysis is None or analysis.single_hop:
            # The documented shortcut: still a node, so it still appears in traces.
            return {"plan": single, "pending_queries": [PendingQuery(subquestion_id="S1", query=state.question)]}
        call = self.structured(
            state,
            "plan_research",
            "research_planning",
            "plan_research",
            {"question": state.question, "analysis": _json(analysis), "max_subquestions": str(MAX_SUBQUESTIONS)},
            ResearchPlan,
            900,
        )
        if call.fatal:
            return {**call.updates, **call.failure("plan_research")}
        plan = call.parsed if isinstance(call.parsed, ResearchPlan) else single
        updates: Update = {
            **call.updates,
            "plan": plan,
            "pending_queries": [PendingQuery(subquestion_id=s.id, query=s.question) for s in plan.subquestions],
        }
        if call.parsed is None:
            updates["limitations"] = self._degraded(call, "Research planning")
        return updates

    # ----- 3. retrieve (no LLM) -----------------------------------------------------------------------------------

    def retrieve(self, state: ResearchState) -> Update:
        index = chunk_index(state.evidence)
        count = len(state.evidence)
        new: dict[str, EvidenceItem] = {}
        log_rows: list[RetrievalCall] = []
        for pending in state.pending_queries:
            started = time.perf_counter()
            try:
                results = self.deps.search.search(
                    state.workspace_id, pending.query, self.deps.per_query_k, state.filters
                )
            except MezvuatError as exc:
                return _fail("retrieve", "retrieval_unavailable", f"{exc.code}: {exc.message}")
            fresh = 0
            for ranked in results:
                if ranked.chunk.workspace_id != state.workspace_id:
                    # Defence in depth: retrieval already filters by workspace.
                    continue
                if ranked.chunk.chunk_id in index:
                    continue
                count += 1
                evidence_id = f"E{count}"
                index[ranked.chunk.chunk_id] = evidence_id
                new[evidence_id] = self._item(evidence_id, ranked, state.iteration, pending)
                fresh += 1
            log_rows.append(
                RetrievalCall(
                    round=state.iteration,
                    subquestion_id=pending.subquestion_id,
                    query=pending.query,
                    results=len(results),
                    new_evidence=fresh,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
        if new:
            new = self._attach_pages(state.workspace_id, new)
        return {"evidence": new, "retrieval_log": log_rows, "pending_queries": []}

    @staticmethod
    def _item(evidence_id: str, ranked: RankedEvidence, round_: int, pending: PendingQuery) -> EvidenceItem:
        chunk = ranked.chunk
        return EvidenceItem(
            evidence_id=evidence_id,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_version_id=chunk.document_version_id,
            document_title=chunk.document_title,
            version_label=chunk.version_label,
            is_latest_version=chunk.is_latest_version,
            text=chunk.text,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_path=list(chunk.section_path),
            section_kind=chunk.section_kind,
            page_spans=[],
            retrieval_round=round_,
            subquestion_id=pending.subquestion_id,
            query=pending.query,
            rank=ranked.rank,
            score=ranked.score,
        )

    def _attach_pages(self, workspace_id: str, items: dict[str, EvidenceItem]) -> dict[str, EvidenceItem]:
        version_ids = sorted({item.document_version_id for item in items.values()})
        try:
            spans = self.deps.pages.page_spans(workspace_id, version_ids)
        except MezvuatError:
            log.warning("page map unavailable; citations keep chunk-level pages")
            return items
        attached: dict[str, EvidenceItem] = {}
        for evidence_id, item in items.items():
            overlapping = [
                PageSpan(page=s.page, start=s.start, end=s.end)
                for s in spans.get(item.document_version_id, [])
                if s.start < item.char_end and s.end > item.char_start
            ]
            attached[evidence_id] = item.model_copy(update={"page_spans": overlapping})
        return attached

    # ----- 4. assess_evidence -------------------------------------------------------------------------------------

    def _subquestions(self, state: ResearchState) -> list[SubQuestion]:
        return state.plan.subquestions if state.plan else [SubQuestion(id="S1", question=state.question)]

    def assess_evidence(self, state: ResearchState) -> Update:
        subquestions = self._subquestions(state)
        if not state.evidence:
            missing = [
                Coverage(subquestion_id=s.id, covered=False, missing="no evidence retrieved") for s in subquestions
            ]
            return {"sufficiency": Sufficiency(subquestions=missing, rationale="Retrieval returned no evidence.")}
        boundary = new_boundary()
        shown: list[EvidenceItem] = []
        for sub in subquestions:
            items = sorted(
                (i for i in state.evidence.values() if i.subquestion_id == sub.id),
                key=lambda i: (i.retrieval_round, i.rank),
            )
            shown.extend(items[: self.deps.assess_items_per_subquestion])
        call = self.structured(
            state,
            "assess_evidence",
            "evidence_assessment",
            "assess_evidence",
            {
                "subquestions": "\n".join(f"{s.id}: {s.question}" for s in subquestions),
                "evidence": evidence_section(shown, boundary, max_chars=self.deps.assess_item_chars),
            },
            Sufficiency,
            700,
        )
        if call.fatal:
            return {**call.updates, **call.failure("assess_evidence")}
        if not isinstance(call.parsed, Sufficiency):
            return {**call.updates, "sufficiency": None, "limitations": self._degraded(call, "Evidence assessment")}
        by_id = {c.subquestion_id: c for c in call.parsed.subquestions}
        normalized = [
            by_id.get(s.id, Coverage(subquestion_id=s.id, covered=False, missing="not assessed")) for s in subquestions
        ]
        return {
            **call.updates,
            "sufficiency": Sufficiency(subquestions=normalized, rationale=call.parsed.rationale),
        }

    def loop_allowed(self, state: ResearchState) -> bool:
        return (
            state.iteration < MAX_ITER
            and self.can_call(state, calls=RESERVED_CALLS + 2)
            and self.remaining_s(state) > self.deps.min_loop_time_s
        )

    def route_after_assess(self, state: ResearchState) -> str:
        if state.status == "failed":
            return "fail"
        sufficiency = state.sufficiency
        if sufficiency is None or all(c.covered for c in sufficiency.subquestions):
            return "synthesize"
        return "reformulate" if self.loop_allowed(state) else "synthesize"

    # ----- 5. reformulate -----------------------------------------------------------------------------------------

    def reformulate(self, state: ResearchState) -> Update:
        sufficiency = state.sufficiency
        uncovered = [c for c in (sufficiency.subquestions if sufficiency else []) if not c.covered]
        questions = {s.id: s.question for s in self._subquestions(state)}
        tried = {row.query.strip().lower() for row in state.retrieval_log}
        call = self.structured(
            state,
            "reformulate",
            "query_rewriting",
            "reformulate",
            {
                "uncovered": "\n".join(
                    f"{c.subquestion_id}: {questions.get(c.subquestion_id, '')} (missing: {c.missing})"
                    for c in uncovered
                ),
                "tried": "\n".join(f"- {row.query}" for row in state.retrieval_log),
                "max_new_queries": str(MAX_NEW_QUERIES),
            },
            Reformulation,
            500,
        )
        updates: Update = {**call.updates, "iteration": state.iteration + 1}
        if call.fatal:
            return {**updates, **call.failure("reformulate")}
        pending: list[PendingQuery] = []
        if isinstance(call.parsed, Reformulation):
            allowed = {c.subquestion_id for c in uncovered}
            for rewrite in call.parsed.rewrites:
                if rewrite.subquestion_id not in allowed:
                    continue
                for query in rewrite.new_queries:
                    if query.lower() not in tried:
                        tried.add(query.lower())
                        pending.append(PendingQuery(subquestion_id=rewrite.subquestion_id, query=query))
        else:
            updates["limitations"] = self._degraded(call, "Additional research")
        updates["pending_queries"] = pending
        return updates

    @staticmethod
    def route_after_reformulate(state: ResearchState) -> str:
        if state.status == "failed":
            return "fail"
        return "retrieve" if state.pending_queries else "synthesize"

    # ----- 6. synthesize ------------------------------------------------------------------------------------------

    def select_for_synthesis(self, state: ResearchState) -> list[EvidenceItem]:
        """Round-robin over subquestions by rank, bounded by count and characters."""
        queues: dict[str, list[EvidenceItem]] = {}
        for item in sorted(state.evidence.values(), key=lambda i: (i.retrieval_round, i.rank, int(i.evidence_id[1:]))):
            queues.setdefault(item.subquestion_id, []).append(item)
        chosen: list[EvidenceItem] = []
        chars = 0
        while len(chosen) < MAX_EVIDENCE_FOR_SYNTHESIS and any(queues.values()):
            for queue in queues.values():
                if not queue or len(chosen) >= MAX_EVIDENCE_FOR_SYNTHESIS:
                    continue
                item = queue.pop(0)
                if chosen and chars + len(item.text) > self.deps.synthesis_evidence_chars:
                    continue
                chosen.append(item)
                chars += len(item.text)
        return sorted(chosen, key=lambda i: int(i.evidence_id[1:]))

    def synthesize(self, state: ResearchState) -> Update:
        resynthesis = state.draft is not None
        updates: Update = {"resynth_count": state.resynth_count + 1} if resynthesis else {}
        if not resynthesis and state.sufficiency is not None:
            gaps = [c for c in state.sufficiency.subquestions if not c.covered]
            questions = {s.id: s.question for s in self._subquestions(state)}
            updates["limitations"] = [
                f"Evidence may be incomplete for: {questions.get(c.subquestion_id, c.subquestion_id)}" for c in gaps
            ]
        chosen = self.select_for_synthesis(state)
        if not chosen:
            empty = Draft(claims=[], conflicts=[], gaps=["No evidence was retrieved for this question."])
            return {**updates, "draft": empty, "shown_evidence_ids": []}
        feedback = ""
        if resynthesis and state.verification:
            problems = [f"- {v.claim_id}: {'; '.join(v.problems)}" for v in state.verification if v.problems]
            feedback = (
                "\nYour previous draft had claims that failed verification. Fix them by quoting exactly, or drop "
                "them:\n" + "\n".join(problems) + "\n"
            )
        call = self.structured(
            state,
            "synthesize",
            "evidence_synthesis",
            "synthesize",
            {"question": state.question, "evidence": evidence_section(chosen, new_boundary()), "feedback": feedback},
            Draft,
            3000,
        )
        updates = {**updates, **call.updates}
        if call.fatal:
            return {**updates, **call.failure("synthesize")}
        if isinstance(call.parsed, Draft):
            return {**updates, "draft": call.parsed, "shown_evidence_ids": [i.evidence_id for i in chosen]}
        limitation = self._degraded(call, "Re-synthesis" if resynthesis else "Synthesis")
        if resynthesis:
            # Keep the previous draft; verification will drop what still fails.
            return {**updates, "limitations": [*updates.get("limitations", []), *limitation]}
        empty = Draft(claims=[], conflicts=[], gaps=["The answer could not be synthesised."])
        return {
            **updates,
            "draft": empty,
            "shown_evidence_ids": [],
            "limitations": [*updates.get("limitations", []), *limitation],
        }

    # ----- 7. verify ----------------------------------------------------------------------------------------------

    def verify(self, state: ResearchState) -> Update:
        draft = state.draft or Draft(claims=[], conflicts=[], gaps=[])
        registry = {k: v for k, v in state.evidence.items() if k in set(state.shown_evidence_ids)}
        verdicts: list[ClaimVerdict] = []
        seen: set[str] = set()
        for position, claim in enumerate(draft.claims, start=1):
            claim_id = claim.id if claim.id and claim.id not in seen else f"C{position}"
            seen.add(claim_id)
            check = check_claim(
                claim.evidence_ids,
                [QuoteInput(evidence_id=q.evidence_id, text=q.text) for q in claim.quotes],
                registry,
            )
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim_id,
                    text=claim.text,
                    kind=claim.kind,
                    status="supported" if check.passed else "unsupported",
                    method="quote_match" if check.passed else "none",
                    evidence_ids=list(dict.fromkeys(claim.evidence_ids)),
                    spans=[s for s in check.spans if s.status == "ok"] if check.passed else list(check.spans),
                    problems=check.problems,
                )
            )
        return {"verification": verdicts}

    def route_after_verify(self, state: ResearchState) -> str:
        if state.status == "failed":
            return "fail"
        failing = any(v.status in ("unsupported", "contradicted") for v in state.verification or [])
        if failing and state.resynth_count < MAX_RESYNTH and self.can_call(state, calls=RESERVED_CALLS):
            return "synthesize"
        return "respond"

    # ----- 8. respond ---------------------------------------------------------------------------------------------

    def respond(self, state: ResearchState) -> Update:
        writer = get_stream_writer()
        verdicts = state.verification or []
        verified = [v for v in verdicts if v.status in ("supported", "partially_supported")]
        dropped = [v for v in verdicts if v not in verified]
        limitations = list(dict.fromkeys(state.limitations))
        limitations += [f"The system could not verify this claim, so it was left out: {v.text}" for v in dropped]
        if any(v.method == "quote_match" for v in verified):
            limitations.append(QUOTE_ONLY_LIMITATION)
        if state.draft:
            limitations += [f"Not answered by the evidence: {gap[:300]}" for gap in state.draft.gaps[:5]]
        updates: Update = {}
        flags: list[str] = []
        allowed = {e for v in verified for e in v.evidence_ids}
        markdown: str | None = None
        if not verified:
            flags.append("abstained")
        elif not self.can_call(state):
            flags.append("budget_exhausted_before_response")
            limitations.append("The answer is a plain list of the verified claims; the response budget was used up.")
        else:
            markdown, call_updates, stream_flags = self._stream_answer(state, verified, writer)
            updates.update(call_updates)
            flags += stream_flags
            if markdown is not None:
                report = scan_markers(markdown, allowed)
                if not report.valid:
                    flags.append("unknown_citation_markers")
                    markdown = None
                elif not report.used:
                    flags.append("answer_without_citations")
                    markdown = None
                if markdown is None:
                    writer({"event": "answer_replaced", "reason": flags[-1]})
        rendering = "model" if markdown is not None else "deterministic"
        body = markdown if markdown is not None else self._deterministic(verified)
        used = set(scan_markers(body, allowed).used) if verified else set()
        citations = [
            Citation(
                evidence_id=span.evidence_id,
                claim_id=v.claim_id,
                chunk_id=state.evidence[span.evidence_id].chunk_id,
                document_id=state.evidence[span.evidence_id].document_id,
                document_version_id=state.evidence[span.evidence_id].document_version_id,
                document_title=state.evidence[span.evidence_id].document_title,
                version_label=state.evidence[span.evidence_id].version_label,
                page=span.page,
                char_start=span.char_start,
                char_end=span.char_end,
                quote=span.quote,
            )
            for v in verified
            for span in v.spans
            if span.status == "ok"
            and span.evidence_id in used
            and span.char_start is not None
            and span.char_end is not None
        ]
        full = body
        if limitations:
            full += "\n\n### Limitations\n\n" + "\n".join(f"- {line}" for line in limitations)
        answer = FinalAnswer(
            markdown=full,
            rendering=rendering,
            citations=citations,
            claims=verdicts,
            limitations=limitations,
            flags=flags,
        )
        writer({"event": "answer_final", "rendering": rendering})
        return {**updates, "answer": answer, "status": "completed"}

    @staticmethod
    def _deterministic(verified: Iterable[ClaimVerdict]) -> str:
        lines = []
        for v in verified:
            markers = "".join(f"[{e}]" for e in v.evidence_ids)
            label = "*Interpretation:* " if v.kind == "interpretation" else ""
            lines.append(f"- {label}{v.text} {markers}")
        if not lines:
            return ABSTAIN_TEXT
        return "Based on the retrieved evidence:\n\n" + "\n".join(lines)

    def _stream_answer(
        self, state: ResearchState, verified: list[ClaimVerdict], writer: Any
    ) -> tuple[str | None, Update, list[str]]:
        prompt = load_prompt("respond")
        claims = "\n".join(f"- ({v.kind}) {v.text} {''.join(f'[{e}]' for e in v.evidence_ids)}" for v in verified)
        user = prompt.render({"question": state.question, "claims": claims})
        current = state
        updates: Update = {}
        flags: list[str] = []
        for attempt in range(2):
            if attempt and not self.can_call(current):
                break
            request = self._request(
                current, "respond", "final_response", prompt, user, schema=None, max_tokens=1500, temperature=0.2
            )
            parts: list[str] = []
            try:
                done: LLMResult | None = None
                for event in self.deps.llm.stream(request):
                    if isinstance(event, StreamDelta):
                        parts.append(event.text)
                        writer({"event": "answer_delta", "text": event.text})
                    elif isinstance(event, StreamDone):
                        done = event.result
                if done is None:
                    raise StreamIncompleteError("stream ended without a final event")
            except StreamIncompleteError as exc:
                writer({"event": "answer_reset", "reason": "response interrupted, retrying"})
                step = self._account(current, "respond", "final_response", prompt, exc.attempts, "none")
                updates = self._merge(updates, step)
                current = current.model_copy(update={"budget": step["budget"]})
                flags.append("stream_interrupted")
                continue
            except InferenceError as exc:
                step = self._account(current, "respond", "final_response", prompt, exc.attempts, "none")
                updates = self._merge(updates, step)
                flags.append(f"response_failed:{exc.code}")
                writer({"event": "answer_reset", "reason": exc.code})
                return None, updates, flags
            step = self._account(current, "respond", "final_response", prompt, done.attempts, "none")
            updates = self._merge(updates, step)
            if done.finish_reason == "length":
                flags.append("truncated")
            elif done.finish_reason == "content_filter":
                flags.append("provider_filtered")
                return None, updates, flags
            return "".join(parts).strip() or None, updates, flags
        return None, updates, flags

    @staticmethod
    def _merge(left: Update, right: Update) -> Update:
        return {"llm_calls": [*left.get("llm_calls", []), *right["llm_calls"]], "budget": right["budget"]}

    # ----- fail ---------------------------------------------------------------------------------------------------

    @staticmethod
    def fail(state: ResearchState) -> Update:
        return {"status": "failed", "answer": None}

    @staticmethod
    def or_fail(next_node: str) -> Any:
        def route(state: ResearchState) -> str:
            return "fail" if state.status == "failed" else next_node

        route.__name__ = f"route_to_{next_node}"
        return route
