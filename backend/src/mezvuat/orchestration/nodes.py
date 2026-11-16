"""The eight research nodes and their routing (ADR-0008).

Nodes return partial state updates. The LLM has no tools: nodes return data and the routing functions decide the
control flow. When a bound is hit the graph moves forward, towards an answer that states the gap; it never loops
further and never fabricates. Fatal errors (retrieval or inference unavailable, data policy) go to `fail`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, TypeVar

from langgraph.config import get_stream_writer
from pydantic import BaseModel

from mezvuat.citations import (
    EvidenceItem,
    NliVerdict,
    PageSpan,
    PromptLeakDetector,
    QuoteInput,
    SpanRef,
    check_claim,
    check_quote,
    chunk_index,
    injection_signals,
    scan_markers,
    verify_claims,
)
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
    evidence_key,
)
from mezvuat.orchestration.deps import WorkflowDeps
from mezvuat.orchestration.models import (
    MAX_NEW_QUERIES,
    MAX_SUBQUESTIONS,
    Citation,
    ClaimVerdict,
    Conflict,
    Coverage,
    Draft,
    FinalAnswer,
    JudgeCheck,
    JudgeOutput,
    QueryAnalysis,
    Reformulation,
    ResearchPlan,
    SubQuestion,
    Sufficiency,
    SupportMethod,
    VerificationInfo,
)
from mezvuat.orchestration.prompts import PROMPT_NAMES, Prompt, evidence_section, load_prompt, new_boundary
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
from mezvuat.temporal import DateFact, VersionDates, describe_dates, resolve_as_of

log = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)
Update = dict[str, Any]

QUOTE_ONLY_LIMITATION = (
    "Citations were checked for verbatim quotes at the cited location only. Whether each quote fully supports its "
    "claim has not been checked by a semantic model yet."
)
NLI_UNAVAILABLE_LIMITATION = "The NLI verifier was unavailable for this run, so claims kept the quote-match check only."
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


_ANY_BRACKET_CITATION = re.compile(r"\[[^\]\n]{0,40}\]")


def _plain(text: str) -> str:
    """Model text shown outside the verified claims carries no citation-like markers."""
    return " ".join(_ANY_BRACKET_CITATION.sub("", text).split())


def _json(model: BaseModel | None) -> str:
    return "null" if model is None else json.dumps(model.model_dump(mode="json"), ensure_ascii=False)


class ResearchNodes:
    def __init__(self, deps: WorkflowDeps) -> None:
        self.deps = deps
        self.system = load_prompt("system")
        self.leaks = PromptLeakDetector(load_prompt(name).template for name in PROMPT_NAMES)

    # ----- budget -------------------------------------------------------------------------------------------------

    def remaining_s(self, state: ResearchState) -> float:
        return state.budget.deadline - self.deps.clock()

    def can_call(self, state: ResearchState, calls: int = 1) -> bool:
        budget = state.budget
        return budget.llm_calls_left >= calls and budget.tokens_left > 0 and self.remaining_s(state) > 1.0

    def _records(
        self, state: ResearchState, node: str, task_type: str, prompt: Prompt, attempts: Sequence[LLMAttempt], mode: str
    ) -> list[LLMCallRecord]:
        trace_id = self.deps.langfuse_trace_id(state.run_id)
        return [
            LLMCallRecord(
                client_request_id=a.client_request_id,
                run_id=state.run_id,
                workflow_node=node,
                task_type=task_type,
                backend=a.backend,
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
                modelmesh_request_id=a.modelmesh_request_id,
                modelmesh_trace_id=a.modelmesh_trace_id,
                cache_status=a.cache_status,
                cost_estimated_usd=a.cost_estimated_usd,
                fallback_reason=a.fallback_reason,
                langfuse_trace_id=trace_id,
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
        evidence: Sequence[EvidenceItem] = (),
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
            workspace_id=state.workspace_id,
            evidence_keys=tuple(evidence_key(i.chunk_id, i.document_version_id, i.text) for i in evidence),
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
        evidence: Sequence[EvidenceItem] = (),
    ) -> _Call:
        prompt = load_prompt(prompt_name)
        if not self.can_call(state):
            return _Call(skipped="the run's LLM call, token or time budget was exhausted")
        request = self._request(
            state,
            node,
            task_type,
            prompt,
            prompt.render(values),
            schema=schema,
            max_tokens=max_tokens,
            evidence=evidence,
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
        updates: Update = {"retrieval_log": log_rows, "pending_queries": []}
        if new:
            new = self._attach_pages(state.workspace_id, new)
            unseen = sorted({i.document_id for i in new.values()} - set(state.documents))
            if unseen:
                try:
                    updates["documents"] = self.deps.temporal.versions(state.workspace_id, unseen)
                except MezvuatError:
                    log.warning("temporal metadata unavailable; dates will show as unknown")
        return {**updates, "evidence": new}

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
            injection_signals=injection_signals(chunk.text),
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
                (i for i in self.eligible(state)[0] if i.subquestion_id == sub.id),
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
                "evidence": evidence_section(
                    shown, boundary, max_chars=self.deps.assess_item_chars, dates=self.dates_label(state)
                ),
            },
            Sufficiency,
            700,
            evidence=shown,
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
        eligible = self.eligible(state)[0]
        for item in sorted(eligible, key=lambda i: (i.retrieval_round, i.rank, int(i.evidence_id[1:]))):
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
        if not resynthesis:
            updates["limitations"] = [*updates.get("limitations", []), *self.eligible(state)[1]]
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
            {
                "question": state.question,
                "evidence": evidence_section(chosen, new_boundary(), dates=self.dates_label(state)),
                "feedback": feedback,
            },
            Draft,
            3000,
            evidence=chosen,
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
            if self.leaks.leaks(claim.text):
                check = check.model_copy(
                    update={"passed": False, "problems": [*check.problems, "claim reproduces the system instructions"]}
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
        return self._semantic_checks(state, verdicts, registry)

    def _semantic_checks(
        self, state: ResearchState, verdicts: list[ClaimVerdict], registry: dict[str, EvidenceItem]
    ) -> Update:
        """NLI on every claim that passed the quote check, then the LLM judge for the ones NLI left undecided."""
        passing = [i for i, v in enumerate(verdicts) if v.status == "supported"]
        nli = self.deps.nli
        if nli is None or not passing:
            return {"verification": verdicts, "nli_unavailable": False}
        if self.remaining_s(state) < self.deps.nli_min_time_s:
            log.warning("too little run time left for NLI; claims keep the quote check", extra={"run_id": state.run_id})
            return {"verification": verdicts, "nli_unavailable": True}
        try:
            results = verify_claims(
                [(verdicts[i].text, verdicts[i].spans) for i in passing],
                registry,
                nli,
                self.deps.nli_thresholds,
                sentences=self.deps.nli_premise_sentences,
            )
        except Exception:  # The verifier is an extra layer; its failure must not cost the user the run.
            log.exception("NLI verification failed; claims keep the quote-match check", extra={"run_id": state.run_id})
            return {"verification": verdicts, "nli_unavailable": True}
        updates: Update = {}
        current = state
        judge_calls = 0
        judge_stopped = False
        for index, result in zip(passing, results, strict=True):
            verdict = self._apply_nli(verdicts[index], result)
            if (
                result.label == "not_enough_info"
                and self.deps.llm_judge
                and not judge_stopped
                and judge_calls < self.deps.max_judge_calls
                # One judge call, and the response still has its call.
                and self.can_call(current, calls=2)
            ):
                judge_calls += 1
                verdict, step, judge_stopped = self._judge(current, verdict, registry)
                if step:
                    updates = self._merge(updates, step)
                    current = current.model_copy(update={"budget": step["budget"]})
            verdicts[index] = verdict
        return {**updates, "verification": verdicts, "nli_unavailable": False}

    @staticmethod
    def _apply_nli(verdict: ClaimVerdict, result: NliVerdict) -> ClaimVerdict:
        if result.label == "supported":
            return verdict.model_copy(update={"status": "supported", "method": "nli", "nli": result})
        if result.label == "contradicted":
            against = [s.evidence_id for s in result.spans if s.label == "contradicted"]
            problem = f"NLI: evidence {', '.join(against)} contradicts this claim"
            return verdict.model_copy(
                update={
                    "status": "contradicted",
                    "method": "nli",
                    "nli": result,
                    "problems": [*verdict.problems, problem],
                }
            )
        # Kept, but only the quote check stands behind it: shown as partially supported, method quote_match.
        problem = "NLI: the quoted evidence does not clearly entail this claim"
        return verdict.model_copy(
            update={"status": "partially_supported", "nli": result, "problems": [*verdict.problems, problem]}
        )

    def _judge(
        self, state: ResearchState, verdict: ClaimVerdict, registry: dict[str, EvidenceItem]
    ) -> tuple[ClaimVerdict, Update, bool]:
        """One LLM-judge call (ADR-0009 step 3). Returns the verdict, the call's bookkeeping, and whether to stop."""
        cited = [registry[e] for e in verdict.evidence_ids if e in registry]
        call = self.structured(
            state,
            "verify",
            "citation_verification",
            "judge_claim",
            {
                "claim": _plain(verdict.text),
                "evidence": evidence_section(cited, new_boundary(), dates=self.dates_label(state)),
            },
            JudgeOutput,
            400,
            evidence=cited,
        )
        if not isinstance(call.parsed, JudgeOutput):
            reason = call.skipped or (call.error.code if call.error else "no output")
            problem = f"LLM judge unavailable ({reason})"
            return verdict.model_copy(update={"problems": [*verdict.problems, problem]}), call.updates, True
        out = call.parsed
        model = call.result.model if call.result else None
        if out.verdict == "not_enough_info":
            check = JudgeCheck(verdict="not_enough_info", accepted=True, span=None, problems=[], model=model)
            return verdict.model_copy(update={"judge": check}), call.updates, False
        span = check_quote(QuoteInput(evidence_id=out.evidence_id, text=out.quote), set(verdict.evidence_ids), registry)
        if span.status != "ok":
            problems = [f"LLM judge said {out.verdict}, but its quote for {out.evidence_id} failed: {span.status}"]
            check = JudgeCheck(verdict=out.verdict, accepted=False, span=None, problems=problems, model=model)
            judged = verdict.model_copy(update={"judge": check, "problems": [*verdict.problems, *problems]})
            return judged, call.updates, False
        check = JudgeCheck(verdict=out.verdict, accepted=True, span=span, problems=[], model=model)
        if out.verdict == "supported":
            judged = verdict.model_copy(update={"status": "supported", "method": "llm_judge", "judge": check})
            return judged, call.updates, False
        problem = f"LLM judge: evidence {out.evidence_id} contradicts this claim"
        judged = verdict.model_copy(
            update={
                "status": "contradicted",
                "method": "llm_judge",
                "judge": check,
                "problems": [*verdict.problems, problem],
            }
        )
        return judged, call.updates, False

    def route_after_verify(self, state: ResearchState) -> str:
        if state.status == "failed":
            return "fail"
        failing = any(v.status in ("unsupported", "contradicted") for v in state.verification or [])
        if failing and state.resynth_count < MAX_RESYNTH and self.can_call(state, calls=RESERVED_CALLS):
            return "synthesize"
        return "respond"

    # ----- temporal scoping (ADR-0010) ----------------------------------------------------------------------------

    @staticmethod
    def as_of(state: ResearchState) -> tuple[date | None, str | None]:
        """The explicit request date wins; a date the analysis read from the question is used and disclosed."""
        if state.as_of is not None:
            return state.as_of, None
        raw = state.analysis.temporal_scope.as_of if state.analysis else None
        if raw:
            try:
                parsed = date.fromisoformat(raw.strip())
            except ValueError:
                return None, None
            return parsed, f"The question was read as asking about the state on {parsed.isoformat()}."
        return None, None

    @staticmethod
    def wants_latest(state: ResearchState) -> bool:
        return state.filters.latest_only or bool(state.analysis and state.analysis.temporal_scope.wants_latest)

    @staticmethod
    def _version(state: ResearchState, item: EvidenceItem) -> VersionDates | None:
        for version in state.documents.get(item.document_id, []):
            if version.version_id == item.document_version_id:
                return version
        return None

    def _latest_flag(self, state: ResearchState, item: EvidenceItem) -> bool | None:
        version = self._version(state, item)
        return version.is_latest_version if version is not None else item.is_latest_version

    def version_facts(self, state: ResearchState, item: EvidenceItem) -> list[DateFact]:
        version = self._version(state, item)
        return list(version.facts) if version else []

    def dates_label(self, state: ResearchState) -> Callable[[EvidenceItem], str]:
        return lambda item: describe_dates(self.version_facts(state, item))

    @staticmethod
    def _doc_name(items: list[EvidenceItem]) -> str:
        return items[0].document_title or items[0].document_id

    def eligible(self, state: ResearchState) -> tuple[list[EvidenceItem], list[str]]:
        """Evidence a temporal question may use, and the limitations that explain what was excluded."""
        items = list(state.evidence.values())
        as_of, read_as = self.as_of(state)
        notes: list[str] = [read_as] if read_as else []
        by_doc: dict[str, list[EvidenceItem]] = {}
        for item in items:
            by_doc.setdefault(item.document_id, []).append(item)
        kept: list[EvidenceItem] = []
        if as_of is not None:
            for document_id, doc_items in by_doc.items():
                name = self._doc_name(doc_items)
                versions = state.documents.get(document_id, [])
                if not versions:
                    notes.append(f"{name}: no version dates are known, so it was not used for a claim about {as_of}.")
                    continue
                resolution = resolve_as_of(versions, as_of)
                labels = {v.version_id: v.version_label or v.version_id for v in versions}
                if resolution.unresolved:
                    unresolved = ", ".join(labels[v] for v in resolution.unresolved)
                    notes.append(
                        f"{name}: the dates of version(s) {unresolved} are unknown or too coarse to tell whether they "
                        f"applied on {as_of}; they were not used (temporally unresolved)."
                    )
                if resolution.applicable is None:
                    notes.append(f"{name}: no version can be shown to have applied on {as_of}.")
                    continue
                chosen = [i for i in doc_items if i.document_version_id == resolution.applicable]
                if not chosen:
                    notes.append(
                        f"{name}: version {labels[resolution.applicable]} applied on {as_of}, but none of its text "
                        "was among the retrieved evidence."
                    )
                kept.extend(chosen)
            return kept, notes
        if self.wants_latest(state):
            for doc_items in by_doc.values():
                # The version table is authoritative; the chunk payload's copy of the flag is the fallback.
                flags = {i.evidence_id: self._latest_flag(state, i) for i in doc_items}
                if any(flag is True for flag in flags.values()):
                    kept.extend(i for i in doc_items if flags[i.evidence_id] is True)
                else:
                    notes.append(f"{self._doc_name(doc_items)}: which version is the latest is unknown.")
                    kept.extend(doc_items)
            return kept, notes
        return items, notes

    @staticmethod
    def _source_lines(citations: list[Citation]) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()
        for citation in citations:
            if citation.evidence_id in seen:
                continue
            seen.add(citation.evidence_id)
            pages = sorted({c.page for c in citations if c.evidence_id == citation.evidence_id and c.page is not None})
            where = f", p. {', '.join(str(p) for p in pages)}" if pages else ""
            version = f" ({citation.version_label})" if citation.version_label else ""
            title = _plain(citation.document_title or citation.document_id)
            lines.append(f"- [{citation.evidence_id}] {title}{version}{where}: {describe_dates(citation.dates)}")
        return lines

    # ----- 8. respond ---------------------------------------------------------------------------------------------

    def respond(self, state: ResearchState) -> Update:
        writer = get_stream_writer()
        verdicts = state.verification or []
        verified = [v for v in verdicts if v.status in ("supported", "partially_supported")]
        dropped = [v for v in verdicts if v not in verified]
        limitations = list(dict.fromkeys(state.limitations))
        limitations += [self._dropped_note(v) for v in dropped]
        limitations += self._check_notes(state, verdicts, verified)
        flagged = [state.evidence[e] for e in state.shown_evidence_ids if state.evidence[e].injection_signals]
        if flagged:
            ids = ", ".join(item.evidence_id for item in flagged)
            limitations.append(
                f"Evidence {ids} contained instruction-like text. It was treated as quoted data, not as instructions."
            )
        if state.draft:
            limitations += [
                f"Not answered by the evidence: {_plain(gap)[:300]}"
                for gap in state.draft.gaps[:5]
                if not self.leaks.leaks(gap)
            ]
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
                if self.leaks.leaks(markdown):
                    flags.append("prompt_leak")
                    markdown = None
                elif not report.valid:
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
            citation
            for v in verified
            for span in v.spans
            if span.evidence_id in used and (citation := self._citation(state, v.claim_id, span)) is not None
        ]
        conflicts = self._conflicts(state, verdicts)
        full = body
        if conflicts:
            full += "\n\n### Conflicting evidence\n\n" + "\n".join(self._conflict_lines(conflicts))
        if citations:
            full += "\n\n### Sources\n\n" + "\n".join(self._source_lines(citations))
        if limitations:
            full += "\n\n### Limitations\n\n" + "\n".join(f"- {line}" for line in limitations)
        answer = FinalAnswer(
            markdown=full,
            rendering=rendering,
            citations=citations,
            claims=verdicts,
            limitations=limitations,
            flags=flags,
            conflicts=conflicts,
            verification=self._verification_info(state, verdicts),
        )
        writer({"event": "answer_final", "rendering": rendering})
        return {**updates, "answer": answer, "status": "completed"}

    def _dropped_note(self, verdict: ClaimVerdict) -> str:
        if self.leaks.leaks(verdict.text):
            return "A claim that reproduced the system's instructions was withheld."
        if verdict.status == "contradicted":
            return (
                f"This claim was left out because its cited evidence contradicts it (see Conflicting evidence): "
                f"{_plain(verdict.text)}"
            )
        return f"The system could not verify this claim, so it was left out: {_plain(verdict.text)}"

    def _check_notes(
        self, state: ResearchState, verdicts: list[ClaimVerdict], verified: list[ClaimVerdict]
    ) -> list[str]:
        """Say which check stands behind the answer's claims (ADR-0009 amendment, 2026-11-16)."""
        notes: list[str] = []
        unchecked = [v for v in verified if v.method == "quote_match" and v.nli is None]
        if unchecked:
            if state.nli_unavailable:
                notes.append(NLI_UNAVAILABLE_LIMITATION)
            notes.append(QUOTE_ONLY_LIMITATION)
        checked = [v for v in verdicts if v.nli is not None]
        if checked:
            model = checked[0].nli.model_id if checked[0].nli else "an NLI model"
            calibrated = all(v.nli.thresholds_calibrated for v in checked if v.nli)
            thresholds = (
                "Its thresholds were calibrated on labelled claim and evidence pairs."
                if calibrated
                else "Its decision thresholds are provisional: they have not been calibrated on labelled data yet."
            )
            notes.append(
                f"Claims were checked by a verbatim quote match, then by an NLI model ({model}) that compares each "
                f"claim with its quoted span and the neighbouring sentences. {thresholds}"
            )
        weak = [v for v in verified if v.status == "partially_supported"]
        if weak:
            notes.append(
                f"{len(weak)} claim(s) are backed only by their verbatim quote: the NLI check did not find that the "
                "quoted text entails them, so they are marked partially supported."
            )
        judged = [v for v in verdicts if v.judge is not None]
        if judged:
            notes.append(
                f"{len(judged)} claim(s) that the NLI model left undecided were checked by an LLM judge; a judge "
                "verdict counts only when its quote is verbatim in the cited evidence."
            )
        return notes

    def _citation(self, state: ResearchState, claim_id: str, span: SpanRef) -> Citation | None:
        if span.status != "ok" or span.char_start is None or span.char_end is None:
            return None
        item = state.evidence[span.evidence_id]
        return Citation(
            evidence_id=span.evidence_id,
            claim_id=claim_id,
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            document_version_id=item.document_version_id,
            document_title=item.document_title,
            version_label=item.version_label,
            page=span.page,
            char_start=span.char_start,
            char_end=span.char_end,
            quote=span.quote,
            dates=self.version_facts(state, item),
            injection_suspect=bool(item.injection_signals),
        )

    def _conflicts(self, state: ResearchState, verdicts: list[ClaimVerdict]) -> list[Conflict]:
        """Contradicted claims, with the spans that contradict them and any cited spans that support them."""
        conflicts: list[Conflict] = []
        for v in verdicts:
            if v.status != "contradicted" or self.leaks.leaks(v.text):
                continue
            spans = {s.evidence_id: s for s in v.spans if s.status == "ok"}
            against: list[SpanRef] = []
            backing: list[SpanRef] = []
            if v.method == "llm_judge" and v.judge is not None and v.judge.span is not None:
                against.append(v.judge.span)
            elif v.nli is not None:
                labelled = [(s.label, spans[s.evidence_id]) for s in v.nli.spans if s.evidence_id in spans]
                against = [span for label, span in labelled if label == "contradicted"]
                backing = [span for label, span in labelled if label == "supported"]
            contradicted_by = [c for s in against if (c := self._citation(state, v.claim_id, s)) is not None]
            supported_by = [c for s in backing if (c := self._citation(state, v.claim_id, s)) is not None]
            if not contradicted_by:
                continue
            between = bool(supported_by) and {c.document_id for c in contradicted_by} != {
                c.document_id for c in supported_by
            }
            conflicts.append(
                Conflict(
                    claim_id=v.claim_id,
                    claim_text=_plain(v.text),
                    method=v.method,
                    contradicted_by=contradicted_by,
                    supported_by=supported_by,
                    between_documents=between,
                )
            )
        return conflicts

    @staticmethod
    def _conflict_lines(conflicts: list[Conflict]) -> list[str]:
        """Markdown for the conflict section. Evidence ids appear without brackets: these are not answer citations."""

        def where(c: Citation) -> str:
            version = f" ({c.version_label})" if c.version_label else ""
            page = f", p. {c.page}" if c.page is not None else ""
            title = _plain(c.document_title or c.document_id)
            return f"evidence {c.evidence_id}, {title}{version}{page}: {describe_dates(c.dates)}"

        method = {"nli": "NLI", "llm_judge": "LLM judge"}
        lines: list[str] = []
        for conflict in conflicts:
            lines.append(f"- Claim left out of the answer: {conflict.claim_text}")
            check = method.get(conflict.method, conflict.method)
            lines += [f"  - Contradicted by {where(c)} ({check})" for c in conflict.contradicted_by]
            lines += [f"  - Supported by {where(c)} ({check})" for c in conflict.supported_by]
            if conflict.between_documents:
                lines.append("  - The documents disagree on this point.")
        return lines

    def _verification_info(self, state: ResearchState, verdicts: list[ClaimVerdict]) -> VerificationInfo:
        checked = [v.nli for v in verdicts if v.nli is not None]
        order: tuple[SupportMethod, ...] = ("quote_match", "nli", "llm_judge")
        used = {"quote_match"} if any(v.method != "none" for v in verdicts) else set()
        used |= {"nli"} if checked else set()
        used |= {"llm_judge"} if any(v.judge is not None for v in verdicts) else set()
        counts: dict[str, int] = {}
        for v in verdicts:
            counts[v.status] = counts.get(v.status, 0) + 1
        return VerificationInfo(
            checks=[m for m in order if m in used],
            nli_model=checked[0].model_id if checked else (self.deps.nli.model_id if self.deps.nli else None),
            nli_thresholds_calibrated=self.deps.nli_thresholds.calibrated if self.deps.nli is not None else None,
            nli_unavailable=state.nli_unavailable,
            llm_judge_enabled=self.deps.llm_judge,
            claims_by_status=counts,
        )

    @staticmethod
    def _deterministic(verified: Iterable[ClaimVerdict]) -> str:
        lines = []
        for v in verified:
            markers = "".join(f"[{e}]" for e in v.evidence_ids)
            label = "*Interpretation:* " if v.kind == "interpretation" else ""
            lines.append(f"- {label}{_plain(v.text)} {markers}")
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
