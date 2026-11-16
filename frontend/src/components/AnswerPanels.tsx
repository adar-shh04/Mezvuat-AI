"use client";

import { useState } from "react";
import { api, errorMessage, type Citation, type ClaimVerdict, type FinalAnswer } from "@/lib/api";
import { describeDates } from "@/lib/citations";
import { Badge } from "./Badges";

const FLAG_TEXT: Record<string, string> = {
  abstained: "No claim could be verified, so the system abstained.",
  budget_exhausted_before_response: "The budget ran out before the answer was written; verified claims are listed.",
  prompt_leak: "The model's answer reproduced system instructions and was withheld.",
  unknown_citation_markers: "The model's answer cited evidence it was not given and was replaced.",
  answer_without_citations: "The model's answer had no citations and was replaced.",
};

const STATUS_TONE: Record<ClaimVerdict["status"], "ok" | "warn" | "bad" | "neutral"> = {
  supported: "ok",
  partially_supported: "warn",
  unsupported: "bad",
  contradicted: "bad",
  unverifiable: "neutral",
};

export function LimitationsPanel({ answer }: { answer: FinalAnswer }) {
  const flags = answer.flags.filter((f) => f in FLAG_TEXT);
  return (
    <section className="panel" aria-label="Limitations">
      <h3>Limitations</h3>
      {answer.rendering === "deterministic" && (
        <p className="muted">This answer is the plain list of verified claims, not model-written prose.</p>
      )}
      {flags.length > 0 && (
        <ul>
          {flags.map((f) => (
            <li key={f}>
              <Badge tone="warn">{f}</Badge> {FLAG_TEXT[f]}
            </li>
          ))}
        </ul>
      )}
      {answer.limitations.length === 0 && flags.length === 0 ? (
        <p className="muted">None reported.</p>
      ) : (
        <ul>
          {answer.limitations.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

const METHOD_TEXT: Record<string, string> = {
  quote_match: "quote match",
  nli: "NLI",
  llm_judge: "LLM judge",
  none: "none",
};

/** Which checks decided the claims, from the answer's verification summary. */
export function checksText(answer: FinalAnswer): string {
  const info = answer.verification;
  if (!info) return "Claims were checked by exact quote match.";
  const checks = info.checks.map((c) => METHOD_TEXT[c] ?? c).join(", then ");
  const parts = [`Checked by ${checks || "nothing"}.`];
  if (info.nli_model) {
    parts.push(
      `NLI model: ${info.nli_model}, ${info.nli_thresholds_calibrated ? "calibrated thresholds" : "provisional thresholds (not calibrated yet)"}.`,
    );
  }
  if (info.nli_unavailable) parts.push("The NLI verifier was unavailable for this run.");
  return parts.join(" ");
}

function ConflictSpan({
  citation,
  kind,
  claimId,
  onCite,
}: {
  citation: Citation;
  kind: "contradicts" | "supports";
  claimId: string;
  onCite: (claimId: string, evidenceId: string) => void;
}) {
  const where = [citation.document_title ?? citation.document_id, citation.version_label]
    .filter(Boolean)
    .join(" ");
  return (
    <li>
      <Badge tone={kind === "contradicts" ? "bad" : "ok"}>{kind}</Badge>{" "}
      <button
        type="button"
        className="cite"
        aria-label={`Show conflicting evidence ${citation.evidence_id}`}
        onClick={() => onCite(claimId, citation.evidence_id)}
      >
        {citation.evidence_id}
      </button>{" "}
      {where}
      {citation.page !== null && citation.page !== undefined ? `, p. ${citation.page}` : ""}
      <span className="muted"> · {describeDates(citation.dates ?? [])}</span>
      <blockquote className="muted">{citation.quote}</blockquote>
    </li>
  );
}

export function ConflictsPanel({
  answer,
  onCite,
}: {
  answer: FinalAnswer;
  onCite: (claimId: string, evidenceId: string) => void;
}) {
  const conflicts = answer.conflicts ?? [];
  return (
    <section className="panel" aria-label="Conflicts">
      <h3>Conflicting evidence</h3>
      {conflicts.length > 0 ? (
        <ul className="plain">
          {conflicts.map((c) => (
            <li key={c.claim_id} data-testid="conflict">
              <div>
                <strong>Left out of the answer:</strong> {c.claim_text}{" "}
                <Badge>{METHOD_TEXT[c.method] ?? c.method}</Badge>
                {c.between_documents && <Badge tone="warn">sources disagree</Badge>}
              </div>
              <ul className="plain">
                {c.contradicted_by.map((cit) => (
                  <ConflictSpan
                    key={`against-${cit.evidence_id}`}
                    citation={cit}
                    kind="contradicts"
                    claimId={c.claim_id}
                    onCite={onCite}
                  />
                ))}
                {c.supported_by.map((cit) => (
                  <ConflictSpan
                    key={`for-${cit.evidence_id}`}
                    citation={cit}
                    kind="supports"
                    claimId={c.claim_id}
                    onCite={onCite}
                  />
                ))}
              </ul>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">No contradicted claims.</p>
      )}
      {!answer.verification?.nli_model && (
        <p className="muted">Without the NLI verifier, contradicting evidence is not detected.</p>
      )}
    </section>
  );
}

export function ClaimsPanel({
  answer,
  workspaceId,
  runId,
  onCite,
}: {
  answer: FinalAnswer;
  workspaceId: string;
  runId: string;
  onCite: (evidenceId: string) => void;
}) {
  if (answer.claims.length === 0) return null;
  return (
    <section className="panel" aria-label="Claims">
      <h3>Claims and how they were checked</h3>
      <p className="muted" data-testid="checks-used">
        {checksText(answer)}
      </p>
      <ul className="plain">
        {answer.claims.map((c) => (
          <li key={c.claim_id}>
            <div className="row">
              <Badge tone={STATUS_TONE[c.status]}>{c.status.replaceAll("_", " ")}</Badge>
              <Badge>{c.method.replaceAll("_", " ")}</Badge>
              {c.kind === "interpretation" && <Badge>interpretation</Badge>}
            </div>
            <div>{c.text}</div>
            <div className="row">
              {c.evidence_ids.map((e) => (
                <button key={e} type="button" className="cite" onClick={() => onCite(e)}>
                  {e}
                </button>
              ))}
              <ClaimFeedback workspaceId={workspaceId} runId={runId} claimId={c.claim_id} />
            </div>
            {c.problems.length > 0 && <div className="muted">{c.problems.join("; ")}</div>}
          </li>
        ))}
      </ul>
    </section>
  );
}

function ClaimFeedback({ workspaceId, runId, claimId }: { workspaceId: string; runId: string; claimId: string }) {
  const [state, setState] = useState<"idle" | "sent" | "error">("idle");
  if (state === "sent") return <span className="muted">flagged</span>;
  return (
    <button
      type="button"
      className="link"
      title={state === "error" ? "Could not save; try again" : "Report this claim as wrong or unsupported"}
      onClick={() =>
        void api
          .sendFeedback(workspaceId, runId, { rating: "down", claim_id: claimId })
          .then(() => setState("sent"))
          .catch(() => setState("error"))
      }
    >
      {state === "error" ? "retry flag" : "flag claim"}
    </button>
  );
}

export function FeedbackButtons({ workspaceId, runId }: { workspaceId: string; runId: string }) {
  const [rating, setRating] = useState<"up" | "down" | null>(null);
  const [comment, setComment] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "sent">("idle");
  const [error, setError] = useState<string | null>(null);

  async function send(value: "up" | "down") {
    setRating(value);
    setStatus("sending");
    setError(null);
    try {
      await api.sendFeedback(workspaceId, runId, { rating: value, comment: comment.trim() || null });
      setStatus("sent");
    } catch (e) {
      setError(errorMessage(e));
      setStatus("idle");
    }
  }

  return (
    <div className="row" aria-label="Answer feedback">
      {status === "sent" ? (
        <span className="muted" role="status">
          Thanks: feedback saved ({rating === "up" ? "helpful" : "not helpful"}).
        </span>
      ) : (
        <>
          <span className="muted">Was this answer helpful?</span>
          <input
            type="text"
            className="grow"
            aria-label="Feedback comment"
            placeholder="Optional comment"
            maxLength={2000}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
          <button type="button" className="secondary" disabled={status === "sending"} onClick={() => void send("up")}>
            Helpful
          </button>
          <button type="button" className="secondary" disabled={status === "sending"} onClick={() => void send("down")}>
            Not helpful
          </button>
        </>
      )}
      {error && (
        <span className="error" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}
