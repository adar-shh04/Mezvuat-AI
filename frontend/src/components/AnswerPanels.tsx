"use client";

import { useState } from "react";
import { api, errorMessage, type ClaimVerdict, type FinalAnswer } from "@/lib/api";
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

export function ConflictsPanel({ answer }: { answer: FinalAnswer }) {
  const contradicted = answer.claims.filter((c) => c.status === "contradicted");
  return (
    <section className="panel" aria-label="Conflicts">
      <h3>Conflicting evidence</h3>
      {contradicted.length > 0 ? (
        <ul>
          {contradicted.map((c) => (
            <li key={c.claim_id}>
              {c.text} <span className="muted">({c.evidence_ids.join(", ")})</span>
              {c.problems.length > 0 && <div className="muted">{c.problems.join("; ")}</div>}
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">No contradicted claims.</p>
      )}
      <p className="muted">
        Claims are checked by exact quote match only. Detecting conflicts between sources needs the NLI verifier,
        which is not built yet (M5), so conflicting evidence can go unflagged.
      </p>
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
