"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage, type RetrievalMode, type RunOut, type Workspace } from "@/lib/api";
import { citationsFor } from "@/lib/citations";
import {
  EVENT_NAMES,
  TERMINAL_EVENTS,
  initialProgress,
  reduceProgress,
  type Progress,
  type RunEvent,
} from "@/lib/events";
import { AnswerMarkdown, withoutLimitations } from "./AnswerMarkdown";
import { ClaimsPanel, ConflictsPanel, FeedbackButtons, LimitationsPanel } from "./AnswerPanels";
import { BackendBadges, ClassificationBadge } from "./Badges";
import { EvidenceViewer } from "./EvidenceViewer";
import { RunProgress } from "./RunProgress";

type Turn = {
  key: string;
  question: string;
  runId: string | null;
  progress: Progress;
  run: RunOut | null;
  error: string | null;
};

type Selection = { turnKey: string; evidenceId: string } | null;

const MODES: Array<{ value: RetrievalMode; label: string }> = [
  { value: "hybrid_rrf", label: "Hybrid (BM25 + dense)" },
  { value: "hybrid_rrf_rerank", label: "Hybrid + reranker" },
  { value: "bm25", label: "BM25 only" },
  { value: "dense", label: "Dense only" },
];

function progressFromRun(run: RunOut): Progress {
  return {
    ...initialProgress,
    status: run.status,
    answer: run.answer,
    error: run.error ? { code: run.error.code, message: run.error.message ?? "The run failed." } : null,
  };
}

export function ResearchChat({ workspaceId }: { workspaceId: string }) {
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid_rrf");
  const [asOf, setAsOf] = useState("");
  const [latestOnly, setLatestOnly] = useState(false);
  const [selection, setSelection] = useState<Selection>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const sources = useRef(new Map<string, EventSource>());

  useEffect(() => {
    api
      .getWorkspace(workspaceId)
      .then(setWorkspace)
      .catch((e: unknown) => setLoadError(errorMessage(e)));
  }, [workspaceId]);

  useEffect(() => {
    const open = sources.current;
    return () => {
      for (const source of open.values()) source.close();
      open.clear();
    };
  }, []);

  const update = useCallback((key: string, change: (turn: Turn) => Turn) => {
    setTurns((all) => all.map((t) => (t.key === key ? change(t) : t)));
  }, []);

  const loadRun = useCallback(
    async (key: string, runId: string) => {
      try {
        const run = await api.getRun(workspaceId, runId);
        update(key, (t) => ({
          ...t,
          run,
          // The stored run is the source of truth when the stream was lost before its terminal event.
          progress: t.progress.status === "completed" || t.progress.status === "failed" ? t.progress : progressFromRun(run),
        }));
      } catch (e) {
        update(key, (t) => ({ ...t, error: errorMessage(e) }));
      }
    },
    [update, workspaceId],
  );

  const follow = useCallback(
    (key: string, runId: string) => {
      const source = new EventSource(api.runEventsUrl(workspaceId, runId));
      sources.current.set(key, source);
      for (const name of EVENT_NAMES) {
        source.addEventListener(name, (message) => {
          const event = { event: name, data: JSON.parse((message as MessageEvent<string>).data) } as RunEvent;
          update(key, (t) => ({ ...t, progress: reduceProgress(t.progress, event) }));
          if (TERMINAL_EVENTS.includes(name)) {
            source.close();
            sources.current.delete(key);
            void loadRun(key, runId);
          }
        });
      }
      // EventSource reconnects by itself (sending Last-Event-ID). If it gives up, read the stored run instead.
      source.onerror = () => {
        if (source.readyState === EventSource.CLOSED) {
          sources.current.delete(key);
          void loadRun(key, runId);
        }
      };
    },
    [loadRun, update, workspaceId],
  );

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text) return;
    const key = `turn-${Date.now()}`;
    setTurns((all) => [...all, { key, question: text, runId: null, progress: initialProgress, run: null, error: null }]);
    setQuestion("");
    try {
      const created = await api.createRun(workspaceId, {
        question: text,
        mode,
        as_of: asOf || null,
        filters: {
          document_ids: [],
          latest_only: latestOnly,
          include_references: false,
          published_from: null,
          published_to: null,
        },
      });
      update(key, (t) => ({ ...t, runId: created.run_id }));
      follow(key, created.run_id);
    } catch (e) {
      update(key, (t) => ({ ...t, error: errorMessage(e), progress: { ...t.progress, status: "failed" } }));
    }
  }

  const selectedTurn = selection ? turns.find((t) => t.key === selection.turnKey) : undefined;
  const selectedCitations =
    selection && selectedTurn?.progress.answer
      ? citationsFor(selectedTurn.progress.answer.citations, selection.evidenceId)
      : [];
  const running = turns.some((t) => t.progress.status === "queued" || t.progress.status === "running");

  return (
    <>
      <div className="row spread">
        <div className="row">
          <h1>Research{workspace ? `: ${workspace.name}` : ""}</h1>
          {workspace && <ClassificationBadge classification={workspace.data_classification} />}
        </div>
        <Link href={`/workspaces/${encodeURIComponent(workspaceId)}`}>← Documents</Link>
      </div>
      {loadError && (
        <p className="error" role="alert">
          {loadError}
        </p>
      )}
      <div className="chat-layout">
        <div>
          <section className="panel" aria-label="Conversation">
            {turns.length === 0 && (
              <p className="muted">
                Ask a question about this workspace&apos;s documents. Every claim in the answer cites an exact quote;
                click a citation to see it on its page.
              </p>
            )}
            {turns.map((turn) => (
              <TurnView
                key={turn.key}
                turn={turn}
                workspaceId={workspaceId}
                selected={selection?.turnKey === turn.key ? selection.evidenceId : null}
                onCite={(evidenceId) => setSelection({ turnKey: turn.key, evidenceId })}
              />
            ))}
          </section>
          <form className="panel" onSubmit={ask} aria-label="Ask a question">
            <textarea
              aria-label="Question"
              rows={3}
              maxLength={2000}
              placeholder="What does the latest version of the paper say about…?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void ask(e);
              }}
            />
            <div className="row spread">
              <div className="row">
                <label className="row">
                  Retrieval
                  <select value={mode} onChange={(e) => setMode(e.target.value as RetrievalMode)}>
                    {MODES.map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="row" title="Use only the version of each document that applied on this date">
                  As of
                  <input type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} />
                </label>
                <label className="row">
                  <input type="checkbox" checked={latestOnly} onChange={(e) => setLatestOnly(e.target.checked)} />
                  Latest versions only
                </label>
              </div>
              <button type="submit" disabled={!question.trim() || running}>
                Ask
              </button>
            </div>
          </form>
        </div>
        <aside className="sticky">
          {selection && selectedCitations.length > 0 ? (
            <EvidenceViewer
              key={`${selection.turnKey}-${selection.evidenceId}`}
              workspaceId={workspaceId}
              citations={selectedCitations}
              onClose={() => setSelection(null)}
            />
          ) : (
            <section className="panel">
              <h2>Evidence</h2>
              <p className="muted">Click a citation such as [E1] to open the cited page with the quote highlighted.</p>
            </section>
          )}
        </aside>
      </div>
    </>
  );
}

function TurnView({
  turn,
  workspaceId,
  selected,
  onCite,
}: {
  turn: Turn;
  workspaceId: string;
  selected: string | null;
  onCite: (evidenceId: string) => void;
}) {
  const { progress, run } = turn;
  const answer = progress.answer;
  return (
    <article className="turn" aria-label={`Question: ${turn.question}`}>
      <div className="question">{turn.question}</div>
      <RunProgress progress={progress} />
      {progress.notices.map((n, i) => (
        <p key={i} className="muted">
          {n}
        </p>
      ))}
      {!answer && progress.streamed && (
        <div className="streaming" aria-live="polite" data-testid="streamed-answer">
          {progress.streamed}
        </div>
      )}
      {turn.error && (
        <p className="error" role="alert">
          {turn.error}
        </p>
      )}
      {progress.error && (
        <p className="error" role="alert">
          The run failed ({progress.error.code}): {progress.error.message}
        </p>
      )}
      {answer && (
        <>
          <AnswerMarkdown
            markdown={withoutLimitations(answer.markdown)}
            citations={answer.citations}
            selected={selected}
            onCite={onCite}
          />
          <BackendBadges run={run} summary={run?.summary} />
          {turn.runId && <FeedbackButtons workspaceId={workspaceId} runId={turn.runId} />}
          <LimitationsPanel answer={answer} />
          <ConflictsPanel answer={answer} />
          {turn.runId && <ClaimsPanel answer={answer} workspaceId={workspaceId} runId={turn.runId} onCite={onCite} />}
        </>
      )}
    </article>
  );
}
