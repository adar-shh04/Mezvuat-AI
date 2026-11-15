"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { api, errorMessage, type Citation, type VersionText } from "@/lib/api";
import { describeDates, highlightSegments } from "@/lib/citations";
import { Badge } from "./Badges";

export type ViewerTarget = {
  documentId: string;
  versionId: string;
  // The cited page; null when the version has no page map (TXT, Markdown, HTML).
  page: number | null;
  spans: ReadonlyArray<{ char_start: number; char_end: number }>;
};

export function targetFromCitations(citations: ReadonlyArray<Citation>): ViewerTarget | null {
  const first = citations[0];
  if (!first) return null;
  const sameVersion = citations.filter((c) => c.document_version_id === first.document_version_id);
  return {
    documentId: first.document_id,
    versionId: first.document_version_id,
    page: first.page,
    spans: sameVersion.map((c) => ({ char_start: c.char_start, char_end: c.char_end })),
  };
}

export function documentHref(workspaceId: string, target: ViewerTarget): string {
  const query = new URLSearchParams({ version: target.versionId });
  if (target.page !== null) query.set("page", String(target.page));
  const span = target.spans[0];
  if (span) {
    query.set("start", String(span.char_start));
    query.set("end", String(span.char_end));
  }
  return `/workspaces/${encodeURIComponent(workspaceId)}/documents/${encodeURIComponent(target.documentId)}?${query}`;
}

/** The cited page with the exact span highlighted. Offsets are absolute, so they need no re-matching here. */
export function PageText({ workspaceId, target }: { workspaceId: string; target: ViewerTarget }) {
  const [page, setPage] = useState<number | null>(target.page);
  const [data, setData] = useState<VersionText | null>(null);
  const [error, setError] = useState<string | null>(null);
  const firstMark = useRef<HTMLElement>(null);
  const anchor = target.spans[0]?.char_start;

  useEffect(() => {
    let live = true;
    const where = page !== null ? { page } : anchor !== undefined ? { at: anchor } : {};
    api
      .versionText(workspaceId, target.documentId, target.versionId, where)
      .then((text) => {
        if (!live) return;
        setData(text);
        setError(null);
      })
      .catch((e: unknown) => {
        if (live) setError(errorMessage(e));
      });
    return () => {
      live = false;
    };
  }, [workspaceId, target.documentId, target.versionId, page, anchor]);

  useEffect(() => {
    firstMark.current?.scrollIntoView?.({ block: "center" });
  }, [data]);

  if (error) {
    return (
      <p className="error" role="alert">
        Could not load the document text: {error}
      </p>
    );
  }
  if (!data) return <p className="muted">Loading the cited page…</p>;

  const segments = highlightSegments(data.text, data.start, target.spans);
  const firstHighlight = segments.findIndex((s) => s.highlighted);
  const spillsOver = target.spans.some((s) => s.char_end > data.end && s.char_start < data.end);
  return (
    <div>
      <div className="row spread">
        <span aria-label="Current page">
          {data.page !== null ? `Page ${data.page} of ${data.page_count}` : "No page map (text excerpt)"}
        </span>
        {data.page !== null && (
          <div className="row">
            <button
              type="button"
              className="secondary"
              disabled={data.page <= 1}
              onClick={() => setPage((data.page ?? 1) - 1)}
            >
              Previous page
            </button>
            <button
              type="button"
              className="secondary"
              disabled={data.page >= data.page_count}
              onClick={() => setPage((data.page ?? 0) + 1)}
            >
              Next page
            </button>
          </div>
        )}
      </div>
      {target.page !== null && data.page !== null && data.page !== target.page && (
        <p className="muted">You are on page {data.page}; the citation is on page {target.page}.</p>
      )}
      {spillsOver && <p className="muted">The quoted span continues on the next page.</p>}
      <div className="page-text" data-testid="page-text">
        {segments.map((s, i) => {
          if (!s.highlighted) return <span key={i}>{s.text}</span>;
          return (
            <mark key={i} ref={i === firstHighlight ? firstMark : undefined}>
              {s.text}
            </mark>
          );
        })}
      </div>
    </div>
  );
}

/** The evidence viewer in the research chat: one evidence id, its spans, dates and warnings, and the page. */
export function EvidenceViewer({
  workspaceId,
  citations,
  onClose,
}: {
  workspaceId: string;
  citations: ReadonlyArray<Citation>;
  onClose: () => void;
}) {
  const target = targetFromCitations(citations);
  const first = citations[0];
  if (!first || !target) return null;
  const suspect = citations.some((c) => c.injection_suspect);
  return (
    <section className="panel" aria-label="Evidence viewer">
      <div className="row spread">
        <h2>Evidence {first.evidence_id}</h2>
        <button type="button" className="secondary" onClick={onClose}>
          Close
        </button>
      </div>
      <dl className="meta">
        <dt>Document</dt>
        <dd data-testid="evidence-document">{first.document_title ?? first.document_id}</dd>
        <dt>Version</dt>
        <dd>{first.version_label ?? "unlabelled"}</dd>
        <dt>Cited page</dt>
        <dd data-testid="evidence-page">{first.page ?? "unknown"}</dd>
        <dt>Dates</dt>
        <dd>{describeDates(first.dates ?? [])}</dd>
      </dl>
      {suspect && (
        <p className="warning" role="alert" data-testid="injection-warning">
          <strong>Possible prompt injection.</strong> This passage contains instruction-like text. Mezvuat treated it
          as quoted data, not as instructions; read it with care.
        </p>
      )}
      <h3>Quoted span{citations.length > 1 ? "s" : ""}</h3>
      {citations.map((c, i) => (
        <blockquote key={`${c.claim_id}-${i}`}>
          {c.quote}
          <div className="row muted">
            <span>
              claim {c.claim_id} · characters {c.char_start}–{c.char_end}
            </span>
            {c.injection_suspect && <Badge tone="warn">injection suspect</Badge>}
          </div>
        </blockquote>
      ))}
      <PageText key={`${target.versionId}-${target.spans[0]?.char_start}`} workspaceId={workspaceId} target={target} />
      <p>
        <Link href={documentHref(workspaceId, target)}>Open in the document view</Link>
      </p>
    </section>
  );
}
