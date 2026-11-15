"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage, type DocumentOut, type DocumentVersion, type SubmitOut, type Workspace } from "@/lib/api";
import { dateKindLabel, formatDate } from "@/lib/citations";
import { ClassificationBadge, IngestionBadge, Badge } from "./Badges";

const POLL_MS = 3000;
const IN_FLIGHT = new Set(["pending", "processing", "indexed_dense"]);

export function WorkspaceDocuments({ workspaceId }: { workspaceId: string }) {
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [version, setVersion] = useState(0);
  const reload = useCallback(() => setVersion((v) => v + 1), []);

  useEffect(() => {
    let live = true;
    Promise.all([api.getWorkspace(workspaceId), api.listDocuments(workspaceId)])
      .then(([ws, docs]) => {
        if (!live) return;
        setWorkspace(ws);
        setDocuments(docs);
        setError(null);
      })
      .catch((e: unknown) => {
        if (live) setError(errorMessage(e));
      });
    return () => {
      live = false;
    };
  }, [workspaceId, version]);

  // Poll while any version is still being ingested, so statuses move without a reload.
  const busy = documents?.some((d) => d.versions.some((v) => IN_FLIGHT.has(v.ingestion_status))) ?? false;
  useEffect(() => {
    if (!busy) return;
    const timer = window.setInterval(reload, POLL_MS);
    return () => window.clearInterval(timer);
  }, [busy, reload]);

  return (
    <>
      <div className="row spread">
        <div className="row">
          <h1>{workspace?.name ?? "Workspace"}</h1>
          {workspace && <ClassificationBadge classification={workspace.data_classification} />}
        </div>
        <Link href={`/workspaces/${encodeURIComponent(workspaceId)}/research`}>Open research chat →</Link>
      </div>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <IngestForms workspaceId={workspaceId} onSubmitted={reload} />
      <section className="panel">
        <h2>Documents</h2>
        {documents === null && !error && <p className="muted">Loading…</p>}
        {documents?.length === 0 && <p className="muted">No documents yet. Upload a file or ingest a URL.</p>}
        <ul className="plain" aria-label="Documents">
          {documents?.map((d) => (
            <li key={d.id}>
              <DocumentRow workspaceId={workspaceId} document={d} />
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}

function DocumentRow({ workspaceId, document }: { workspaceId: string; document: DocumentOut }) {
  const versions = [...document.versions].sort((a, b) => (b.version_ordinal ?? 0) - (a.version_ordinal ?? 0));
  return (
    <article aria-label={document.title ?? document.id}>
      <div className="row spread">
        <Link href={`/workspaces/${encodeURIComponent(workspaceId)}/documents/${encodeURIComponent(document.id)}`}>
          <strong>{document.title ?? "Untitled document"}</strong>
        </Link>
        {document.canonical_key && <span className="muted">{document.canonical_key}</span>}
      </div>
      {document.authors && document.authors.length > 0 && (
        <div className="muted">{document.authors.slice(0, 6).join(", ")}{document.authors.length > 6 ? ", …" : ""}</div>
      )}
      <VersionTable versions={versions} />
    </article>
  );
}

export function VersionTable({ versions }: { versions: DocumentVersion[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Version</th>
          <th>Status</th>
          <th>Dates by kind</th>
          <th>Ingested (not a document date)</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody>
        {versions.map((v) => (
          <tr key={v.id}>
            <td>
              <div className="row">
                <span>{v.version_label ?? (v.version_ordinal !== null ? `#${v.version_ordinal}` : "—")}</span>
                {v.is_latest_version && <Badge tone="ok">latest</Badge>}
                {v.probable_duplicate_of && (
                  <Badge tone="warn" title={`Same text as ${v.probable_duplicate_of}`}>
                    probable duplicate
                  </Badge>
                )}
              </div>
            </td>
            <td>
              <IngestionBadge status={v.ingestion_status} />
              {v.error_message && <div className="error">{v.error_code ? `${v.error_code}: ` : ""}{v.error_message}</div>}
            </td>
            <td>
              {v.temporal_facts.length === 0 ? (
                <span className="muted">unknown</span>
              ) : (
                <ul className="plain">
                  {v.temporal_facts.map((f, i) => (
                    <li key={`${f.kind}-${i}`}>
                      {dateKindLabel(f.kind)} {formatDate(f.value, f.precision)}{" "}
                      <span className="muted">
                        ({f.precision} precision, {f.source}, {f.confidence})
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </td>
            <td>{new Date(v.ingested_at).toLocaleString()}</td>
            <td>
              <div>{v.source_type}</div>
              {v.license && <div className="muted">{v.license}</div>}
              {v.public_corpus_verified && <Badge tone="info">verified public corpus</Badge>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function IngestForms({ workspaceId, onSubmitted }: { workspaceId: string; onSubmitted: () => void }) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);

  async function submit(work: () => Promise<SubmitOut>) {
    setBusy(true);
    setMessage(null);
    try {
      const result = await work();
      setMessage({
        tone: "ok",
        text:
          result.status === "duplicate"
            ? "Already in this workspace: no new version was created."
            : "Accepted. Ingestion runs in the background; the status below updates on its own.",
      });
      onSubmitted();
    } catch (e) {
      setMessage({ tone: "error", text: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  }

  function upload(event: React.FormEvent) {
    event.preventDefault();
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    void submit(() => api.uploadDocument(workspaceId, file)).then(() => {
      if (fileInput.current) fileInput.current.value = "";
    });
  }

  function ingest(event: React.FormEvent) {
    event.preventDefault();
    if (!url.trim()) return;
    void submit(() => api.ingestUrl(workspaceId, url.trim())).then(() => setUrl(""));
  }

  return (
    <section className="panel">
      <div className="grid-2">
        <form onSubmit={upload} aria-label="Upload a document">
          <h2>Upload</h2>
          <div className="row">
            <input ref={fileInput} type="file" aria-label="Document file" accept=".pdf,.txt,.md,.markdown,.html,.htm" />
            <button type="submit" disabled={busy}>
              Upload
            </button>
          </div>
          <p className="muted">PDF, plain text, Markdown or HTML.</p>
        </form>
        <form onSubmit={ingest} aria-label="Ingest a URL">
          <h2>Ingest a URL</h2>
          <div className="row">
            <input
              type="url"
              className="grow"
              aria-label="Document URL"
              placeholder="https://arxiv.org/abs/…"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
            <button type="submit" disabled={busy || !url.trim()}>
              Ingest
            </button>
          </div>
          <p className="muted">Fetched on the server through the SSRF guard; private addresses are refused.</p>
        </form>
      </div>
      {message && (
        <p className={message.tone === "ok" ? "muted" : "error"} role="status">
          {message.text}
        </p>
      )}
    </section>
  );
}
