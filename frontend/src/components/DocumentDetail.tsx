"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, errorMessage, type DocumentOut } from "@/lib/api";
import { PageText, type ViewerTarget } from "./EvidenceViewer";
import { VersionTable } from "./WorkspaceDocuments";

export type DocumentFocus = {
  versionId: string | null;
  page: number | null;
  start: number | null;
  end: number | null;
};

/** A document's metadata and versions, and one version's text, opened at a cited page and span if given. */
export function DocumentDetail({
  workspaceId,
  documentId,
  focus,
}: {
  workspaceId: string;
  documentId: string;
  focus: DocumentFocus;
}) {
  const [document, setDocument] = useState<DocumentOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [versionId, setVersionId] = useState<string | null>(focus.versionId);

  useEffect(() => {
    api
      .getDocument(workspaceId, documentId)
      .then((d) => {
        setDocument(d);
        setError(null);
      })
      .catch((e: unknown) => setError(errorMessage(e)));
  }, [workspaceId, documentId]);

  const versions = document ? [...document.versions].sort((a, b) => (b.version_ordinal ?? 0) - (a.version_ordinal ?? 0)) : [];
  const chosen = versions.find((v) => v.id === versionId) ?? versions.find((v) => v.is_latest_version) ?? versions[0];
  const focused = chosen && chosen.id === focus.versionId;
  const target: ViewerTarget | null = chosen
    ? {
        documentId,
        versionId: chosen.id,
        page: focused ? focus.page : null,
        spans:
          focused && focus.start !== null && focus.end !== null ? [{ char_start: focus.start, char_end: focus.end }] : [],
      }
    : null;

  return (
    <>
      <p>
        <Link href={`/workspaces/${encodeURIComponent(workspaceId)}`}>← Documents</Link>
      </p>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {document && (
        <>
          <h1 data-testid="document-title">{document.title ?? "Untitled document"}</h1>
          {document.authors && <p className="muted">{document.authors.join(", ")}</p>}
          <section className="panel">
            <h2>Versions</h2>
            <VersionTable versions={versions} />
          </section>
          <section className="panel" aria-label="Document text">
            <div className="row spread">
              <h2>Text</h2>
              <label className="row">
                Version
                <select value={chosen?.id ?? ""} onChange={(e) => setVersionId(e.target.value)}>
                  {versions.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.version_label ?? v.id}
                      {v.is_latest_version ? " (latest)" : ""}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {target ? (
              <PageText key={`${target.versionId}-${target.page}`} workspaceId={workspaceId} target={target} />
            ) : (
              <p className="muted">This document has no versions.</p>
            )}
          </section>
        </>
      )}
    </>
  );
}
