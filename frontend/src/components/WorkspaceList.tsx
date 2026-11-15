"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, errorMessage, type Workspace } from "@/lib/api";
import { ClassificationBadge } from "./Badges";

export function WorkspaceList() {
  const [workspaces, setWorkspaces] = useState<Workspace[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);

  const [version, setVersion] = useState(0);
  const reload = useCallback(() => setVersion((v) => v + 1), []);

  useEffect(() => {
    let live = true;
    api
      .listWorkspaces()
      .then((list) => {
        if (!live) return;
        setWorkspaces(list);
        setError(null);
      })
      .catch((e: unknown) => {
        if (live) setError(errorMessage(e));
      });
    return () => {
      live = false;
    };
  }, [version]);

  async function create(event: React.FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    try {
      await api.createWorkspace(name.trim());
      setName("");
      reload();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove(workspace: Workspace) {
    const ok = window.confirm(
      `Delete "${workspace.name}"? This removes its documents, indexes and runs and cannot be undone.`,
    );
    if (!ok) return;
    try {
      await api.deleteWorkspace(workspace.id);
      reload();
    } catch (e) {
      setError(errorMessage(e));
    }
  }

  return (
    <>
      <section className="panel">
        <h2>New workspace</h2>
        <form className="row" onSubmit={create}>
          <input
            type="text"
            className="grow"
            aria-label="Workspace name"
            placeholder="Workspace name"
            value={name}
            maxLength={200}
            onChange={(e) => setName(e.target.value)}
          />
          <button type="submit" disabled={busy || !name.trim()}>
            Create
          </button>
        </form>
        <p className="muted">
          A workspace is private until every document in it is verified to come from the CC-licensed demo corpus
          (decision D1). Private workspaces only use local models.
        </p>
      </section>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <section className="panel">
        <h2>Workspaces</h2>
        {workspaces === null && !error && <p className="muted">Loading…</p>}
        {workspaces?.length === 0 && <p className="muted">No workspaces yet.</p>}
        <ul className="plain" aria-label="Workspaces">
          {workspaces?.map((w) => (
            <li key={w.id} className="row spread">
              <div className="row">
                <Link href={`/workspaces/${encodeURIComponent(w.id)}`}>{w.name}</Link>
                <ClassificationBadge classification={w.data_classification} />
                <span className="muted">created {new Date(w.created_at).toLocaleDateString()}</span>
              </div>
              <div className="row">
                <Link href={`/workspaces/${encodeURIComponent(w.id)}/research`}>Research</Link>
                <button type="button" className="danger" onClick={() => void remove(w)}>
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
