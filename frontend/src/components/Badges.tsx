import type { LlmCall, RunOut, RunSummary } from "@/lib/api";

type Tone = "ok" | "warn" | "bad" | "info" | "neutral";

export function Badge({ tone = "neutral", title, children }: { tone?: Tone; title?: string; children: React.ReactNode }) {
  return (
    <span className={`badge ${tone === "neutral" ? "" : tone}`} title={title}>
      {children}
    </span>
  );
}

export function ClassificationBadge({ classification }: { classification: string }) {
  return classification === "public" ? (
    <Badge tone="info" title="Every document is verified to come from the public demo corpus">
      public corpus
    </Badge>
  ) : (
    <Badge tone="neutral" title="Calls carrying document text stay on local models (decision D1)">
      private
    </Badge>
  );
}

const STATUS_TONE: Record<string, Tone> = {
  ready: "ok",
  indexed_dense: "info",
  processing: "info",
  pending: "neutral",
  failed: "bad",
};

const STATUS_LABEL: Record<string, string> = {
  ready: "ready",
  indexed_dense: "indexing",
  processing: "processing",
  pending: "queued",
  failed: "failed",
};

export function IngestionBadge({ status }: { status: string }) {
  return <Badge tone={STATUS_TONE[status] ?? "neutral"}>{STATUS_LABEL[status] ?? status}</Badge>;
}

/**
 * Which inference path answered. Until M8 the only path is the direct backend; a run that fell back from
 * ModelMesh will carry `fallback: true` in its summary, or calls on more than one backend.
 */
export function BackendBadges({ run, summary }: { run: Pick<RunOut, "data_policy" | "llm_calls"> | null; summary?: RunSummary | null }) {
  const calls: LlmCall[] = run?.llm_calls ?? [];
  const backends = [...new Set(calls.map((c) => c.backend))];
  const backend = summary?.backend ?? backends[0] ?? null;
  const providers = summary?.providers?.length ? summary.providers : [...new Set(calls.map((c) => c.provider).filter(Boolean))];
  const models = [...new Set(calls.map((c) => c.model).filter(Boolean))];
  const fallback = summary?.fallback === true || backends.length > 1;
  return (
    <div className="row" aria-label="Inference details">
      {run && (
        <Badge
          tone={run.data_policy === "local_only" ? "neutral" : "info"}
          title="Decision D1: public workspaces may use cloud models; private ones stay local"
        >
          {run.data_policy === "local_only" ? "local only" : "cloud allowed (public corpus)"}
        </Badge>
      )}
      {backend && <Badge tone="info">backend: {backend}</Badge>}
      {providers.map((p) => (
        <Badge key={p}>provider: {p}</Badge>
      ))}
      {models.map((m) => (
        <Badge key={m}>model: {m}</Badge>
      ))}
      {fallback ? (
        <Badge tone="warn" title="The primary backend was unavailable for at least one call">
          fallback used
        </Badge>
      ) : (
        backend && <Badge tone="ok">no fallback</Badge>
      )}
    </div>
  );
}
