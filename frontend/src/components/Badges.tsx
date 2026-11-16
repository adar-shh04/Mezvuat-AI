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
 * Which inference path answered, from the run's `llm_calls` rows and summary (M8). A call answered by a direct
 * backend because ModelMesh could not serve it carries `fallback_reason`; the summary lists the reasons. Cost is
 * ModelMesh's estimate and is shown only when ModelMesh reported one.
 */
export function BackendBadges({ run, summary }: { run: Pick<RunOut, "data_policy" | "llm_calls"> | null; summary?: RunSummary | null }) {
  const calls: LlmCall[] = run?.llm_calls ?? [];
  const backends = summary?.backends?.length ? [...summary.backends] : [...new Set(calls.map((c) => c.backend))];
  if (!backends.length && summary?.backend) backends.push(summary.backend);
  const providers = summary?.providers?.length ? summary.providers : [...new Set(calls.map((c) => c.provider).filter(Boolean))];
  const models = [...new Set(calls.map((c) => c.model).filter(Boolean))];
  const fallbackCalls = calls.filter((c) => c.fallback_reason && c.status === "ok");
  const reasons = summary?.fallback_reasons?.length
    ? summary.fallback_reasons
    : [...new Set(calls.map((c) => c.fallback_reason).filter((r): r is string => Boolean(r)))];
  const fallback = summary?.fallback === true || reasons.length > 0;
  const cacheHits = calls.filter((c) => c.cache_status?.startsWith("hit")).length;
  const costs = calls.map((c) => c.cost_estimated_usd).filter((c): c is number => typeof c === "number");
  const cost = costs.length ? costs.reduce((a, b) => a + b, 0) : null;
  const answered = calls.filter((c) => c.status === "ok").length;
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
      {backends.map((b) => (
        <Badge key={b} tone="info">
          backend: {b}
        </Badge>
      ))}
      {providers.map((p) => (
        <Badge key={p}>provider: {p}</Badge>
      ))}
      {models.map((m) => (
        <Badge key={m}>model: {m}</Badge>
      ))}
      {fallback ? (
        <>
          <Badge
            tone="warn"
            title={`ModelMesh could not serve ${fallbackCalls.length} of ${answered} answered calls; a direct backend answered them under the same data policy`}
          >
            fallback used
          </Badge>
          {reasons.map((r) => (
            <Badge key={r} tone="warn">
              fallback reason: {r}
            </Badge>
          ))}
        </>
      ) : (
        backends.length > 0 && <Badge tone="ok">no fallback</Badge>
      )}
      {cacheHits > 0 && <Badge title="Calls ModelMesh answered from its cache">cache hits: {cacheHits}</Badge>}
      {cost !== null && (
        <Badge title="ModelMesh's estimate from its pricing tables; not billing data">
          est. cost ${cost.toFixed(5)}
        </Badge>
      )}
    </div>
  );
}
