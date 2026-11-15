import { NODE_LABELS, WORKFLOW_NODES, type Progress } from "@/lib/events";

/** One chip per workflow stage, updated from the SSE `node` events as each stage finishes. */
export function RunProgress({ progress }: { progress: Progress }) {
  const counts = new Map<string, number>();
  for (const node of progress.nodes) counts.set(node, (counts.get(node) ?? 0) + 1);
  const finished = progress.status === "completed" || progress.status === "failed";
  const last = progress.nodes[progress.nodes.length - 1];
  const status =
    progress.status === "queued"
      ? "Queued…"
      : progress.status === "running"
        ? last
          ? `Running: ${NODE_LABELS[last] ?? last} done`
          : "Starting…"
        : progress.status === "completed"
          ? "Completed"
          : "Failed";
  return (
    <div>
      <div className="muted" role="status" aria-live="polite" data-testid="run-status">
        {status}
      </div>
      <ol className="progress" aria-label="Workflow progress">
        {WORKFLOW_NODES.map((node) => {
          const count = counts.get(node) ?? 0;
          const state = count > 0 ? "done" : finished ? "skipped" : "pending";
          return (
            <li key={node} data-node={node} data-state={state}>
              {NODE_LABELS[node]}
              {count > 1 ? ` ×${count}` : ""}
            </li>
          );
        })}
        {counts.has("fail") && (
          <li data-node="fail" data-state="done">
            {NODE_LABELS.fail}
          </li>
        )}
      </ol>
    </div>
  );
}
