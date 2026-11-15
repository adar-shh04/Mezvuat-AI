// Server-Sent Events from `GET /runs/{id}/events` (backend `api/routers/runs.py`). SSE is not described by
// OpenAPI, so the event payloads are typed here; the answer inside them is the generated `FinalAnswer`.
import type { FinalAnswer, RunSummary } from "./api";

// The graph's stages in their usual order (ADR-0008). `reformulate` runs only when evidence is insufficient.
export const WORKFLOW_NODES = [
  "understand_query",
  "plan_research",
  "retrieve",
  "assess_evidence",
  "reformulate",
  "synthesize",
  "verify",
  "respond",
] as const;

export const NODE_LABELS: Record<string, string> = {
  understand_query: "Understand the question",
  plan_research: "Plan the research",
  retrieve: "Retrieve evidence",
  assess_evidence: "Assess the evidence",
  reformulate: "Reformulate queries",
  synthesize: "Synthesize claims",
  verify: "Verify quotes",
  respond: "Write the answer",
  fail: "Stopped",
};

export type RunEvent =
  | { event: "run_queued" | "run_started"; data: { run_id: string } }
  | { event: "node"; data: { node: string } }
  | { event: "answer_delta"; data: { text: string } }
  | { event: "answer_reset" | "answer_replaced"; data: { reason: string } }
  | { event: "answer_final"; data: { rendering: "model" | "deterministic" } }
  | { event: "run_completed"; data: { run_id: string; answer: FinalAnswer | null; summary?: RunSummary } }
  | {
      event: "run_failed";
      data: {
        run_id: string;
        code?: string;
        message?: string | null;
        error?: { code: string; message: string | null } | null;
      };
    };

export const EVENT_NAMES: ReadonlyArray<RunEvent["event"]> = [
  "run_queued",
  "run_started",
  "node",
  "answer_delta",
  "answer_reset",
  "answer_replaced",
  "answer_final",
  "run_completed",
  "run_failed",
];

export const TERMINAL_EVENTS: ReadonlyArray<RunEvent["event"]> = ["run_completed", "run_failed"];

export type Progress = {
  status: "queued" | "running" | "completed" | "failed";
  // Nodes in the order they finished; a node can repeat (the bounded research and re-synthesis loops).
  nodes: string[];
  streamed: string;
  notices: string[];
  answer: FinalAnswer | null;
  error: { code: string; message: string } | null;
};

export const initialProgress: Progress = {
  status: "queued",
  nodes: [],
  streamed: "",
  notices: [],
  answer: null,
  error: null,
};

export function reduceProgress(state: Progress, event: RunEvent): Progress {
  switch (event.event) {
    case "run_queued":
    case "answer_final":
      return state;
    case "run_started":
      return { ...state, status: "running" };
    case "node":
      return { ...state, status: "running", nodes: [...state.nodes, event.data.node] };
    case "answer_delta":
      return { ...state, streamed: state.streamed + event.data.text };
    case "answer_reset":
      return { ...state, streamed: "", notices: [...state.notices, `Answer restarted: ${event.data.reason}`] };
    case "answer_replaced":
      return {
        ...state,
        notices: [
          ...state.notices,
          `The model's answer failed validation (${event.data.reason}); the verified claims are shown instead.`,
        ],
      };
    case "run_completed":
      return { ...state, status: "completed", answer: event.data.answer };
    case "run_failed": {
      const code = event.data.code ?? event.data.error?.code ?? "run_failed";
      const message = event.data.message ?? event.data.error?.message ?? "The run did not produce an answer.";
      return { ...state, status: "failed", error: { code, message } };
    }
  }
}
