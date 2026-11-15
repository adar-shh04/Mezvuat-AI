// Typed client for the Mezvuat API. Every type comes from `api-types.ts`, generated from the backend's OpenAPI
// (`npm run gen:api`). Requests go to this app's `/api/v1` proxy, which adds the API key on the server side.
import type { components } from "./api-types";

type Schemas = components["schemas"];
export type Workspace = Schemas["WorkspaceOut"];
export type DocumentOut = Schemas["DocumentOut"];
export type DocumentVersion = Schemas["DocumentVersionOut"];
export type TemporalFact = Schemas["TemporalFactOut"];
export type SubmitOut = Schemas["SubmitOut"];
export type RunCreate = Schemas["RunCreate"];
export type RunCreated = Schemas["RunCreated"];
export type RunOut = Schemas["RunOut"];
export type RunSummary = Schemas["RunSummaryOut"];
export type LlmCall = Schemas["LlmCallOut"];
export type FinalAnswer = Schemas["FinalAnswer"];
export type Citation = Schemas["Citation"];
export type ClaimVerdict = Schemas["ClaimVerdict"];
export type DateFact = Schemas["DateFact"];
export type FeedbackCreate = Schemas["FeedbackCreate"];
export type FeedbackOut = Schemas["FeedbackOut"];
export type VersionText = Schemas["VersionTextOut"];
export type RetrievalMode = RunCreate["mode"];

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

const BASE = "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(0, "network_error", "The Mezvuat server could not be reached.");
  }
  if (response.status === 204) return undefined as T;
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) throw toError(response.status, body);
  return body as T;
}

function toError(status: number, body: unknown): ApiError {
  if (body && typeof body === "object") {
    const record = body as Record<string, unknown>;
    const error = record.error as { code?: string; message?: string } | undefined;
    if (error?.message) return new ApiError(status, error.code ?? "error", error.message);
    if (Array.isArray(record.detail)) {
      const first = record.detail[0] as { msg?: string } | undefined;
      return new ApiError(status, "invalid_input", first?.msg ?? "The request was not valid.");
    }
  }
  return new ApiError(status, "error", `Request failed with status ${status}.`);
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});
const seg = encodeURIComponent;

export const api = {
  listWorkspaces: () => request<Workspace[]>("/workspaces"),
  getWorkspace: (id: string) => request<Workspace>(`/workspaces/${seg(id)}`),
  createWorkspace: (name: string) =>
    request<Workspace>("/workspaces", json({ name } satisfies Schemas["WorkspaceCreate"])),
  deleteWorkspace: (id: string) => request<void>(`/workspaces/${seg(id)}`, { method: "DELETE" }),

  listDocuments: (ws: string) => request<DocumentOut[]>(`/workspaces/${seg(ws)}/documents`),
  getDocument: (ws: string, doc: string) => request<DocumentOut>(`/workspaces/${seg(ws)}/documents/${seg(doc)}`),
  uploadDocument: (ws: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<SubmitOut>(`/workspaces/${seg(ws)}/documents`, { method: "POST", body: form });
  },
  ingestUrl: (ws: string, url: string) =>
    request<SubmitOut>(`/workspaces/${seg(ws)}/documents/url`, json({ url } satisfies Schemas["UrlIngestRequest"])),
  versionText: (ws: string, doc: string, version: string, where: { page?: number; at?: number }) => {
    const query = new URLSearchParams();
    if (where.page !== undefined) query.set("page", String(where.page));
    if (where.at !== undefined) query.set("at", String(where.at));
    return request<VersionText>(
      `/workspaces/${seg(ws)}/documents/${seg(doc)}/versions/${seg(version)}/text?${query.toString()}`,
    );
  },

  createRun: (ws: string, body: RunCreate) => request<RunCreated>(`/workspaces/${seg(ws)}/runs`, json(body)),
  getRun: (ws: string, run: string) => request<RunOut>(`/workspaces/${seg(ws)}/runs/${seg(run)}`),
  runEventsUrl: (ws: string, run: string) => `${BASE}/workspaces/${seg(ws)}/runs/${seg(run)}/events`,
  sendFeedback: (ws: string, run: string, body: FeedbackCreate) =>
    request<FeedbackOut>(`/workspaces/${seg(ws)}/runs/${seg(run)}/feedback`, json(body)),
};
