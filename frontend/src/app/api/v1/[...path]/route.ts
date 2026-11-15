// Same-origin proxy to the Mezvuat API. The browser never sees MEZVUAT_API_KEY: it is added here, on the server.
// Bodies stream in both directions, so uploads and the SSE progress stream pass through unbuffered.
import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const BACKEND = process.env.MEZVUAT_API_URL ?? "http://127.0.0.1:18000";
const FORWARD_REQUEST = ["accept", "content-type", "last-event-id", "x-request-id"];
const FORWARD_RESPONSE = ["content-type", "cache-control", "x-request-id", "retry-after"];

type Context = { params: Promise<{ path: string[] }> };

async function proxy(request: NextRequest, context: Context): Promise<Response> {
  const { path } = await context.params;
  if (path.some((part) => part === "" || part === "." || part === "..")) {
    return Response.json({ error: { code: "invalid_path", message: "Invalid API path." } }, { status: 400 });
  }
  const target = new URL(`/api/v1/${path.map(encodeURIComponent).join("/")}`, BACKEND);
  target.search = request.nextUrl.search;

  const headers = new Headers();
  for (const name of FORWARD_REQUEST) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const key = process.env.MEZVUAT_API_KEY;
  if (key) headers.set("authorization", `Bearer ${key}`);

  const init: RequestInit & { duplex?: "half" } = {
    method: request.method,
    headers,
    cache: "no-store",
    redirect: "manual",
    signal: request.signal,
  };
  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = request.body;
    init.duplex = "half";
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, init);
  } catch {
    return Response.json(
      { error: { code: "backend_unreachable", message: "The Mezvuat API is not reachable." } },
      { status: 502 },
    );
  }
  const out = new Headers();
  for (const name of FORWARD_RESPONSE) {
    const value = upstream.headers.get(name);
    if (value) out.set(name, value);
  }
  if (out.get("content-type")?.startsWith("text/event-stream")) {
    out.set("cache-control", "no-cache, no-transform");
    out.set("x-accel-buffering", "no");
  }
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export { proxy as DELETE, proxy as GET, proxy as PATCH, proxy as POST, proxy as PUT };
