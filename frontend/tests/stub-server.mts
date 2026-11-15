// A stand-in for the Mezvuat API on 127.0.0.1, used by the smoke test that goes through the Next.js proxy.
// Run with `node tests/stub-server.mts` (Node 24 strips the types). GET /__seen returns the recorded requests.
import { createServer } from "node:http";
import { createMockApi } from "./fixtures.mts";

const port = Number(process.env.STUB_PORT ?? 3199);
const api = createMockApi();

const server = createServer((request, response) => {
  const url = new URL(request.url ?? "/", `http://127.0.0.1:${port}`);
  if (url.pathname === "/__seen") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({ seen: api.seen, unmatched: api.unmatched }));
    return;
  }
  if (url.pathname === "/health") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end('{"status":"ok"}');
    return;
  }
  const chunks: Buffer[] = [];
  request.on("data", (chunk: Buffer) => chunks.push(chunk));
  request.on("end", () => {
    const header = (name: string) => {
      const value = request.headers[name];
      return typeof value === "string" ? value : null;
    };
    const reply = api.handle({
      method: request.method ?? "GET",
      path: url.pathname,
      search: url.search,
      contentType: header("content-type"),
      rawBody: chunks.length ? Buffer.concat(chunks).toString("utf8") : null,
      authorization: header("authorization"),
      lastEventId: header("last-event-id"),
    });
    response.writeHead(reply.status, { "content-type": reply.contentType });
    if (reply.contentType !== "text/event-stream") {
      response.end(reply.body);
      return;
    }
    // Stream the events one by one, as the real API does, so the proxy has to pass them through unbuffered.
    const events = reply.body.split("\n\n").filter(Boolean);
    let i = 0;
    const timer = setInterval(() => {
      const event = events[i++];
      if (event === undefined) {
        clearInterval(timer);
        response.end();
        return;
      }
      response.write(`${event}\n\n`);
    }, 100);
    response.on("close", () => clearInterval(timer));
  });
});

server.listen(port, "127.0.0.1", () => console.log(`mock Mezvuat API on http://127.0.0.1:${port}`));
