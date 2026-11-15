// M7 smoke test. The Mezvuat API is MOCKED: the first two tests answer the browser's requests with
// `page.route`; the third goes through the real Next.js proxy to `stub-server.mts`. This checks the UI against
// the OpenAPI-typed contract, not a live backend.
import { expect, test } from "@playwright/test";
import { QUOTE_INJECTION, QUOTE_RESULT, WORKSPACE, type Recorded } from "./fixtures.mts";
import { mockBackend } from "./mock-backend";

const STUB = `http://127.0.0.1:${process.env.STUB_PORT ?? 3199}`;

test("clicking a citation opens the right document, page and highlighted span (D1, D2, D8)", async ({ page }) => {
  const backend = await mockBackend(page);
  await page.goto(`/workspaces/${WORKSPACE.id}/research`);
  await expect(page.getByRole("heading", { name: `Research: ${WORKSPACE.name}` })).toBeVisible();

  await page.getByRole("textbox", { name: "Question" }).fill("How did hybrid retrieval change between versions?");
  await page.getByRole("button", { name: "Ask" }).click();

  // Streamed progress per node, from the SSE events.
  await expect(page.getByTestId("run-status")).toHaveText("Completed");
  const progress = page.getByRole("list", { name: "Workflow progress" });
  for (const node of ["understand_query", "plan_research", "retrieve", "assess_evidence", "synthesize", "verify", "respond"]) {
    await expect(progress.locator(`li[data-node="${node}"]`)).toHaveAttribute("data-state", "done");
  }
  await expect(progress.locator('li[data-node="reformulate"]')).toHaveAttribute("data-state", "skipped");

  // Sanitized Markdown: raw HTML, scripts, images and javascript: links never reach the DOM.
  const answer = page.locator(".answer");
  await expect(answer).toContainText("hybrid retrieval improved ranking quality over BM25");
  await expect(answer.locator("script, img, b")).toHaveCount(0);
  await expect(answer.locator('a[href^="javascript"]')).toHaveCount(0);
  await expect(answer).toContainText("bad link");
  await expect(answer.getByRole("link", { name: "the paper" })).toHaveAttribute("href", "https://arxiv.org/abs/2403.01234");
  await expect(answer).toContainText("(https://arxiv.org/abs/2403.01234)");
  expect(await page.evaluate(() => (window as unknown as { __pwned?: boolean }).__pwned)).toBeUndefined();

  // Backend, fallback and data-policy badges, from the run's llm_calls and summary.
  const badges = page.getByLabel("Inference details");
  await expect(badges).toContainText("backend: direct");
  await expect(badges).toContainText("provider: groq");
  await expect(badges).toContainText("no fallback");
  await expect(badges).toContainText("cloud allowed (public corpus)");

  // Citation E1: document, page 2, exact span highlighted, no injection warning.
  await page.getByRole("button", { name: "Show evidence E1" }).first().click();
  const viewer = page.getByRole("region", { name: "Evidence viewer" });
  await expect(viewer.getByTestId("evidence-document")).toHaveText("Temporal RAG Paper");
  await expect(viewer.getByTestId("evidence-page")).toHaveText("2");
  await expect(viewer.getByLabel("Current page")).toHaveText("Page 2 of 3");
  await expect(viewer.locator("mark")).toHaveText(QUOTE_RESULT);
  await expect(viewer.getByTestId("injection-warning")).toHaveCount(0);
  await expect(viewer).toContainText("Published 2024-03-05 · Revised 2024-06-10");
  const textRequest = backend.seen.find((r) => r.path.endsWith("/documents/doc_rag/versions/ver_2/text"));
  expect(textRequest?.search).toBe("?page=2");

  // Citation E2: page 1, its span highlighted, with the injection-suspect warning.
  await page.getByRole("button", { name: "Show evidence E2" }).first().click();
  await expect(viewer.getByLabel("Current page")).toHaveText("Page 1 of 3");
  await expect(viewer.locator("mark")).toHaveText(QUOTE_INJECTION);
  await expect(viewer.getByTestId("injection-warning")).toBeVisible();

  // Limitations and conflicts panels.
  await expect(page.getByRole("region", { name: "Limitations" })).toContainText("exact quote match only");
  await expect(page.getByRole("region", { name: "Conflicts" })).toContainText("No contradicted claims");

  // Feedback buttons post to the run's feedback endpoint.
  await page.getByRole("button", { name: "Helpful", exact: true }).click();
  await expect(page.getByLabel("Answer feedback")).toContainText("feedback saved");
  const feedback = backend.seen.filter((r) => r.path.endsWith("/feedback"));
  expect(feedback.map((r) => r.body)).toEqual([{ rating: "up", comment: null }]);

  // The document view opens at the same version, page and span.
  await viewer.getByRole("link", { name: "Open in the document view" }).click();
  await expect(page).toHaveURL(/\/documents\/doc_rag\?version=ver_2&page=1&start=\d+&end=\d+$/);
  await expect(page.getByTestId("document-title")).toHaveText("Temporal RAG Paper");
  await expect(page.getByLabel("Current page")).toHaveText("Page 1 of 3");
  await expect(page.getByTestId("page-text").locator("mark")).toHaveText(QUOTE_INJECTION);

  expect(backend.unmatched).toEqual([]);
});

test("workspaces and the document list show versions, dates by kind and ingestion status", async ({ page }) => {
  const backend = await mockBackend(page);
  await page.goto("/");
  await expect(page.getByRole("link", { name: WORKSPACE.name })).toBeVisible();
  await page.getByRole("link", { name: WORKSPACE.name }).click();

  await expect(page.getByRole("heading", { name: WORKSPACE.name })).toBeVisible();
  const rag = page.getByRole("article", { name: "Temporal RAG Paper" });
  await expect(rag).toContainText("v2");
  await expect(rag).toContainText("latest");
  await expect(rag).toContainText("Published 2024-03-05");
  // A month-precision date is shown at month precision, never as a day.
  await expect(rag).toContainText("Revised 2024-06");
  await expect(rag).not.toContainText("Revised 2024-06-01");
  await expect(rag).toContainText("verified public corpus");
  const broken = page.getByRole("article", { name: "Scanned report" });
  await expect(broken).toContainText("failed");
  await expect(broken).toContainText("The PDF has no text layer.");
  await expect(broken).toContainText("unknown");

  await page.getByLabel("Document URL").fill("https://arxiv.org/abs/2403.01234v2");
  await page.getByRole("button", { name: "Ingest", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Accepted");
  expect(backend.seen.filter((r) => r.method === "POST").map((r) => r.body)).toEqual([
    { url: "https://arxiv.org/abs/2403.01234v2" },
  ]);
  expect(backend.unmatched).toEqual([]);
});

test("through the Next.js proxy: SSE streams event by event and the API key stays on the server", async ({
  page,
  request,
}) => {
  await page.goto(`/workspaces/${WORKSPACE.id}/research`);
  await page.getByRole("textbox", { name: "Question" }).fill("How did hybrid retrieval change between versions?");
  await page.getByRole("button", { name: "Ask" }).click();

  // The stub sends one event every 100 ms, so the streamed text shows before the final answer replaces it.
  await page.getByTestId("streamed-answer").waitFor();
  await expect(page.getByTestId("streamed-answer")).toContainText("In the revised version");
  await expect(page.getByTestId("run-status")).toHaveText("Completed");
  await expect(page.getByTestId("streamed-answer")).toHaveCount(0);

  await page.getByRole("button", { name: "Show evidence E1" }).first().click();
  const viewer = page.getByRole("region", { name: "Evidence viewer" });
  await expect(viewer.getByLabel("Current page")).toHaveText("Page 2 of 3");
  await expect(viewer.locator("mark")).toHaveText(QUOTE_RESULT);

  const { seen, unmatched } = (await (await request.get(`${STUB}/__seen`)).json()) as {
    seen: Recorded[];
    unmatched: string[];
  };
  expect(unmatched).toEqual([]);
  expect(seen.some((r) => r.path.endsWith("/events"))).toBe(true);
  expect(seen.some((r) => r.path.endsWith("/versions/ver_2/text") && r.search === "?page=2")).toBe(true);
  expect(seen.every((r) => r.authorization === "Bearer smoke-test-key")).toBe(true);
  expect(await page.content()).not.toContain("smoke-test-key");
});
