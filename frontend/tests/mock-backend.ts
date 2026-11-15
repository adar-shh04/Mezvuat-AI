// Serves the mock API through Playwright's `page.route`: the browser's own requests are answered before they
// leave it, so the Next.js proxy and the real backend are never reached.
import type { Page } from "@playwright/test";
import { createMockApi } from "./fixtures.mts";

export async function mockBackend(page: Page) {
  const api = createMockApi();
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const headers = request.headers();
    const reply = api.handle({
      method: request.method(),
      path: url.pathname,
      search: url.search,
      contentType: headers["content-type"] ?? null,
      rawBody: request.postData(),
      authorization: headers["authorization"] ?? null,
      lastEventId: headers["last-event-id"] ?? null,
    });
    await route.fulfill({ status: reply.status, contentType: reply.contentType, body: reply.body });
  });
  return api;
}
