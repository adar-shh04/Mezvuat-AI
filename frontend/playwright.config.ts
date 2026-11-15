import { defineConfig, devices } from "@playwright/test";

// The smoke test mocks the Mezvuat API; no backend runs. Two tests answer the browser's requests with
// `page.route`. One goes through the Next.js `/api/v1` proxy to `tests/stub-server.mts`, a stand-in API.
// MEZVUAT_API_KEY here is a fake value used to check that the proxy adds the key on the server side.
// PW_SERVER=start tests the production build (`npm run build` first); the default is the dev server.
const PORT = 3100;
const STUB_PORT = 3199;
const BASE_URL = `http://localhost:${PORT}`;
const next =
  process.env.PW_SERVER === "start"
    ? `npx next start --hostname localhost --port ${PORT}`
    : `npx next dev --hostname localhost --port ${PORT}`;
process.env.STUB_PORT = String(STUB_PORT);

export default defineConfig({
  testDir: "./tests",
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "node tests/stub-server.mts",
      url: `http://127.0.0.1:${STUB_PORT}/health`,
      reuseExistingServer: false,
      timeout: 30_000,
      env: { STUB_PORT: String(STUB_PORT) },
    },
    {
      command: next,
      url: BASE_URL,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: {
        MEZVUAT_API_URL: `http://127.0.0.1:${STUB_PORT}`,
        MEZVUAT_API_KEY: "smoke-test-key",
        NEXT_TELEMETRY_DISABLED: "1",
      },
    },
  ],
});
