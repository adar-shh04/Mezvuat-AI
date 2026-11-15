import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // A self-contained server for the Docker image (frontend/Dockerfile sets NEXT_OUTPUT=standalone). Local
  // `next start` and the Playwright production run use the regular build.
  output: process.env.NEXT_OUTPUT === "standalone" ? "standalone" : undefined,
  // Do not let `next dev` write AGENTS.md / CLAUDE.md into the repository.
  agentRules: false,
  poweredByHeader: false,
  reactStrictMode: true,
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "same-origin" },
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
    ];
  },
};

export default nextConfig;
