"use client";

import ReactMarkdown, { type Components } from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import type { Citation } from "@/lib/api";
import { citedEvidenceId, linkMarkers } from "@/lib/citations";

/**
 * Renders answer Markdown sanitized (ADR-0009 §5): raw HTML is skipped, the hast tree goes through
 * rehype-sanitize (GitHub's schema), react-markdown's URL filter drops `javascript:` and similar links, images
 * are never loaded, and links show their full URL. `[E#]` markers become buttons that open the evidence viewer.
 */
export function AnswerMarkdown({
  markdown,
  citations,
  selected,
  onCite,
}: {
  markdown: string;
  citations: ReadonlyArray<Citation>;
  selected: string | null;
  onCite: (evidenceId: string) => void;
}) {
  const known = new Set(citations.map((c) => c.evidence_id));
  const suspect = new Set(citations.filter((c) => c.injection_suspect).map((c) => c.evidence_id));
  const components: Components = {
    a({ href, children }) {
      const evidenceId = citedEvidenceId(href);
      if (evidenceId) {
        if (!known.has(evidenceId)) return <span className="muted">[{evidenceId}]</span>;
        return (
          <button
            type="button"
            className="cite"
            data-suspect={suspect.has(evidenceId)}
            aria-pressed={selected === evidenceId}
            aria-label={`Show evidence ${evidenceId}`}
            title={suspect.has(evidenceId) ? "This evidence contains instruction-like text" : "Show the cited span"}
            onClick={() => onCite(evidenceId)}
          >
            {evidenceId}
          </button>
        );
      }
      // react-markdown's URL filter empties unsafe hrefs (javascript:, data:); show those as plain text.
      if (!href) return <span>{children}</span>;
      // ADR-0009: links are shown with their full URL, so the reader sees where they lead.
      return (
        <>
          <a href={href} target="_blank" rel="noreferrer noopener">
            {children}
          </a>{" "}
          <span className="muted">({href})</span>
        </>
      );
    },
    img() {
      return null; // Answers never need images, and remote images would leak that the answer was viewed.
    },
  };
  return (
    <div className="answer">
      <ReactMarkdown skipHtml rehypePlugins={[rehypeSanitize]} components={components}>
        {linkMarkers(markdown)}
      </ReactMarkdown>
    </div>
  );
}

const LIMITATIONS_HEADING = "\n\n### Limitations\n\n";

/** The backend appends a Limitations section to the Markdown; the limitations panel shows it instead. */
export function withoutLimitations(markdown: string): string {
  const at = markdown.lastIndexOf(LIMITATIONS_HEADING);
  return at === -1 ? markdown : markdown.slice(0, at);
}
