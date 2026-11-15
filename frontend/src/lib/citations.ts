// Citation markers, span highlighting and date display. Offsets are absolute in the version's normalized text
// (ADR-0009); the viewer's page text starts at `start`, so a span on the page is [char_start - start, char_end - start).
import type { Citation, DateFact } from "./api";

const MARKER = /\[(E\d+)\]/g;
export const CITE_PREFIX = "#cite-";

/** Turns `[E3]` into a fragment link that the Markdown renderer swaps for a citation button. */
export function linkMarkers(markdown: string): string {
  return markdown.replace(MARKER, (_match, id: string) => `[${id}](${CITE_PREFIX}${id})`);
}

/** The evidence id behind a rendered link, allowing for the sanitizer's `user-content-` clobber prefix. */
export function citedEvidenceId(href: string | undefined): string | null {
  if (!href) return null;
  const match = /^#(?:user-content-)?cite-(E\d+)$/.exec(href);
  return match?.[1] ?? null;
}

export type Segment = { text: string; highlighted: boolean };

/** Splits page text into plain and highlighted runs. Overlapping spans merge; spans past the page are clipped. */
export function highlightSegments(
  pageText: string,
  pageStart: number,
  spans: ReadonlyArray<{ char_start: number; char_end: number }>,
): Segment[] {
  const ranges = spans
    .map((s): [number, number] => [
      Math.max(0, s.char_start - pageStart),
      Math.min(pageText.length, s.char_end - pageStart),
    ])
    .filter(([a, b]) => b > a)
    .sort((x, y) => x[0] - y[0]);
  const merged: Array<[number, number]> = [];
  for (const [a, b] of ranges) {
    const last = merged[merged.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else merged.push([a, b]);
  }
  const out: Segment[] = [];
  let cursor = 0;
  for (const [a, b] of merged) {
    if (a > cursor) out.push({ text: pageText.slice(cursor, a), highlighted: false });
    out.push({ text: pageText.slice(a, b), highlighted: true });
    cursor = b;
  }
  if (cursor < pageText.length) out.push({ text: pageText.slice(cursor), highlighted: false });
  return out;
}

const KIND_LABELS: Record<string, string> = {
  published: "Published",
  revised: "Revised",
  effective_from: "Effective from",
  effective_to: "Effective to",
};

export function dateKindLabel(kind: string): string {
  return KIND_LABELS[kind] ?? kind.replaceAll("_", " ");
}

/** A date at its stated precision: a year-precision date shows the year only, never 1 January (ADR-0010). */
export function formatDate(value: string, precision: string): string {
  if (precision === "year") return value.slice(0, 4);
  if (precision === "month") return value.slice(0, 7);
  return value.slice(0, 10);
}

export function describeDates(dates: ReadonlyArray<Pick<DateFact, "kind" | "value" | "precision">>): string {
  if (dates.length === 0) return "dates unknown";
  return dates.map((d) => `${dateKindLabel(d.kind)} ${formatDate(d.value, d.precision)}`).join(" · ");
}

export function citationsFor(citations: ReadonlyArray<Citation>, evidenceId: string): Citation[] {
  return citations.filter((c) => c.evidence_id === evidenceId);
}
