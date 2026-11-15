import { DocumentDetail, type DocumentFocus } from "@/components/DocumentDetail";

type SearchParams = Record<string, string | string[] | undefined>;

function int(value: string | string[] | undefined): number | null {
  if (typeof value !== "string" || !/^\d+$/.test(value)) return null;
  return Number(value);
}

export default async function DocumentPage({
  params,
  searchParams,
}: {
  params: Promise<{ workspaceId: string; documentId: string }>;
  searchParams: Promise<SearchParams>;
}) {
  const { workspaceId, documentId } = await params;
  const query = await searchParams;
  const focus: DocumentFocus = {
    versionId: typeof query.version === "string" ? query.version : null,
    page: int(query.page),
    start: int(query.start),
    end: int(query.end),
  };
  return <DocumentDetail workspaceId={workspaceId} documentId={documentId} focus={focus} />;
}
