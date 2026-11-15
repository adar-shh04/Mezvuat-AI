import { WorkspaceDocuments } from "@/components/WorkspaceDocuments";

export default async function WorkspacePage({ params }: { params: Promise<{ workspaceId: string }> }) {
  const { workspaceId } = await params;
  return <WorkspaceDocuments workspaceId={workspaceId} />;
}
