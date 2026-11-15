import { ResearchChat } from "@/components/ResearchChat";

export default async function ResearchPage({ params }: { params: Promise<{ workspaceId: string }> }) {
  const { workspaceId } = await params;
  return <ResearchChat workspaceId={workspaceId} />;
}
