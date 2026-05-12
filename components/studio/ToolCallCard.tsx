import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ToolDefinition } from "@/lib/types";

export function ToolCallCard({ tool }: { tool: ToolDefinition }) {
  return (
    <Card className="shadow-none">
      <CardHeader className="p-4 pb-2">
        <div className="flex items-start justify-between gap-3">
          <CardTitle className="text-sm">{tool.name}</CardTitle>
          <Badge>{tool.executionMode}</Badge>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0 text-xs leading-5 text-slate-400">
        <p>{tool.description}</p>
        <p className="mt-2">Policy: {tool.requiresPolicyCheck ? "required" : "not required"} · Approval: {tool.requiresHumanApproval ? "required" : "not required"}</p>
      </CardContent>
    </Card>
  );
}
