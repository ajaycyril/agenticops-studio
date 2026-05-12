import type { ActionName, PolicyDecision, ToolDefinition } from "@/lib/types";

export function executeSandboxTool(action: ActionName, tool: ToolDefinition | undefined, policy: PolicyDecision, approved: boolean) {
  if (!tool) {
    return { status: "blocked" as const, result: "Tool is not registered." };
  }
  if (tool.executionMode === "disabled" || policy.blocked) {
    return { status: "blocked" as const, result: policy.reason };
  }
  if (policy.requiresHumanApproval && !approved) {
    return { status: "skipped" as const, result: "Human approval is required before execution." };
  }
  return {
    status: "success" as const,
    result: `${action} executed in ${tool.executionMode} mode. Physical integrations are sandboxed in the public demo.`
  };
}
