import { isRoboflowConfigured } from "@/lib/env";
import type { ToolDefinition } from "@/lib/types";

export function getToolRegistry(): ToolDefinition[] {
  return [
    {
      name: "runVisionModel",
      description: "Run smoke/fire computer vision inference through Roboflow hosted API or sample fallback.",
      riskClass: "medium",
      executionMode: isRoboflowConfigured() ? "real" : "sandbox",
      requiresPolicyCheck: false,
      requiresHumanApproval: false,
      inputSchemaName: "VisionRequest",
      outputSchemaName: "VisionResult"
    },
    {
      name: "runRiskModel",
      description: "Run local/browser ML risk prediction against current incident features.",
      riskClass: "medium",
      executionMode: "real",
      requiresPolicyCheck: false,
      requiresHumanApproval: false,
      inputSchemaName: "IncidentState",
      outputSchemaName: "MLResult"
    },
    {
      name: "retrieveSOP",
      description: "Retrieve local SOP markdown snippets relevant to fire response.",
      riskClass: "low",
      executionMode: "real",
      requiresPolicyCheck: false,
      requiresHumanApproval: false,
      inputSchemaName: "SOPQuery",
      outputSchemaName: "SOPSnippet[]"
    },
    {
      name: "checkPolicy",
      description: "Evaluate TypeScript policy guardrails before any proposed action.",
      riskClass: "high",
      executionMode: "real",
      requiresPolicyCheck: false,
      requiresHumanApproval: false,
      inputSchemaName: "ActionProposal",
      outputSchemaName: "PolicyDecision"
    },
    {
      name: "requestHumanApproval",
      description: "Ask a demo operator to approve or reject approval-gated actions.",
      riskClass: "high",
      executionMode: "real",
      requiresPolicyCheck: true,
      requiresHumanApproval: false,
      inputSchemaName: "ApprovalRequest",
      outputSchemaName: "ApprovalDecision"
    },
    {
      name: "dispatchDrone",
      description: "Sandbox drone dispatch command. This is never a live drone SDK call in the public demo.",
      riskClass: "high",
      executionMode: "sandbox",
      requiresPolicyCheck: true,
      requiresHumanApproval: true,
      inputSchemaName: "DroneDispatchRequest",
      outputSchemaName: "SandboxToolResult"
    },
    {
      name: "unlockGate",
      description: "Sandbox access-control command. This never unlocks a real gate.",
      riskClass: "critical",
      executionMode: "sandbox",
      requiresPolicyCheck: true,
      requiresHumanApproval: true,
      inputSchemaName: "GateUnlockRequest",
      outputSchemaName: "SandboxToolResult"
    },
    {
      name: "notifyOperator",
      description: "Sandbox operator notification.",
      riskClass: "medium",
      executionMode: "sandbox",
      requiresPolicyCheck: true,
      requiresHumanApproval: false,
      inputSchemaName: "OperatorNotification",
      outputSchemaName: "SandboxToolResult"
    },
    {
      name: "notifyAuthority",
      description: "Sandbox authority notification. No emergency service is contacted.",
      riskClass: "critical",
      executionMode: "sandbox",
      requiresPolicyCheck: true,
      requiresHumanApproval: true,
      inputSchemaName: "AuthorityNotification",
      outputSchemaName: "SandboxToolResult"
    },
    {
      name: "generateIncidentReport",
      description: "Generate an executive or technical incident report using OpenAI or deterministic fallback.",
      riskClass: "low",
      executionMode: "real",
      requiresPolicyCheck: false,
      requiresHumanApproval: false,
      inputSchemaName: "DecisionRecord",
      outputSchemaName: "IncidentReport"
    },
    {
      name: "writeDecisionRecord",
      description: "Write a local/session decision record for auditability.",
      riskClass: "low",
      executionMode: "real",
      requiresPolicyCheck: false,
      requiresHumanApproval: false,
      inputSchemaName: "DecisionRecord",
      outputSchemaName: "DecisionRecord"
    }
  ];
}
