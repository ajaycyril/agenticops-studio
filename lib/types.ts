export type OccupancyStatus = "none" | "detected" | "unknown";
export type RiskLevel = "low" | "medium" | "high" | "critical";
export type ActionName =
  | "request_camera_recheck"
  | "dispatch_drone"
  | "unlock_gate"
  | "notify_operator"
  | "notify_authority"
  | "generate_report"
  | "write_decision_record";

export type VisionDetection = {
  className: "smoke" | "fire" | "person" | "unknown";
  confidence: number;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
};

export type IncidentState = {
  incidentId: string;
  siteId: string;
  scenarioName: string;
  smokePpm: number;
  temperatureC: number;
  cameraSmokeConfidence: number;
  cameraFireConfidence: number;
  occupancyStatus: OccupancyStatus;
  droneAvailable: boolean;
  gateLocked: boolean;
  sensorHealth: number;
  historicalFalseAlarmRate: number;
  windSpeedKmh: number;
  imageUrl?: string;
  visionProvider: "roboflow" | "sample" | "manual";
  visionDetections: VisionDetection[];
  humanApproval: {
    unlockGate: boolean;
    dispatchDrone: boolean;
    notifyAuthority: boolean;
  };
};

export type RuleResult = {
  triggered: boolean;
  severity: RiskLevel;
  rulesEvaluated: {
    ruleId: string;
    description: string;
    passed: boolean;
    inputUsed: string[];
  }[];
  action: "no_alarm" | "raise_alarm" | "escalate_operator";
  explanation: string;
};

export type MLResult = {
  fireProbability: number;
  riskLevel: RiskLevel;
  modelVersion: string;
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    falsePositiveRate: number;
  };
  featureImportance: {
    feature: string;
    importance: number;
  }[];
  explanation: string;
};

export type VisionResult = {
  provider: "roboflow" | "sample";
  modelId: string;
  detections: VisionDetection[];
  maxSmokeConfidence: number;
  maxFireConfidence: number;
  latencyMs: number;
  setupRequired?: boolean;
  message?: string;
};

export type AgentRiskAssessment = {
  level: RiskLevel;
  reason: string;
  evidence: string[];
  uncertainty: string[];
};

export type ActionProposal = {
  action: ActionName;
  riskClass: RiskLevel;
  reason: string;
  evidenceUsed: string[];
  confidence: number;
  requiresPolicyCheck: boolean;
  requiresHumanApproval: boolean;
};

export type AgenticResult = {
  incidentSummary: string;
  riskAssessment: AgentRiskAssessment;
  proposedActions: ActionProposal[];
  questionsForHuman: string[];
  sopReferences: string[];
  explanation: string;
};

export type ToolDefinition = {
  name: string;
  description: string;
  riskClass: RiskLevel;
  executionMode: "real" | "sandbox" | "disabled";
  requiresPolicyCheck: boolean;
  requiresHumanApproval: boolean;
  inputSchemaName: string;
  outputSchemaName: string;
};

export type PolicyDecision = {
  action: ActionName;
  allowed: boolean;
  requiresHumanApproval: boolean;
  blocked: boolean;
  reason: string;
  policyIds: string[];
};

export type TraceEvent = {
  id: string;
  timestamp: string;
  type:
    | "incident_created"
    | "vision_model_called"
    | "rule_engine_evaluated"
    | "ml_model_training_started"
    | "ml_model_training_completed"
    | "ml_model_predicted"
    | "agent_called"
    | "agent_output_validated"
    | "tool_proposed"
    | "schema_validated"
    | "policy_checked"
    | "human_approval_requested"
    | "human_approval_resolved"
    | "action_executed"
    | "action_blocked"
    | "decision_record_written"
    | "error";
  actor: "system" | "vision_model" | "ml_model" | "llm_agent" | "tool" | "policy" | "human";
  input?: unknown;
  output?: unknown;
  latencyMs?: number;
  status: "success" | "blocked" | "pending" | "failed";
  explanation?: string;
};

export type DecisionRecord = {
  incidentId: string;
  runId: string;
  createdAt: string;
  scenarioName: string;
  inputs: IncidentState;
  ruleResult: RuleResult;
  mlResult: MLResult;
  visionResult: VisionResult;
  agenticResult: AgenticResult;
  policyDecisions: PolicyDecision[];
  actionsProposed: ActionProposal[];
  actionsApproved: {
    action: ActionName;
    approvedBy: "demo-operator";
    approvedAt: string;
  }[];
  actionsBlocked: {
    action: ActionName;
    reason: string;
    policyIds: string[];
  }[];
  actionsExecuted: {
    action: ActionName;
    executionMode: "real" | "sandbox" | "disabled";
    status: "success" | "blocked" | "skipped";
    result: string;
  }[];
  governance: {
    policyVersion: string;
    activeModelVersion: string;
    visionProvider: string;
    llmModel: string;
    agentVersion: string;
  };
  trace: TraceEvent[];
};

