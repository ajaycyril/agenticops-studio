import type { IncidentState, RuleResult } from "@/lib/types";

export function evaluateRules(incident: IncidentState): RuleResult {
  const rulesEvaluated = [
    {
      ruleId: "RULE-SMOKE-70",
      description: "IF smoke_ppm >= 70 THEN raise_alarm",
      passed: incident.smokePpm >= 70,
      inputUsed: ["smokePpm"]
    },
    {
      ruleId: "RULE-HEAT-55",
      description: "IF temperature_c >= 55 THEN mark_high_heat",
      passed: incident.temperatureC >= 55,
      inputUsed: ["temperatureC"]
    },
    {
      ruleId: "RULE-CRITICAL-100-50",
      description: "IF smoke_ppm >= 100 AND temperature_c >= 50 THEN mark_critical",
      passed: incident.smokePpm >= 100 && incident.temperatureC >= 50,
      inputUsed: ["smokePpm", "temperatureC"]
    }
  ];

  const critical = rulesEvaluated[2].passed;
  const highHeat = rulesEvaluated[1].passed;
  const smokeAlarm = rulesEvaluated[0].passed;
  const severity = critical ? "critical" : highHeat ? "high" : smokeAlarm ? "medium" : "low";

  return {
    triggered: smokeAlarm || highHeat || critical,
    severity,
    rulesEvaluated,
    action: critical ? "escalate_operator" : smokeAlarm || highHeat ? "raise_alarm" : "no_alarm",
    explanation:
      "Rule-based automation is fast and deterministic, but this mode only uses smoke and temperature. It cannot reason across camera evidence, occupancy, SOP, tool state, policy, or approval."
  };
}
