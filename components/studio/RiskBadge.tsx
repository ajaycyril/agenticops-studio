import { Badge } from "@/components/ui/badge";
import type { RiskLevel } from "@/lib/types";

const riskClass: Record<RiskLevel, string> = {
  low: "border-emerald-300/30 bg-emerald-300/10 text-emerald-100",
  medium: "border-amber-300/30 bg-amber-300/10 text-amber-100",
  high: "border-orange-400/30 bg-orange-400/10 text-orange-100",
  critical: "border-red-400/40 bg-red-500/15 text-red-100"
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  return <Badge className={riskClass[level]}>{level.toUpperCase()}</Badge>;
}
