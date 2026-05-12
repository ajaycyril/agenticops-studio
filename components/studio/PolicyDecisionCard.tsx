import { ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { PolicyDecision } from "@/lib/types";

export function PolicyDecisionCard({ decision }: { decision: PolicyDecision }) {
  return (
    <Card className="shadow-none">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
              <ShieldAlert className="h-4 w-4 text-cyan-200" /> {decision.action}
            </div>
            <p className="mt-2 text-xs leading-5 text-slate-400">{decision.reason}</p>
          </div>
          <Badge className={decision.blocked ? "border-red-400/30 bg-red-500/10 text-red-100" : "border-emerald-300/30 bg-emerald-300/10 text-emerald-100"}>
            {decision.blocked ? "blocked" : decision.requiresHumanApproval ? "approval" : "allowed"}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}
