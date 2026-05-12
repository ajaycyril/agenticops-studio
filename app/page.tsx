import Link from "next/link";
import { ArrowRight, BrainCircuit, Cpu, ShieldCheck } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ENTERPRISE_LINE, PRODUCT_THESIS } from "@/lib/constants";

export default function HomePage() {
  return (
    <AppShell>
      <section className="glass-grid overflow-hidden rounded-lg border border-white/10 bg-slate-950/40 p-6 md:p-12">
        <div className="max-w-5xl">
          <Badge className="border-cyan-300/30 bg-cyan-300/10 text-cyan-100">Physical AI fire-response reference implementation</Badge>
          <h1 className="mt-6 max-w-4xl text-4xl font-semibold tracking-tight text-white md:text-6xl">AgenticOps Studio</h1>
          <p className="mt-5 max-w-4xl text-lg leading-8 text-slate-300">
            A live control-tower demo showing how a physical fire incident moves from sensors and vision to ML risk, governed agentic
            planning, policy approval, traceability, and a decision record.
          </p>
          <p className="mt-4 max-w-4xl text-base leading-7 text-cyan-100">{ENTERPRISE_LINE}</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button asChild>
              <Link href="/studio">
                Open Live Studio <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button asChild variant="secondary">
              <Link href="/architecture">View Architecture</Link>
            </Button>
          </div>
        </div>
      </section>

      <section className="mt-6 grid gap-4 md:grid-cols-3">
        <Card className="shadow-none">
          <CardHeader>
            <Cpu className="h-5 w-5 text-cyan-200" />
            <CardTitle>Rules Detect</CardTitle>
            <CardDescription>Fixed smoke and heat thresholds raise alarms quickly, but do not reason across context.</CardDescription>
          </CardHeader>
        </Card>
        <Card className="shadow-none">
          <CardHeader>
            <BrainCircuit className="h-5 w-5 text-amber-200" />
            <CardTitle>ML Predicts</CardTitle>
            <CardDescription>TensorFlow.js estimates fire probability from fused sensor, camera, and context features.</CardDescription>
          </CardHeader>
        </Card>
        <Card className="shadow-none">
          <CardHeader>
            <ShieldCheck className="h-5 w-5 text-emerald-200" />
            <CardTitle>Agentic AI Governs</CardTitle>
            <CardDescription>OpenAI planning, policy, approvals, sandbox tools, traces, and records coordinate safe execution.</CardDescription>
          </CardHeader>
        </Card>
      </section>

      <p className="mt-6 max-w-4xl text-sm leading-6 text-slate-400">{PRODUCT_THESIS}</p>
    </AppShell>
  );
}
