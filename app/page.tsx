import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import { ArrowRight, BrainCircuit, Cpu, FileJson, ShieldCheck } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ENTERPRISE_LINE, PRODUCT_THESIS } from "@/lib/constants";

export default function HomePage() {
  const cards: Array<[string, string, LucideIcon]> = [
    ["Rule-based automation detects", "Fixed if/else logic for smoke and heat thresholds.", Cpu],
    ["ML predicts", "Browser-trained TensorFlow.js model estimates fire probability.", BrainCircuit],
    ["Agentic AI coordinates", "LLM proposes evidence-backed actions through a tool registry.", ArrowRight],
    ["Enterprise AI governs", "Policy, approval, tracing, and records control execution.", ShieldCheck]
  ];

  return (
    <AppShell>
      <section className="glass-grid overflow-hidden rounded-lg border border-white/10 bg-slate-950/40 p-6 md:p-10">
        <div className="max-w-4xl">
          <Badge className="border-cyan-300/30 bg-cyan-300/10 text-cyan-100">Physical AI fire-response reference implementation</Badge>
          <h1 className="mt-6 text-4xl font-semibold tracking-tight text-white md:text-6xl">AgenticOps Studio</h1>
          <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-300">
            AgenticOps Studio is an interactive full-stack explainer for Physical AI and enterprise agentic AI. It demonstrates how
            rule-based automation, ML-based prediction, and governed agentic AI behave differently in a fire-response workflow.
          </p>
          <p className="mt-4 max-w-3xl text-base leading-7 text-cyan-100">{ENTERPRISE_LINE}</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button asChild>
              <Link href="/studio">
                Open Studio <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button asChild variant="secondary">
              <Link href="/architecture">View Architecture</Link>
            </Button>
          </div>
        </div>
      </section>

      <section className="mt-6 grid gap-4 md:grid-cols-4">
        {cards.map(([title, body, Icon]) => (
          <Card key={String(title)}>
            <CardHeader>
              <Icon className="h-5 w-5 text-cyan-200" />
              <CardTitle>{title}</CardTitle>
              <CardDescription>{body}</CardDescription>
            </CardHeader>
          </Card>
        ))}
      </section>

      <section className="mt-6 grid gap-4 lg:grid-cols-[1.1fr_.9fr]">
        <Card>
          <CardHeader>
            <CardTitle>{PRODUCT_THESIS}</CardTitle>
            <CardDescription>
              Most agent demos stop at tool calling. AgenticOps Studio shows the missing enterprise layers: edge perception, risk models,
              tools, policy, human approval, observability, and auditable execution.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 text-sm text-slate-300 sm:grid-cols-2">
            {["Physical sensors", "Edge vision", "Risk model", "SOP retrieval", "Policy engine", "Human approvals", "Sandbox tools", "Decision records"].map(
              (item) => (
                <div key={item} className="rounded-md border border-white/10 bg-black/20 p-3">
                  {item}
                </div>
              )
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <FileJson className="h-5 w-5 text-amber-200" />
            <CardTitle>Reference implementation, not a fake animation</CardTitle>
            <CardDescription>
              API routes validate inputs, TensorFlow.js trains in the browser, Roboflow and OpenAI run through server routes when keys are
              configured, and fallback modes keep the public demo deployable.
            </CardDescription>
          </CardHeader>
        </Card>
      </section>
    </AppShell>
  );
}
