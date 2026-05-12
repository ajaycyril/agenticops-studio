import Link from "next/link";
import { Activity, ExternalLink, ShieldCheck } from "lucide-react";
import { APP_NAME } from "@/lib/constants";
import { Button } from "@/components/ui/button";

export function TopNav() {
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-slate-950/75 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3">
        <Link href="/" className="flex items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-md border border-cyan-300/30 bg-cyan-300/10">
            <Activity className="h-5 w-5 text-cyan-200" />
          </span>
          <span>
            <span className="block text-sm font-semibold tracking-wide text-slate-50">{APP_NAME}</span>
            <span className="block text-xs text-slate-400">Physical AI command tower</span>
          </span>
        </Link>
        <nav className="hidden items-center gap-1 md:flex">
          <Button asChild variant="ghost" size="sm">
            <Link href="/studio">Studio</Link>
          </Button>
          <Button asChild variant="ghost" size="sm">
            <Link href="/architecture">Architecture</Link>
          </Button>
          <Button asChild variant="ghost" size="sm">
            <Link href="/enterprise">Enterprise</Link>
          </Button>
        </nav>
        <div className="flex items-center gap-2">
          <span className="hidden items-center gap-2 rounded-md border border-emerald-300/20 bg-emerald-300/10 px-3 py-2 text-xs text-emerald-100 sm:flex">
            <ShieldCheck className="h-4 w-4" /> Sandbox actions only
          </span>
          <Button asChild variant="secondary" size="sm">
            <a href="https://github.com/" target="_blank" rel="noreferrer">
              <ExternalLink className="h-4 w-4" /> Repo
            </a>
          </Button>
        </div>
      </div>
    </header>
  );
}
