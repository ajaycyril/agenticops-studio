import Link from "next/link";
import { Boxes, Building2, Network } from "lucide-react";

const items = [
  { href: "/studio", label: "Studio", icon: Boxes },
  { href: "/architecture", label: "Architecture", icon: Network },
  { href: "/enterprise", label: "Enterprise", icon: Building2 }
];

export function SideNav() {
  return (
    <aside className="hidden w-60 shrink-0 border-r border-white/10 bg-black/15 p-4 lg:block">
      <div className="space-y-2">
        {items.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className="flex items-center gap-3 rounded-md px-3 py-2 text-sm text-slate-300 hover:bg-white/8 hover:text-white"
          >
            <item.icon className="h-4 w-4 text-cyan-200" />
            {item.label}
          </Link>
        ))}
      </div>
    </aside>
  );
}
