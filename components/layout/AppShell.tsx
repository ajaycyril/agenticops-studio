import { TopNav } from "@/components/layout/TopNav";
import { SideNav } from "@/components/layout/SideNav";
import { StatusBar } from "@/components/layout/StatusBar";

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen">
      <TopNav />
      <div className="mx-auto flex max-w-7xl">
        <SideNav />
        <main className="min-w-0 flex-1 px-4 py-6 md:px-6 lg:px-8">{children}</main>
      </div>
      <StatusBar />
    </div>
  );
}
