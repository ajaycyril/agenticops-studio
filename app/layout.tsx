import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgenticOps Studio",
  description: "Interactive Physical AI, Edge AI, and governed enterprise agentic AI control tower."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
