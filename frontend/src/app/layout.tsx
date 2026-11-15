import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Mezvuat AI",
  description: "Evidence-grounded research assistant: every claim cites an exact quote.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <Link href="/" className="brand">
            Mezvuat AI
          </Link>
          <span className="tagline">Answers that cite exact quotes</span>
        </header>
        <main className="main">{children}</main>
        <footer className="footer">
          AGPL-3.0 ·{" "}
          <a href="https://github.com/adar-shh04/Mezvuat-AI" rel="noreferrer">
            source
          </a>
        </footer>
      </body>
    </html>
  );
}
