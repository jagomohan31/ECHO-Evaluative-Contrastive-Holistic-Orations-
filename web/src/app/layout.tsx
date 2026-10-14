import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SpeechLens — Contrastive Speech Analytics",
  description: "Evaluate, compare, and polish oratory performance against reference baselines.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 antialiased min-h-screen">
        <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center font-bold text-white shadow-lg shadow-blue-500/25">
                SL
              </div>
              <div>
                <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                  SpeechLens
                </span>
                <span className="text-[10px] font-mono uppercase tracking-widest text-blue-400 ml-2 px-1.5 py-0.5 rounded bg-blue-950/60 border border-blue-800/40">
                  ECHO 2026
                </span>
              </div>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <span>Evaluative Contrastive Holistic Orations</span>
            </div>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
