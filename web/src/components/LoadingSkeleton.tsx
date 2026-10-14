"use client";

import React from "react";

export function LoadingSkeleton() {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100 shadow-xl space-y-6 animate-pulse">
      <div className="flex justify-between items-center pb-4 border-b border-slate-800">
        <div className="space-y-2">
          <div className="h-6 w-48 bg-slate-800 rounded" />
          <div className="h-4 w-32 bg-slate-800/60 rounded" />
        </div>
        <div className="h-10 w-24 bg-slate-800 rounded-full" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
        <div className="space-y-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="space-y-2">
              <div className="flex justify-between">
                <div className="h-4 w-20 bg-slate-800 rounded" />
                <div className="h-4 w-10 bg-slate-800 rounded" />
              </div>
              <div className="h-2 w-full bg-slate-800 rounded-full" />
            </div>
          ))}
        </div>
        <div className="flex justify-center items-center">
          <div className="w-40 h-40 rounded-full border-4 border-dashed border-slate-800 animate-spin" />
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-slate-800">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-16 bg-slate-800/40 rounded-lg" />
        ))}
      </div>
    </div>
  );
}
