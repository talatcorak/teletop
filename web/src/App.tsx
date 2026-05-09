import { useEffect, useState } from "react";

type HealthState =
  | { kind: "loading" }
  | { kind: "ok"; service: string }
  | { kind: "error"; message: string };

export default function App() {
  const [health, setHealth] = useState<HealthState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    fetch("/api/health")
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return (await r.json()) as { status: string; service: string };
      })
      .then((data) => {
        if (cancelled) return;
        if (data.status !== "ok") throw new Error(`status=${data.status}`);
        setHealth({ kind: "ok", service: data.service });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setHealth({ kind: "error", message: err instanceof Error ? err.message : String(err) });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-8">
      <h1 className="text-3xl font-semibold tracking-tight">teletop</h1>
      <p className="mt-2 text-neutral-400 text-sm">
        Remote ESP32 flash &amp; monitor
      </p>
      <div className="mt-6 text-sm">
        {health.kind === "loading" && <span className="text-neutral-400">checking server…</span>}
        {health.kind === "ok" && (
          <span className="text-emerald-400">server ok — {health.service}</span>
        )}
        {health.kind === "error" && (
          <span className="text-red-400">server unreachable: {health.message}</span>
        )}
      </div>
    </div>
  );
}
