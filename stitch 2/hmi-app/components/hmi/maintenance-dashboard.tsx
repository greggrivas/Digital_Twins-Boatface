"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import ChatPanel from "@/components/hmi/chat-panel";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  fetchHmiSnapshot,
  fetchSurfaceData,
  fetchSurfaceMarker,
  recommendMaintenance
} from "@/lib/api";
import type { HmiSurfaceData } from "@/lib/types";

function ProjectedSurface({
  surface,
  marker
}: {
  surface: HmiSurfaceData;
  marker?: { turbine_decay: number; compressor_decay: number; fuel_flow: number };
}) {
  const { turbine_decay_values: xVals, compressor_decay_values: yVals } = surface.axes;
  const zGrid = surface.surface.fuel_flow_z;
  const tGrid = surface.surface.t48_color;
  const b = surface.plot_bounds;

  const width = 520;
  const height = 340;

  const norm = (v: number, min: number, max: number) => (max - min === 0 ? 0.5 : (v - min) / (max - min));

  const project = (x: number, y: number, z: number) => {
    const xn = surface.axis_meta.invert_x ? 1 - norm(x, b.x_min, b.x_max) : norm(x, b.x_min, b.x_max);
    const yn = surface.axis_meta.invert_y ? 1 - norm(y, b.y_min, b.y_max) : norm(y, b.y_min, b.y_max);
    const zn = norm(z, b.z_min, b.z_max);

    const px = (xn - yn) * 170 + width / 2;
    const py = (xn + yn) * 90 - zn * 160 + 35;
    return { px, py };
  };

  const tempVals = tGrid.flat();
  const tMin = Math.min(...tempVals);
  const tMax = Math.max(...tempVals);

  const colorAt = (temp: number) => {
    const n = norm(temp, tMin, tMax);
    const r = Math.round(60 + n * 180);
    const g = Math.round(120 - n * 60);
    const b = Math.round(220 - n * 170);
    return `rgb(${r}, ${g}, ${b})`;
  };

  const rowLines = yVals.map((y, yi) => {
    const points = xVals.map((x, xi) => {
      const p = project(x, y, zGrid[yi][xi]);
      return `${p.px},${p.py}`;
    });
    const tempAvg = tGrid[yi].reduce((a, v) => a + v, 0) / tGrid[yi].length;
    return <polyline key={`row-${yi}`} points={points.join(" ")} fill="none" stroke={colorAt(tempAvg)} strokeWidth="1.3" />;
  });

  const colLines = xVals.map((x, xi) => {
    const points = yVals.map((y, yi) => {
      const p = project(x, y, zGrid[yi][xi]);
      return `${p.px},${p.py}`;
    });
    const tempAvg = yVals.reduce((acc, _y, yi) => acc + tGrid[yi][xi], 0) / yVals.length;
    return <polyline key={`col-${xi}`} points={points.join(" ")} fill="none" stroke={colorAt(tempAvg)} strokeWidth="1.0" opacity="0.7" />;
  });

  const markerPoint =
    marker != null ? project(marker.turbine_decay, marker.compressor_decay, marker.fuel_flow) : null;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-[340px] w-full rounded-md border border-slate-800 bg-[#0d1320]">
      <g opacity="0.35">{colLines}</g>
      <g>{rowLines}</g>
      {markerPoint ? (
        <>
          <line
            x1={markerPoint.px}
            y1={markerPoint.py}
            x2={markerPoint.px}
            y2={height - 30}
            stroke="#7CFC00"
            strokeDasharray="4 4"
            opacity="0.7"
          />
          <circle cx={markerPoint.px} cy={markerPoint.py} r="7" fill="#7CFC00" stroke="#0a0f1a" strokeWidth="2" />
        </>
      ) : null}
      <text x="12" y="20" fill="#93c5fd" fontSize="12">X: Turbine Decay</text>
      <text x="12" y="36" fill="#86efac" fontSize="12">Y: Compressor Decay</text>
      <text x="12" y="52" fill="#fcd34d" fontSize="12">Z: Fuel Flow</text>
    </svg>
  );
}

function Metric({ label, value, unit }: { label: string; value: number | string; unit?: string }) {
  return (
    <div className="rounded border border-slate-800 bg-[#0f1522] p-3">
      <p className="text-xs uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-100">
        {value}
        {unit ? <span className="ml-1 text-sm text-slate-400">{unit}</span> : null}
      </p>
    </div>
  );
}

function SeverityBadge({ severity }: { severity: "healthy" | "warning" | "critical" }) {
  const cls =
    severity === "healthy"
      ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/40"
      : severity === "warning"
        ? "bg-amber-500/15 text-amber-400 border-amber-500/40"
        : "bg-red-500/15 text-red-400 border-red-500/40";
  return <span className={`rounded border px-2 py-1 text-xs font-semibold uppercase ${cls}`}>{severity}</span>;
}

export default function MaintenanceDashboard() {
  const queryClient = useQueryClient();

  const snapshot = useQuery({
    queryKey: ["hmi-snapshot"],
    queryFn: fetchHmiSnapshot
  });

  const speed = snapshot.data?.operating_state.ship_speed;

  const surface = useQuery({
    queryKey: ["hmi-surface", speed],
    queryFn: () => fetchSurfaceData(speed ?? 15),
    enabled: typeof speed === "number"
  });

  const marker = useQuery({
    queryKey: ["hmi-marker", speed, snapshot.data?.predictions.compressor_decay_pred, snapshot.data?.predictions.turbine_decay_pred],
    queryFn: () =>
      fetchSurfaceMarker({
        speed: speed ?? 15,
        compressor_decay_pred: snapshot.data?.predictions.compressor_decay_pred ?? 0.97,
        turbine_decay_pred: snapshot.data?.predictions.turbine_decay_pred ?? 0.99
      }),
    enabled: !!snapshot.data && typeof speed === "number"
  });

  const recommendation = useQuery({
    queryKey: [
      "hmi-recommend",
      snapshot.data?.predictions.compressor_decay_pred,
      snapshot.data?.predictions.turbine_decay_pred
    ],
    queryFn: () =>
      recommendMaintenance({
        compressor_decay: snapshot.data!.predictions.compressor_decay_pred,
        turbine_decay: snapshot.data!.predictions.turbine_decay_pred
      }),
    enabled: !!snapshot.data
  });

  const loading = snapshot.isLoading || surface.isLoading || marker.isLoading;

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-9 space-y-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Gas Turbine Digital Twin Dashboard</CardTitle>
              <p className="text-xs text-slate-400">
                Snapshot #{snapshot.data?.snapshot_id ?? "..."} • Source: {snapshot.data?.source ?? "loading"}
              </p>
            </div>
            <Button
              variant="outline"
              onClick={() => {
                queryClient.invalidateQueries({ queryKey: ["hmi-snapshot"] });
              }}
            >
              New Snapshot
            </Button>
          </CardHeader>
        </Card>

        <div className="grid grid-cols-3 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Operating State</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-2">
              <Metric label="Ship Speed" value={snapshot.data?.operating_state.ship_speed ?? "-"} unit="knots" />
              <Metric label="Lever Pos" value={snapshot.data?.operating_state.lever_pos ?? "-"} />
              <Metric label="GT Torque" value={snapshot.data?.operating_state.gt_torque ?? "-"} unit="kN·m" />
              <Metric label="GT RPM" value={snapshot.data?.operating_state.gt_rpm ?? "-"} unit="rpm" />
              <Metric label="GG RPM" value={snapshot.data?.operating_state.gg_rpm ?? "-"} unit="rpm" />
              <Metric label="Fuel Flow" value={snapshot.data?.operating_state.fuel_flow ?? "-"} unit="kg/s" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Temperature Panel</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Metric label="T48" value={snapshot.data?.temperature_state.t48 ?? "-"} unit="°C" />
              <Metric label="T2" value={snapshot.data?.temperature_state.t2 ?? "-"} unit="°C" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Pressure Panel</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Metric label="P48" value={snapshot.data?.pressure_state.p48 ?? "-"} unit="bar" />
              <Metric label="P2" value={snapshot.data?.pressure_state.p2 ?? "-"} unit="bar" />
              <Metric label="Pexh" value={snapshot.data?.pressure_state.pexh ?? "-"} unit="bar" />
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Health Predictions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="rounded border border-slate-800 bg-[#0f1522] p-3">
                <div className="mb-2 flex items-center justify-between">
                  <p className="text-sm font-semibold">Compressor Decay (Predicted)</p>
                  <SeverityBadge severity={snapshot.data?.predictions.severity ?? "warning"} />
                </div>
                <p className="text-2xl font-bold">{snapshot.data?.predictions.compressor_decay_pred?.toFixed(5) ?? "-"}</p>
              </div>
              <div className="rounded border border-slate-800 bg-[#0f1522] p-3">
                <p className="text-sm font-semibold">Turbine Decay (Predicted)</p>
                <p className="text-2xl font-bold">{snapshot.data?.predictions.turbine_decay_pred?.toFixed(5) ?? "-"}</p>
              </div>
              <div className="rounded border border-slate-800 bg-[#0f1522] p-3 text-sm">
                <p className="mb-1 font-semibold text-slate-200">Recommendation</p>
                <p>Action: {recommendation.data?.action ?? "-"}</p>
                <p>Priority: {recommendation.data?.priority ?? "-"}</p>
                <p>Window: {recommendation.data?.maintenance_window ?? "-"}</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>3D Degradation Surface</CardTitle>
              <p className="text-xs text-slate-400">
                X = Turbine Decay, Y = Compressor Decay, Z = Fuel Flow, color = T48
              </p>
            </CardHeader>
            <CardContent>
              {loading || !surface.data ? (
                <div className="flex h-[340px] items-center justify-center rounded border border-slate-800 bg-[#0d1320] text-slate-400">
                  Loading 3D surface...
                </div>
              ) : (
                <ProjectedSurface surface={surface.data} marker={marker.data?.marker} />
              )}
              <div className="mt-2 text-xs text-slate-400">
                Marker: compressor {marker.data?.marker.compressor_decay?.toFixed(5) ?? "-"}, turbine{" "}
                {marker.data?.marker.turbine_decay?.toFixed(5) ?? "-"}, fuel {marker.data?.marker.fuel_flow?.toFixed(3) ?? "-"}
                kg/s, T48 {marker.data?.marker.t48?.toFixed(1) ?? "-"} °C
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="col-span-3">
        <ChatPanel />
      </div>
    </div>
  );
}
