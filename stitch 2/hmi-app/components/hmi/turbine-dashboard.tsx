"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { StatusBadge } from "@/components/ui/status-badge";
import M501J360View from "@/components/hmi/m501j-360-view";
import {
  fetchHmiSnapshot,
  fetchSurfaceData,
  fetchSurfaceMarker,
  recommendMaintenance,
} from "@/lib/api";
import type { HmiSurfaceData } from "@/lib/types";
import {
  Gauge,
  Activity,
  HeartPulse,
  Wrench,
  AlertTriangle,
  RefreshCw,
  Flame,
  Ship,
  Anchor,
  SlidersHorizontal,
} from "lucide-react";

// Grouped Sensor Panel - organized by category
function SensorPanel({
  data,
}: {
  data?: {
    t1?: number;
    t2?: number;
    t48?: number;
    p1?: number;
    p2?: number;
    p48?: number;
    pexh?: number;
    fuelFlow?: number;
    gtRpm?: number;
    ggRpm?: number;
    gtTorque?: number;
    tic?: number;
    shipSpeed?: number;
    leverPos?: number;
    propTorqueS?: number;
    propTorqueP?: number;
  };
}) {
  const SensorValue = ({
    label,
    value,
    unit,
    color = "text-white",
  }: {
    label: string;
    value: string | number | undefined;
    unit: string;
    color?: string;
  }) => (
    <div className="flex flex-col items-center">
      <span className="text-[9px] text-slate-500 uppercase tracking-wider">{label}</span>
      <span className={`font-mono text-base font-bold ${color}`}>
        {value ?? "-"}
      </span>
      <span className="text-[9px] text-slate-600">{unit}</span>
    </div>
  );

  const GroupCard = ({
    title,
    icon,
    iconColor,
    children,
  }: {
    title: string;
    icon: React.ReactNode;
    iconColor: string;
    children: React.ReactNode;
  }) => (
    <div className="flex-1 min-w-[140px] bg-surface-highlight/30 rounded-lg p-3 border border-slate-800/50">
      <div className={`flex items-center gap-1.5 mb-2 ${iconColor}`}>
        {icon}
        <span className="text-[10px] font-medium uppercase tracking-wide">{title}</span>
      </div>
      <div className="flex justify-around gap-2">
        {children}
      </div>
    </div>
  );

  return (
    <div className="flex flex-wrap gap-2">
      {/* Temperatures */}
      <GroupCard
        title="Temperatures"
        icon={<Flame className="h-3.5 w-3.5" />}
        iconColor="text-orange-400"
      >
        <SensorValue label="T1 Inlet" value={data?.t1?.toFixed(0)} unit="°C" color="text-blue-400" />
        <SensorValue label="T2 Comp" value={data?.t2?.toFixed(0)} unit="°C" color="text-cyan-400" />
        <SensorValue label="T48 Exit" value={data?.t48?.toFixed(0)} unit="°C" color="text-orange-500" />
      </GroupCard>

      {/* Pressures */}
      <GroupCard
        title="Pressures"
        icon={<Activity className="h-3.5 w-3.5" />}
        iconColor="text-blue-400"
      >
        <SensorValue label="P1" value={data?.p1?.toFixed(2)} unit="bar" color="text-blue-300" />
        <SensorValue label="P2" value={data?.p2?.toFixed(1)} unit="bar" color="text-slate-300" />
        <SensorValue label="P48" value={data?.p48?.toFixed(2)} unit="bar" color="text-slate-300" />
        <SensorValue label="Pexh" value={data?.pexh?.toFixed(2)} unit="bar" color="text-amber-400" />
      </GroupCard>

      {/* RPM & Torque */}
      <GroupCard
        title="RPM & Torque"
        icon={<Gauge className="h-3.5 w-3.5" />}
        iconColor="text-emerald-400"
      >
        <SensorValue label="GT RPM" value={data?.gtRpm?.toFixed(0)} unit="rpm" color="text-emerald-400" />
        <SensorValue label="GG RPM" value={data?.ggRpm?.toFixed(0)} unit="rpm" color="text-emerald-400" />
        <SensorValue label="Torque" value={data?.gtTorque?.toFixed(0)} unit="kN·m" color="text-yellow-400" />
      </GroupCard>

      {/* Fuel */}
      <GroupCard
        title="Fuel Flow"
        icon={<Flame className="h-3.5 w-3.5" />}
        iconColor="text-orange-500"
      >
        <SensorValue label="Rate" value={data?.fuelFlow?.toFixed(3)} unit="kg/s" color="text-orange-400" />
      </GroupCard>
    </div>
  );
}

// 3D projected surface with hover tooltips and state marker
function ProjectedSurface({
  surface,
  marker,
  shipSpeed,
}: {
  surface: HmiSurfaceData;
  marker?: {
    turbine_decay: number;
    compressor_decay: number;
    fuel_flow: number;
  };
  shipSpeed?: number;
}) {
  const [hoverInfo, setHoverInfo] = useState<{
    turbine: number;
    compressor: number;
    fuel: number;
    temp: number;
  } | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);

  const { turbine_decay_values: xVals, compressor_decay_values: yVals } =
    surface.axes;
  const zGrid = surface.surface.fuel_flow_z;
  const tGrid = surface.surface.t48_color;
  const b = surface.plot_bounds;

  const width = 760;
  const height = 760;

  const norm = (v: number, min: number, max: number) =>
    max - min === 0 ? 0.5 : (v - min) / (max - min);

  // Projection tuned so the surface reads left/back high -> right/front low.
  const centerX = 300;
  const centerY = 560;
  const scaleX = 190;
  const scaleY = 120;
  const scaleZ = 260;

  const project = (x: number, y: number, z: number) => {
    const xn = norm(x, b.x_min, b.x_max);
    const yn = 1 - norm(y, b.y_min, b.y_max);
    const zn = norm(z, b.z_min, b.z_max);

    const px = centerX + (xn - yn) * scaleX;
    const py = centerY - (xn + yn) * scaleY - zn * scaleZ;
    return { px, py };
  };

  const tempVals = tGrid.flat();
  const tMin = Math.min(...tempVals);
  const tMax = Math.max(...tempVals);

  // Cool to warm gradient: blue -> cyan -> cream -> orange -> red
  const colorAt = (temp: number) => {
    const n = norm(temp, tMin, tMax);
    if (n < 0.25) {
      const t = n / 0.25;
      return `rgb(${Math.round(60 + 40 * t)}, ${Math.round(95 + 95 * t)}, ${Math.round(190 + 35 * t)})`;
    }
    if (n < 0.5) {
      const t = (n - 0.25) / 0.25;
      return `rgb(${Math.round(100 + 110 * t)}, ${Math.round(190 + 35 * t)}, ${Math.round(225 - 25 * t)})`;
    }
    if (n < 0.75) {
      const t = (n - 0.5) / 0.25;
      return `rgb(${Math.round(210 + 45 * t)}, ${Math.round(225 - 65 * t)}, ${Math.round(200 - 90 * t)})`;
    } else {
      const t = (n - 0.75) / 0.25;
      return `rgb(${Math.round(255 - 20 * t)}, ${Math.round(160 - 95 * t)}, ${Math.round(110 - 85 * t)})`;
    }
  };

  const quads: React.ReactNode[] = [];
  for (let yi = 0; yi < yVals.length - 1; yi++) {
    for (let xi = 0; xi < xVals.length - 1; xi++) {
      const p1 = project(xVals[xi], yVals[yi], zGrid[yi][xi]);
      const p2 = project(xVals[xi + 1], yVals[yi], zGrid[yi][xi + 1]);
      const p3 = project(xVals[xi + 1], yVals[yi + 1], zGrid[yi + 1][xi + 1]);
      const p4 = project(xVals[xi], yVals[yi + 1], zGrid[yi + 1][xi]);

      const avgTemp = (tGrid[yi][xi] + tGrid[yi][xi + 1] + tGrid[yi + 1][xi + 1] + tGrid[yi + 1][xi]) / 4;
      const avgFuel = (zGrid[yi][xi] + zGrid[yi][xi + 1] + zGrid[yi + 1][xi + 1] + zGrid[yi + 1][xi]) / 4;
      const avgX = (xVals[xi] + xVals[xi + 1]) / 2;
      const avgY = (yVals[yi] + yVals[yi + 1]) / 2;

      quads.push(
        <polygon
          key={`quad-${yi}-${xi}`}
          points={`${p1.px},${p1.py} ${p2.px},${p2.py} ${p3.px},${p3.py} ${p4.px},${p4.py}`}
          fill={colorAt(avgTemp)}
          stroke="#0b1220"
          strokeWidth="0.35"
          opacity="0.94"
          className="cursor-crosshair"
          onMouseEnter={(e) => {
            const svgEl = (e.target as SVGElement).ownerSVGElement;
            const rect = svgEl?.getBoundingClientRect();
            if (rect) {
              setTooltipPos({
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
              });
              setHoverInfo({
                turbine: avgX,
                compressor: avgY,
                fuel: avgFuel,
                temp: avgTemp,
              });
            }
          }}
          onMouseMove={(e) => {
            const svgEl = (e.target as SVGElement).ownerSVGElement;
            const rect = svgEl?.getBoundingClientRect();
            if (rect) {
              setTooltipPos({
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
              });
            }
          }}
          onMouseLeave={() => {
            setHoverInfo(null);
            setTooltipPos(null);
          }}
        />
      );
    }
  }

  // Mesh lines
  const rowLines = yVals.map((y, yi) => {
    const points = xVals.map((x, xi) => {
      const p = project(x, y, zGrid[yi][xi]);
      return `${p.px},${p.py}`;
    });
    return (
      <polyline
        key={`row-${yi}`}
        points={points.join(" ")}
        fill="none"
        stroke="rgba(255,255,255,0.20)"
        strokeWidth="0.6"
        pointerEvents="none"
      />
    );
  });

  const colLines = xVals.map((x, xi) => {
    const points = yVals.map((y, yi) => {
      const p = project(x, y, zGrid[yi][xi]);
      return `${p.px},${p.py}`;
    });
    return (
      <polyline
        key={`col-${xi}`}
        points={points.join(" ")}
        fill="none"
        stroke="rgba(255,255,255,0.20)"
        strokeWidth="0.6"
        pointerEvents="none"
      />
    );
  });

  const markerPoint =
    marker != null
      ? project(marker.turbine_decay, marker.compressor_decay, marker.fuel_flow)
      : null;

  // Floor corners at z-min
  const floorCorners = [
    project(b.x_min, b.y_min, b.z_min),
    project(b.x_max, b.y_min, b.z_min),
    project(b.x_max, b.y_max, b.z_min),
    project(b.x_min, b.y_max, b.z_min),
  ];

  // Axis anchor points:
  // frontOrigin: where X and Y intersect in front of the surface
  // zBottom/zTop: vertical axis on the right-middle side of the surface
  const frontOrigin = floorCorners[3];
  const xAxisEnd = floorCorners[2];
  const yAxisEnd = floorCorners[0];
  const zBottom = project(b.x_max, b.y_max, b.z_min);
  const zTop = project(b.x_max, b.y_max, b.z_max);

  const xTickVals = [b.x_min, (b.x_min + b.x_max) / 2, b.x_max];
  const yTickVals = [b.y_min, (b.y_min + b.y_max) / 2, b.y_max];

  const colorBarX = width - 92;
  const colorBarY = 150;
  const colorBarHeight = 360;
  const colorBarWidth = 18;

  return (
    <div className="relative h-full w-full">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-full w-full rounded-lg border border-slate-800 bg-[#0d1320]"
        preserveAspectRatio="xMidYMid meet"
      >
        <polygon
          points={floorCorners.map((p) => `${p.px},${p.py}`).join(" ")}
          fill="none"
          stroke="#506178"
          strokeWidth="1.2"
          strokeDasharray="6 3"
          opacity="0.6"
        />

        <g>{quads}</g>
        <g>{rowLines}</g>
        <g>{colLines}</g>

        {/* Foreground axes rendered on top of mesh for readability */}
        <g>
          <line x1={frontOrigin.px} y1={frontOrigin.py} x2={xAxisEnd.px} y2={xAxisEnd.py} stroke="#b8c5d9" strokeWidth="1.5" />
          <line x1={frontOrigin.px} y1={frontOrigin.py} x2={yAxisEnd.px} y2={yAxisEnd.py} stroke="#b8c5d9" strokeWidth="1.5" />
          <line x1={zBottom.px} y1={zBottom.py} x2={zTop.px} y2={zTop.py} stroke="#b8c5d9" strokeWidth="1.5" />

          <text
            x={(frontOrigin.px + xAxisEnd.px) / 2 + 8}
            y={(frontOrigin.py + xAxisEnd.py) / 2 + 28}
            fill="#c1cedf"
            fontSize="12"
            fontWeight="600"
            textAnchor="middle"
          >
            Turbine Decay (Kmt)
          </text>
          <text
            x={(frontOrigin.px + yAxisEnd.px) / 2 - 12}
            y={(frontOrigin.py + yAxisEnd.py) / 2 + 20}
            fill="#c1cedf"
            fontSize="12"
            fontWeight="600"
            textAnchor="end"
          >
            Compressor Decay (Kmc)
          </text>

          <text x={zTop.px + 8} y={zTop.py - 10} fill="#c1cedf" fontSize="12" fontWeight="600" textAnchor="start">
            Fuel Flow (kg/s)
          </text>
          <text x={zBottom.px + 8} y={zBottom.py + 6} fill="#90a3bb" fontSize="10" textAnchor="start">{b.z_min.toFixed(2)}</text>
          <text x={zTop.px + 8} y={zTop.py + 6} fill="#90a3bb" fontSize="10" textAnchor="start">{b.z_max.toFixed(2)}</text>

          {xTickVals.map((tick, idx) => {
            const p = project(tick, b.y_max, b.z_min);
            return (
              <text key={`xtick-${idx}`} x={p.px} y={p.py + 16} fill="#90a3bb" fontSize="10" textAnchor="middle">
                {tick.toFixed(3)}
              </text>
            );
          })}
          {yTickVals.map((tick, idx) => {
            const p = project(b.x_min, tick, b.z_min);
            return (
              <text key={`ytick-${idx}`} x={p.px - 10} y={p.py + 4} fill="#90a3bb" fontSize="10" textAnchor="end">
                {tick.toFixed(3)}
              </text>
            );
          })}
        </g>

        {markerPoint && marker && (
          <>
            <line
              x1={markerPoint.px}
              y1={markerPoint.py}
              x2={markerPoint.px}
              y2={project(marker.turbine_decay, marker.compressor_decay, b.z_min).py}
              stroke="#22c55e"
              strokeDasharray="4 3"
              strokeWidth="2"
              opacity="0.9"
            />
            <circle
              cx={markerPoint.px}
              cy={markerPoint.py}
              r="10"
              fill="#22c55e"
              stroke="#0d1320"
              strokeWidth="3"
            />
            <text x={markerPoint.px + 12} y={markerPoint.py - 8} fill="#22c55e" fontSize="11" fontWeight="700">
              Current
            </text>
          </>
        )}

        <defs>
          <linearGradient id="tempGradient" x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="#4575b4" />
            <stop offset="25%" stopColor="#91bfdb" />
            <stop offset="50%" stopColor="#e0f3f8" />
            <stop offset="75%" stopColor="#fdae61" />
            <stop offset="100%" stopColor="#d73027" />
          </linearGradient>
        </defs>
        <rect
          x={colorBarX}
          y={colorBarY}
          width={colorBarWidth}
          height={colorBarHeight}
          fill="url(#tempGradient)"
          stroke="#64748b"
          strokeWidth="1"
          rx="3"
        />
        <text x={colorBarX + colorBarWidth / 2} y={colorBarY - 15} fill="#94a3b8" fontSize="11" textAnchor="middle" fontWeight="500">
          T48 (°C)
        </text>
        <text x={colorBarX + colorBarWidth + 10} y={colorBarY + 8} fill="#e2e8f0" fontSize="11">
          {tMax.toFixed(0)}
        </text>
        <text x={colorBarX + colorBarWidth + 10} y={colorBarY + colorBarHeight / 2 + 4} fill="#e2e8f0" fontSize="11">
          {((tMin + tMax) / 2).toFixed(0)}
        </text>
        <text x={colorBarX + colorBarWidth + 10} y={colorBarY + colorBarHeight - 2} fill="#e2e8f0" fontSize="11">
          {tMin.toFixed(0)}
        </text>
      </svg>

      {hoverInfo && tooltipPos && (
        <div
          className="absolute pointer-events-none bg-slate-900/95 border border-slate-600 rounded-lg px-3 py-2 shadow-xl z-10"
          style={{
            left: Math.min(tooltipPos.x + 14, 540),
            top: Math.max(tooltipPos.y - 14, 70),
          }}
        >
          <div className="text-xs space-y-1">
            <div className="text-slate-300">Turbine: <span className="text-white font-mono">{hoverInfo.turbine.toFixed(4)}</span></div>
            <div className="text-slate-300">Compressor: <span className="text-white font-mono">{hoverInfo.compressor.toFixed(4)}</span></div>
            <div className="text-slate-300">Fuel: <span className="text-white font-mono">{hoverInfo.fuel.toFixed(3)} kg/s</span></div>
            <div className="text-slate-300">T48: <span className="text-white font-mono">{hoverInfo.temp.toFixed(1)} °C</span></div>
          </div>
        </div>
      )}

      {marker && (
        <div className="absolute bottom-4 left-4 bg-slate-900/90 border border-slate-700 rounded-lg px-4 py-3 shadow-xl">
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 rounded-full bg-green-500 border-2 border-slate-900 shadow-lg" />
            <div>
              <div className="text-sm font-semibold text-white">Current State</div>
              <div className="text-xs text-slate-400 space-x-3 mt-1">
                <span>Turbine: <span className="text-white font-mono">{marker.turbine_decay.toFixed(3)}</span></span>
                <span>Compressor: <span className="text-white font-mono">{marker.compressor_decay.toFixed(3)}</span></span>
                <span>Fuel: <span className="text-white font-mono">{marker.fuel_flow.toFixed(3)} kg/s</span></span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Health Hero Card component
function HealthHeroCard({
  title,
  value,
  severity,
  icon: Icon,
}: {
  title: string;
  value: number;
  severity: "healthy" | "warning" | "critical";
  icon: typeof HeartPulse;
}) {
  const percentage = value * 100;
  // Rescale bar to 90-100% range for visual clarity
  const barPercentage = Math.max(0, Math.min(100, ((value - 0.90) / 0.10) * 100));
  const variant =
    severity === "healthy"
      ? "success"
      : severity === "warning"
        ? "warning"
        : "critical";

  return (
    <Card className="relative overflow-hidden">
      <div
        className={`absolute -right-10 -top-10 h-32 w-32 rounded-full blur-3xl ${
          severity === "healthy"
            ? "bg-emerald-500/10"
            : severity === "warning"
              ? "bg-amber-500/10"
              : "bg-red-500/10"
        }`}
      />
      <CardContent className="relative p-5">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 text-slate-400">
              <Icon className="h-4 w-4" />
              <span className="text-xs font-medium">{title}</span>
            </div>
            <p className="mt-1 text-4xl font-bold tracking-tight text-white">
              {percentage.toFixed(1)}%
            </p>
          </div>
          <StatusBadge status={severity} showPulse={severity === "critical"} />
        </div>
        <div className="mt-3">
          <div className="flex justify-between text-[9px] text-slate-500 mb-1">
            <span>90%</span>
            <span>100%</span>
          </div>
          <ProgressBar value={barPercentage} variant={variant} size="md" />
        </div>
      </CardContent>
    </Card>
  );
}

export default function TurbineDashboard() {
  const snapshot = useQuery({
    queryKey: ["hmi-snapshot"],
    queryFn: fetchHmiSnapshot,
  });

  const speed = snapshot.data?.operating_state.ship_speed;

  const surface = useQuery({
    queryKey: ["hmi-surface", speed],
    queryFn: () => fetchSurfaceData(speed ?? 15),
    enabled: typeof speed === "number",
  });

  const marker = useQuery({
    queryKey: [
      "hmi-marker",
      speed,
      snapshot.data?.predictions.compressor_decay_pred,
      snapshot.data?.predictions.turbine_decay_pred,
    ],
    queryFn: () =>
      fetchSurfaceMarker({
        speed: speed ?? 15,
        compressor_decay_pred:
          snapshot.data?.predictions.compressor_decay_pred ?? 0.97,
        turbine_decay_pred:
          snapshot.data?.predictions.turbine_decay_pred ?? 0.99,
      }),
    enabled: !!snapshot.data && typeof speed === "number",
  });

  const recommendation = useQuery({
    queryKey: [
      "hmi-recommend",
      snapshot.data?.predictions.compressor_decay_pred,
      snapshot.data?.predictions.turbine_decay_pred,
    ],
    queryFn: () =>
      recommendMaintenance({
        compressor_decay: snapshot.data!.predictions.compressor_decay_pred,
        turbine_decay: snapshot.data!.predictions.turbine_decay_pred,
      }),
    enabled: !!snapshot.data,
  });

  const loading = snapshot.isLoading || surface.isLoading || marker.isLoading;

  const compressorDecay =
    snapshot.data?.predictions.compressor_decay_pred ?? 0.97;
  const turbineDecay = snapshot.data?.predictions.turbine_decay_pred ?? 0.99;
  const severity = snapshot.data?.predictions.severity ?? "healthy";

  const sensorData = snapshot.data
    ? {
        t1: snapshot.data.temperature_state.t1,
        t2: snapshot.data.temperature_state.t2,
        t48: snapshot.data.temperature_state.t48,
        p1: snapshot.data.pressure_state.p1,
        p2: snapshot.data.pressure_state.p2,
        p48: snapshot.data.pressure_state.p48,
        pexh: snapshot.data.pressure_state.pexh,
        fuelFlow: snapshot.data.operating_state.fuel_flow,
        gtRpm: snapshot.data.operating_state.gt_rpm,
        ggRpm: snapshot.data.operating_state.gg_rpm,
        gtTorque: snapshot.data.operating_state.gt_torque,
        tic: snapshot.data.operating_state.tic,
        shipSpeed: snapshot.data.operating_state.ship_speed,
        leverPos: snapshot.data.operating_state.lever_pos,
        propTorqueS: snapshot.data.ship_state?.prop_torque_s,
        propTorqueP: snapshot.data.ship_state?.prop_torque_p,
      }
    : undefined;

  return (
    <div className="space-y-4">
      {/* Maintenance Recommendation - Top Priority Alert */}
      <Card>
        <CardContent className="py-3 px-4">
          <div className="flex flex-wrap items-center gap-6">
            <div className="flex items-center gap-3">
              <div
                className={`rounded-lg p-2 ${
                  recommendation.data?.priority === "high"
                    ? "bg-red-500/20"
                    : recommendation.data?.priority === "medium"
                      ? "bg-amber-500/20"
                      : "bg-emerald-500/10"
                }`}
              >
                {recommendation.data?.priority === "high" ? (
                  <AlertTriangle className="h-5 w-5 text-red-400" />
                ) : (
                  <Wrench
                    className={`h-5 w-5 ${
                      recommendation.data?.priority === "medium"
                        ? "text-amber-400"
                        : "text-emerald-400"
                    }`}
                  />
                )}
              </div>
              <div>
                <p className="text-[10px] text-slate-400 uppercase tracking-wide">Action</p>
                <p className="font-semibold capitalize text-white">
                  {recommendation.data?.action ?? "-"}
                </p>
              </div>
            </div>

            <div className="h-8 w-px bg-slate-700" />

            <div>
              <p className="text-[10px] text-slate-400 uppercase tracking-wide">Priority</p>
              <p
                className={`font-semibold capitalize ${
                  recommendation.data?.priority === "high"
                    ? "text-red-400"
                    : recommendation.data?.priority === "medium"
                      ? "text-amber-400"
                      : "text-emerald-400"
                }`}
              >
                {recommendation.data?.priority ?? "-"}
              </p>
            </div>

            <div className="h-8 w-px bg-slate-700" />

            <div>
              <p className="text-[10px] text-slate-400 uppercase tracking-wide">Maintenance Window</p>
              <p className="font-semibold text-white">
                {recommendation.data?.maintenance_window ?? "-"}
              </p>
            </div>

            {recommendation.data?.components &&
              recommendation.data.components.length > 0 && (
                <>
                  <div className="h-8 w-px bg-slate-700" />
                  <div>
                    <p className="text-[10px] text-slate-400 uppercase tracking-wide">Components</p>
                    <p className="font-semibold text-white">
                      {recommendation.data.components.join(", ")}
                    </p>
                  </div>
                </>
              )}
          </div>
        </CardContent>
      </Card>

      {/* Hero: Turbine Schematic + Ship Status */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
        {/* Turbine Visualization - Takes 3 columns */}
        <Card className="overflow-hidden lg:col-span-3">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <Flame className="h-4 w-4 text-orange-400" />
              M501J Turbine 360 View + Live Sensor Strip
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <M501J360View />
            <SensorPanel data={sensorData} />
          </CardContent>
        </Card>

        {/* Ship Status Card - Takes 1 column */}
        <Card className="relative overflow-hidden">
          <div className="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-blue-500/10 blur-3xl" />
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <Ship className="h-4 w-4 text-blue-400" />
              Ship Status
            </CardTitle>
          </CardHeader>
          <CardContent className="relative space-y-4">
            {/* Ship Speed */}
            <div className="rounded-lg bg-surface-highlight/50 p-3">
              <div className="flex items-center gap-2 text-slate-400 mb-1">
                <Gauge className="h-4 w-4" />
                <span className="text-xs uppercase tracking-wide">Ship Speed</span>
              </div>
              <p className="text-3xl font-bold text-white">
                {sensorData?.shipSpeed ?? "-"}
                <span className="text-lg font-normal text-slate-400 ml-1">kn</span>
              </p>
            </div>

            {/* Lever Position */}
            <div className="rounded-lg bg-surface-highlight/50 p-3">
              <div className="flex items-center gap-2 text-slate-400 mb-1">
                <SlidersHorizontal className="h-4 w-4" />
                <span className="text-xs uppercase tracking-wide">Lever Position</span>
              </div>
              <p className="text-2xl font-bold text-white">
                {sensorData?.leverPos?.toFixed(2) ?? "-"}
              </p>
            </div>

            {/* Propeller Torques */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-slate-400 mb-2">
                <Anchor className="h-4 w-4" />
                <span className="text-xs uppercase tracking-wide">Propeller Torque</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-lg bg-surface-highlight/30 p-2 text-center">
                  <span className="text-[10px] text-slate-500 block">PORT</span>
                  <span className="text-lg font-mono font-bold text-white">
                    {sensorData?.propTorqueP?.toFixed(0) ?? "-"}
                  </span>
                  <span className="text-[10px] text-slate-500 block">kN·m</span>
                </div>
                <div className="rounded-lg bg-surface-highlight/30 p-2 text-center">
                  <span className="text-[10px] text-slate-500 block">STBD</span>
                  <span className="text-lg font-mono font-bold text-white">
                    {sensorData?.propTorqueS?.toFixed(0) ?? "-"}
                  </span>
                  <span className="text-[10px] text-slate-500 block">kN·m</span>
                </div>
              </div>
            </div>

            {/* TIC - bottom metric */}
            <div className="rounded-lg bg-surface-highlight/50 p-3">
              <div className="flex items-center gap-2 text-slate-400 mb-1">
                <SlidersHorizontal className="h-4 w-4" />
                <span className="text-xs uppercase tracking-wide">TIC</span>
              </div>
              <p className="text-2xl font-bold text-white">
                {sensorData?.tic?.toFixed(2) ?? "-"}
                <span className="text-base font-normal text-slate-400 ml-1">%</span>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Health Status + 3D Surface Row */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Combined Health Card - Big Square */}
        <Card className="relative overflow-hidden">
          <div className="absolute -right-20 -top-20 h-48 w-48 rounded-full bg-emerald-500/5 blur-3xl" />
          <div className="absolute -left-20 -bottom-20 h-48 w-48 rounded-full bg-blue-500/5 blur-3xl" />
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <HeartPulse className="h-5 w-5 text-primary" />
              System Health Status
            </CardTitle>
          </CardHeader>
          <CardContent className="relative space-y-6 p-6">
            {/* Turbine Health - Top */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Gauge className="h-5 w-5 text-slate-400" />
                  <span className="text-sm font-medium text-slate-300">Turbine Health</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-3xl font-bold text-white">{(turbineDecay * 100).toFixed(1)}%</span>
                  <StatusBadge
                    status={turbineDecay > 0.99 ? "healthy" : turbineDecay > 0.98 ? "warning" : "critical"}
                    showPulse={turbineDecay <= 0.98}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-[9px] text-slate-500 mb-1">
                  <span>90%</span>
                  <span>100%</span>
                </div>
                <ProgressBar
                  value={Math.max(0, Math.min(100, ((turbineDecay - 0.90) / 0.10) * 100))}
                  variant={turbineDecay > 0.99 ? "success" : turbineDecay > 0.98 ? "warning" : "critical"}
                  size="lg"
                />
              </div>
              <p className="text-xs text-slate-500">Decay coefficient: {turbineDecay.toFixed(6)}</p>
            </div>

            <div className="border-t border-slate-700/50" />

            {/* Compressor Health - Bottom */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-slate-400" />
                  <span className="text-sm font-medium text-slate-300">Compressor Health</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-3xl font-bold text-white">{(compressorDecay * 100).toFixed(1)}%</span>
                  <StatusBadge
                    status={compressorDecay > 0.98 ? "healthy" : compressorDecay > 0.96 ? "warning" : "critical"}
                    showPulse={compressorDecay <= 0.96}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-[9px] text-slate-500 mb-1">
                  <span>90%</span>
                  <span>100%</span>
                </div>
                <ProgressBar
                  value={Math.max(0, Math.min(100, ((compressorDecay - 0.90) / 0.10) * 100))}
                  variant={compressorDecay > 0.98 ? "success" : compressorDecay > 0.96 ? "warning" : "critical"}
                  size="lg"
                />
              </div>
              <p className="text-xs text-slate-500">Decay coefficient: {compressorDecay.toFixed(6)}</p>
            </div>
          </CardContent>
        </Card>

        {/* 3D Degradation Surface */}
        <Card className="lg:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <HeartPulse className="h-5 w-5 text-primary" />
              3D Degradation Surface
            </CardTitle>
            <div>
              <p className="text-xs text-slate-300">Fuel Consumption Surface at {speed ?? 12} kn</p>
              <p className="text-xs text-slate-500">Hover surface cells to inspect Kmt, Kmc, fuel flow, and T48</p>
            </div>
          </CardHeader>
          <CardContent className="p-3">
            {loading || !surface.data ? (
              <div className="flex h-[520px] items-center justify-center rounded-lg border border-slate-800 bg-[#0d1320] text-slate-400 text-sm">
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                Loading...
              </div>
            ) : (
              <div className="h-[520px]">
                <ProjectedSurface
                  surface={surface.data}
                  marker={marker.data?.marker}
                  shipSpeed={speed}
                />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

    </div>
  );
}
