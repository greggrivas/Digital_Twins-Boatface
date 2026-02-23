"use client";

import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fetchHmiSnapshot } from "@/lib/api";
import { Table, Database } from "lucide-react";

// Define all the data fields we want to show
const DATA_SECTIONS = [
  {
    title: "Operating State",
    icon: Table,
    fields: [
      { key: "ship_speed", label: "Ship Speed", unit: "kn", source: "operating_state" },
      { key: "lever_pos", label: "Lever Position", unit: "-", source: "operating_state" },
      { key: "gt_torque", label: "GT Torque", unit: "kN·m", source: "operating_state" },
      { key: "gt_rpm", label: "GT RPM", unit: "rpm", source: "operating_state" },
      { key: "gg_rpm", label: "GG RPM", unit: "rpm", source: "operating_state" },
      { key: "fuel_flow", label: "Fuel Flow", unit: "kg/s", source: "operating_state" },
    ],
  },
  {
    title: "Temperature State",
    icon: Table,
    fields: [
      { key: "t1", label: "T1 (External)", unit: "°C", source: "temperature_state" },
      { key: "t2", label: "T2 (Compressor Outlet)", unit: "°C", source: "temperature_state" },
      { key: "t48", label: "T48 (HP Turbine Exit)", unit: "°C", source: "temperature_state" },
    ],
  },
  {
    title: "Pressure State",
    icon: Table,
    fields: [
      { key: "p48", label: "P48 (HP Turbine Exit)", unit: "bar", source: "pressure_state" },
      { key: "p2", label: "P2 (Compressor Outlet)", unit: "bar", source: "pressure_state" },
      { key: "pexh", label: "Pexh (Exhaust)", unit: "bar", source: "pressure_state" },
    ],
  },
  {
    title: "Predictions",
    icon: Database,
    fields: [
      { key: "compressor_decay_pred", label: "Compressor Decay", unit: "-", source: "predictions" },
      { key: "turbine_decay_pred", label: "Turbine Decay", unit: "-", source: "predictions" },
      { key: "severity", label: "Severity", unit: "-", source: "predictions" },
      { key: "confidence_ref", label: "Model Confidence (R²)", unit: "-", source: "predictions" },
    ],
  },
];

function formatValue(value: unknown, key: string): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "string") return value;
  if (typeof value === "number") {
    // Format based on field type
    if (key.includes("decay") || key.includes("confidence")) {
      return value.toFixed(6);
    }
    if (key.includes("rpm") || key.includes("speed")) {
      return value.toFixed(0);
    }
    return value.toFixed(3);
  }
  return String(value);
}

export default function DataTable() {
  const snapshot = useQuery({
    queryKey: ["hmi-snapshot"],
    queryFn: fetchHmiSnapshot,
  });

  const data = snapshot.data;

  return (
    <div className="space-y-6">
      {/* Data Tables Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {DATA_SECTIONS.map((section) => (
          <Card key={section.title}>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <section.icon className="h-4 w-4 text-primary" />
                {section.title}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-800">
                      <th className="py-2 px-3 text-left font-medium text-slate-400">Field</th>
                      <th className="py-2 px-3 text-right font-medium text-slate-400">Value</th>
                      <th className="py-2 px-3 text-left font-medium text-slate-400">Unit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {section.fields.map((field, idx) => {
                      const sourceData = data?.[field.source as keyof typeof data] as Record<string, unknown> | undefined;
                      const value = sourceData?.[field.key];
                      return (
                        <tr
                          key={field.key}
                          className={idx % 2 === 0 ? "bg-surface-highlight/30" : ""}
                        >
                          <td className="py-2 px-3 text-slate-300">{field.label}</td>
                          <td className="py-2 px-3 text-right font-mono text-white">
                            {formatValue(value, field.key)}
                          </td>
                          <td className="py-2 px-3 text-slate-500">{field.unit}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Display Meta Card */}
      {data?.display_meta && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Table className="h-4 w-4 text-primary" />
              Data Ranges & Units Reference
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800">
                    <th className="py-2 px-3 text-left font-medium text-slate-400">Parameter</th>
                    <th className="py-2 px-3 text-left font-medium text-slate-400">Unit</th>
                    <th className="py-2 px-3 text-right font-medium text-slate-400">Min</th>
                    <th className="py-2 px-3 text-right font-medium text-slate-400">Max</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.display_meta.units).map(([key, unit], idx) => {
                    const range = data.display_meta.ranges[key];
                    return (
                      <tr
                        key={key}
                        className={idx % 2 === 0 ? "bg-surface-highlight/30" : ""}
                      >
                        <td className="py-2 px-3 text-slate-300">{key}</td>
                        <td className="py-2 px-3 text-slate-400">{unit}</td>
                        <td className="py-2 px-3 text-right font-mono text-white">
                          {range?.[0]?.toFixed(2) ?? "-"}
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-white">
                          {range?.[1]?.toFixed(2) ?? "-"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Health Bands Reference */}
      {data?.display_meta?.health_bands && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Database className="h-4 w-4 text-primary" />
              Health Classification Bands
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(data.display_meta.health_bands).map(([component, bands]) => (
                <div key={component} className="rounded-lg border border-slate-800 p-4">
                  <h4 className="font-medium text-white capitalize mb-2">{component}</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-emerald-400">Healthy</span>
                      <span className="font-mono text-slate-300">{(bands as Record<string, string>).healthy}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-amber-400">Warning</span>
                      <span className="font-mono text-slate-300">{(bands as Record<string, string>).warning}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-red-400">Critical</span>
                      <span className="font-mono text-slate-300">{(bands as Record<string, string>).critical}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
