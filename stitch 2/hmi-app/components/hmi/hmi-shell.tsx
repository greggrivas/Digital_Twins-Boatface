"use client";

import { useState, useCallback, useEffect } from "react";
import { useQueryClient, useIsFetching } from "@tanstack/react-query";
import { PanelRightClose, PanelRightOpen, RefreshCw, GripVertical, LayoutDashboard, Table } from "lucide-react";
import { StatusBadge } from "@/components/ui/status-badge";
import { Button } from "@/components/ui/button";
import ChatPanel from "@/components/hmi/chat-panel";
import TurbineDashboard from "@/components/hmi/turbine-dashboard";
import DataTable from "@/components/hmi/data-table";
import { cn } from "@/lib/utils";

type ViewMode = "dashboard" | "data";

const MIN_WIDTH = 320;
const MAX_WIDTH = 700;
const DEFAULT_WIDTH = 400;

export default function HmiShell() {
  const [viewMode, setViewMode] = useState<ViewMode>("dashboard");
  const [chatOpen, setChatOpen] = useState(true);
  const [chatWidth, setChatWidth] = useState(DEFAULT_WIDTH);
  const [isDragging, setIsDragging] = useState(false);
  const queryClient = useQueryClient();
  const isFetching = useIsFetching({ queryKey: ["hmi-snapshot"] });

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ["hmi-snapshot"] });
  };

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return;
      const newWidth = window.innerWidth - e.clientX;
      setChatWidth(Math.min(Math.max(newWidth, MIN_WIDTH), MAX_WIDTH));
    },
    [isDragging]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    }
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  return (
    <div className="min-h-screen bg-background-dark text-slate-100">
      {/* Decorative background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 h-80 w-80 rounded-full bg-primary/5 blur-3xl" />
        <div className="absolute top-1/2 -left-40 h-80 w-80 rounded-full bg-emerald-500/5 blur-3xl" />
      </div>

      <header className="sticky top-0 z-50 border-b border-slate-800 bg-surface-dark/80 backdrop-blur-sm px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <StatusBadge status="online" showPulse />
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRefresh}
              disabled={isFetching > 0}
              className="transition-colors hover:bg-surface-highlight"
            >
              <RefreshCw className={cn("h-4 w-4", isFetching > 0 && "animate-spin")} />
            </Button>
            {/* View Mode Toggle */}
            <div className="flex items-center rounded-lg border border-slate-700 bg-surface-highlight/50 p-0.5">
              <button
                onClick={() => setViewMode("dashboard")}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-all",
                  viewMode === "dashboard"
                    ? "bg-primary text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                )}
              >
                <LayoutDashboard className="h-3.5 w-3.5" />
                Dashboard
              </button>
              <button
                onClick={() => setViewMode("data")}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-all",
                  viewMode === "data"
                    ? "bg-primary text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                )}
              >
                <Table className="h-3.5 w-3.5" />
                Data View
              </button>
            </div>
          </div>
          <div className="text-center">
            <h1 className="text-lg font-bold">HMI by IndustryStandard™</h1>
            <p className="text-xs text-slate-400">Gas Turbine Digital Twin</p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setChatOpen(!chatOpen)}
            className="transition-colors hover:bg-surface-highlight"
          >
            <span className="flex items-center gap-2">
              {chatOpen ? (
                <span className="text-xs font-semibold text-slate-200">Assistant</span>
              ) : null}
              {chatOpen ? (
                <PanelRightClose className="h-5 w-5" />
              ) : (
                <PanelRightOpen className="h-5 w-5" />
              )}
            </span>
          </Button>
        </div>
      </header>

      {/* Main content - adjusts width when chat is open */}
      <main
        className="relative z-10 p-6 transition-[margin] duration-300"
        style={{ marginRight: chatOpen ? chatWidth : 0 }}
      >
        {viewMode === "dashboard" ? <TurbineDashboard /> : <DataTable />}
      </main>

      {/* Chat Sidebar - overlay on top */}
      <aside
        className={cn(
          "fixed right-0 top-[57px] h-[calc(100vh-57px)] bg-surface-dark border-l border-slate-800 shadow-2xl transition-transform duration-300 z-40",
          chatOpen ? "translate-x-0" : "translate-x-full"
        )}
        style={{ width: chatWidth }}
      >
        {/* Resize handle */}
        <div
          onMouseDown={handleMouseDown}
          className={cn(
            "absolute left-0 top-0 h-full w-1 cursor-col-resize group hover:bg-primary/50 transition-colors",
            isDragging && "bg-primary"
          )}
        >
          <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
            <GripVertical className="h-6 w-6 text-slate-500" />
          </div>
        </div>

        <ChatPanel />
      </aside>

      {/* Overlay when dragging to prevent iframe issues */}
      {isDragging && <div className="fixed inset-0 z-50" />}
    </div>
  );
}
