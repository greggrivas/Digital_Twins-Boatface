"use client";

import { useMemo, useState } from "react";

const FRAME_COUNT = 24;
const DEFAULT_FRAME_INDEX = 0; // side-oriented view

export default function M501J360View() {
  const frames = useMemo(
    () =>
      Array.from(
        { length: FRAME_COUNT },
        (_v, index) => `/m501j360/clean/img_${String(index + 1).padStart(2, "0")}.png`
      ),
    []
  );
  const [frameIndex, setFrameIndex] = useState(DEFAULT_FRAME_INDEX);

  return (
    <div className="relative h-[320px] overflow-hidden rounded-lg border border-slate-300 bg-[#e8edf0]">
      <img
        src={frames[frameIndex]}
        alt="M501J turbine render sequence"
        className="h-full w-full object-contain"
      />
      <div className="absolute bottom-3 right-3 w-56 rounded-md bg-black/35 px-3 py-2 backdrop-blur-sm">
        <div className="mb-1 flex items-center justify-between text-[11px] text-slate-100">
          <span>Rotate</span>
          <span>{frameIndex + 1}/{FRAME_COUNT}</span>
        </div>
        <input
          type="range"
          min={0}
          max={FRAME_COUNT - 1}
          step={1}
          value={frameIndex}
          onChange={(event) => setFrameIndex(Number(event.target.value))}
          className="w-full accent-cyan-400"
          aria-label="Rotate engine view"
        />
      </div>
    </div>
  );
}
