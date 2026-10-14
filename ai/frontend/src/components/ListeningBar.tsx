"use client";
import { useEffect, useState } from "react";

const BARS = 45;

interface Props { listening: boolean; ended: boolean; onToggle: () => void }

export default function ListeningBar({ listening, ended, onToggle }: Props) {
  const [heights, setHeights] = useState<number[]>(Array(BARS).fill(6));

  useEffect(() => {
    if (!listening) return;
    const t = setInterval(
      () => setHeights(Array.from({ length: BARS }, () => 4 + Math.random() * 24)),
      160
    );
    return () => clearInterval(t);
  }, [listening]);

  const label = ended ? "Meeting ended" : listening ? "Listening..." : "Paused";

  return (
    <div className={`listening ${listening ? "" : "paused"}`}>
      <div className="mic">🎙</div>
      <div className="wave-wrap">
        <span>{label}</span>
        <div className="wave">
          {heights.map((h, i) => <i key={i} style={{ height: h }} />)}
        </div>
      </div>
      <button className="btn" onClick={onToggle} disabled={ended}>
        {listening ? "⏸ Pause" : "▶ Resume"}
      </button>
      <button className="btn icon" aria-label="Settings">⚙️</button>
    </div>
  );
}
