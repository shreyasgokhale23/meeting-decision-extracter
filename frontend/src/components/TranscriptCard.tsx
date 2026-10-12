"use client";
import { useEffect, useRef } from "react";
import type { TranscriptLine } from "@/types/meeting";
import Avatar from "./Avatar";

export default function TranscriptCard({ lines }: { lines: TranscriptLine[] }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight });
  }, [lines.length]);

  return (
    <div className="card">
      <div className="card-head"><h3>💬 Live Transcript</h3><span className="badge-live">● Live</span></div>
      <div className="scroll" ref={ref}>
        {lines.map((m, i) => (
          <div className="msg" key={i}>
            <span className="time">{m.time}</span>
            <Avatar name={m.speaker} />
            <div><b>{m.speaker}</b>{m.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
