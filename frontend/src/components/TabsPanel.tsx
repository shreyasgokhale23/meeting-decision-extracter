"use client";
import { useState } from "react";
import type { Meeting, TabKey } from "@/types/meeting";
import Avatar from "./Avatar";

const TABS: { key: TabKey; label: string }[] = [
  { key: "transcript", label: "📄 Transcript" },
  { key: "actions", label: "✔ Action Items" },
  { key: "decisions", label: "💡 Decisions" },
  { key: "timeline", label: "🕒 Timeline" },
];

export default function TabsPanel({ meeting }: { meeting: Meeting }) {
  const [tab, setTab] = useState<TabKey>("transcript");

  return (
    <div className="card">
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t.key} className={`tab ${tab === t.key ? "active" : ""}`} onClick={() => setTab(t.key)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="scroll short">
        {tab === "transcript" && meeting.transcript.map((m, i) => (
          <div className="tab-row" key={i}>
            <span className="time">{m.time}</span><Avatar name={m.speaker} /><b>{m.speaker}:</b><span>{m.text}</span>
          </div>
        ))}
        {tab === "actions" && meeting.actions.map((a) => (
          <div className="tab-row" key={a.id}><b>{a.owner}</b><span>{a.action} — {a.deadline}</span></div>
        ))}
        {tab === "decisions" && meeting.decisions.map((d) => (
          <div className="tab-row" key={d.id}><span className="time">{d.time}</span><span>{d.decision} ({d.by})</span></div>
        ))}
        {tab === "timeline" && meeting.transcript.map((m, i) => (
          <div className="tab-row" key={i}><span className="time">{m.time}</span><span>{m.speaker} spoke</span></div>
        ))}
      </div>
    </div>
  );
}
