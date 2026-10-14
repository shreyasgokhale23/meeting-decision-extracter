import type { Meeting } from "@/types/meeting";
import { formatDuration, stripWeekday } from "@/lib/utils";
import Avatar from "./Avatar";

export function MeetingInfo({ meeting, ended }: { meeting: Meeting; ended: boolean }) {
  return (
    <div className="card pad">
      <h3>📅 Meeting Info</h3>
      <p className="lbl">Title</p><p>{meeting.title}</p>
      <p className="lbl">Date &amp; Time</p><p>{meeting.date} | {meeting.timeRange}</p>
      <p className="lbl">Status <span className="pill green">{ended ? "Completed" : "In Progress"}</span></p>
    </div>
  );
}

export function Participants({ meeting }: { meeting: Meeting }) {
  return (
    <div className="card pad">
      <h3>👥 Participants</h3>
      <ul className="people">
        {meeting.participants.map((p) => (
          <li key={p.name}><Avatar name={p.name} /><b>{p.name}</b><span className="st">{p.status}</span></li>
        ))}
      </ul>
    </div>
  );
}

export function LiveSummary({ meeting, seconds }: { meeting: Meeting; seconds: number }) {
  return (
    <div className="card pad">
      <h3>📊 Live Summary</h3>
      <ul className="stats">
        <li><span>Action Items</span><b>{meeting.actions.length}</b></li>
        <li><span>Decisions</span><b>{meeting.decisions.length}</b></li>
        <li><span>Speakers Identified</span><b>{meeting.participants.length}</b></li>
        <li><span>Meeting Duration</span><b>{formatDuration(seconds)}</b></li>
      </ul>
    </div>
  );
}

export function buildSummaryText(m: Meeting): string {
  const done = m.actions.filter((a) => a.status === "Done").length;
  return [
    `${m.title} – ${m.date}`,
    `Participants: ${m.participants.map((p) => p.name).join(", ")}`,
    "", "Action Items:",
    ...m.actions.map((a, i) => `${i + 1}. ${a.action} — ${a.owner} — ${stripWeekday(a.deadline)}`),
    "", "Decisions:",
    ...m.decisions.map((d, i) => `${i + 1}. ${d.decision}`),
    "", `Pending: ${m.actions.length - done}`, `Completed: ${done}`,
  ].join("\n");
}

export function MeetingSummary({ meeting }: { meeting: Meeting }) {
  const done = meeting.actions.filter((a) => a.status === "Done").length;

  const download = () => {
    const url = URL.createObjectURL(new Blob([buildSummaryText(meeting)], { type: "text/plain" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: "meeting-summary.txt" });
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="card pad">
      <div className="card-head nopad">
        <h3>📝 Meeting Summary</h3>
        <button className="btn-outline" onClick={download}>⬇ Download</button>
      </div>
      <div id="summary">
        <h4>Participants</h4><p>{meeting.participants.map((p) => p.name).join(", ")}</p>
        <h4>Action Items</h4>
        <ol>{meeting.actions.map((a) => <li key={a.id}>{a.action} — {a.owner} — {stripWeekday(a.deadline)}</li>)}</ol>
        <h4>Decisions</h4>
        <ol>{meeting.decisions.map((d) => <li key={d.id}>{d.decision}</li>)}</ol>
        <h4>Pending Items</h4><p>{meeting.actions.length - done}</p>
        <h4>Completed Items</h4><p>{done}</p>
      </div>
    </div>
  );
}
