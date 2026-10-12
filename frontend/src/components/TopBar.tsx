interface Props {
  title: string; date: string; timeRange: string; count: number;
  ended: boolean; onEnd: () => void;
}

export default function TopBar({ title, date, timeRange, count, ended, onEnd }: Props) {
  return (
    <header className="topbar">
      <div>
        <h1>{title}</h1>
        <p className="muted">📅 {date} &nbsp;|&nbsp; {timeRange} &nbsp;|&nbsp; {count} Participants</p>
      </div>
      <div className="top-actions">
        <span className={`live-dot ${ended ? "off" : ""}`}>{ended ? "Ended" : "Live Meeting"}</span>
        <button className="btn-red" onClick={onEnd} disabled={ended}>⏹ End Meeting</button>
      </div>
    </header>
  );
}
