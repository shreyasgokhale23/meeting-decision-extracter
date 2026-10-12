const links = [
  { label: "🏠 Dashboard", active: true },
  { label: "📅 Meetings" },
  { label: "🕘 History" },
  { label: "⚙️ Settings" },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="logo">🎙</div>
        <div><b>MeetingLens</b><small>Decision Extractor</small></div>
      </div>
      <nav>
        {links.map((l) => (
          <a key={l.label} href="#" className={l.active ? "active" : ""}>{l.label}</a>
        ))}
      </nav>
      <div className="user">
        <div className="avatar">AS</div>
        <div><b>Aarav Sharma</b><small>Project Team</small></div>
      </div>
    </aside>
  );
}
