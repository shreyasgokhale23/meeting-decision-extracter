import type { Decision } from "@/types/meeting";
import { Person } from "./Avatar";

export default function DecisionsCard({ items }: { items: Decision[] }) {
  return (
    <div className="card">
      <div className="card-head"><h3>💡 Decisions</h3><a href="#" className="link">View All →</a></div>
      <div className="scroll short">
        <table>
          <thead><tr><th>#</th><th>Decision</th><th>Decided By</th><th>Time</th><th /></tr></thead>
          <tbody>
            {items.map((d, i) => (
              <tr key={d.id}>
                <td>{i + 1}</td><td>{d.decision}</td>
                <td><Person name={d.by} /></td><td>{d.time}</td>
                <td><button className="dots">⋮</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
