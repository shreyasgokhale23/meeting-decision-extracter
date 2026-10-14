import type { ActionItem } from "@/types/meeting";
import { Person } from "./Avatar";

interface Props { items: ActionItem[]; onToggle: (id: number) => void }

export default function ActionItemsCard({ items, onToggle }: Props) {
  return (
    <div className="card">
      <div className="card-head"><h3>✅ Action Items</h3><a href="#" className="link">View All →</a></div>
      <div className="scroll short">
        <table>
          <thead>
            <tr><th>#</th><th>Action</th><th>Responsible Person</th><th>Deadline</th><th>Status</th><th /></tr>
          </thead>
          <tbody>
            {items.map((a, i) => (
              <tr key={a.id}>
                <td>{i + 1}</td>
                <td>{a.action}</td>
                <td><Person name={a.owner} /></td>
                <td>{a.deadline}</td>
                <td><span className={`pill ${a.status === "Done" ? "done" : "pending"}`}>{a.status}</span></td>
                <td>
                  <button className="dots" title="Toggle status" onClick={() => onToggle(a.id)}>⋮</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
