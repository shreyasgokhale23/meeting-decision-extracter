import { avatarColor } from "@/lib/utils";

export default function Avatar({ name }: { name: string }) {
  return <span className="av" style={{ background: avatarColor(name) }}>{name[0]}</span>;
}

export function Person({ name }: { name: string }) {
  return <span className="who"><Avatar name={name} />{name}</span>;
}
