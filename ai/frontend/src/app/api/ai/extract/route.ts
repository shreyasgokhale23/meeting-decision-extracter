import { NextResponse } from "next/server";

// Placeholder: call your LLM here (keep AI_API_KEY server-side) and return
// { actions: ActionItem[], decisions: Decision[] } extracted from `text`.
export async function POST(req: Request) {
  const { text } = (await req.json()) as { text: string };
  void text;
  return NextResponse.json({ actions: [], decisions: [] });
}
