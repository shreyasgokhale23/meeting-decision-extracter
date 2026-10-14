import { NextResponse } from "next/server";
import { mockMeeting } from "@/lib/mockData";

// TODO: replace with a database / backend call
export async function GET() {
  return NextResponse.json(mockMeeting);
}
