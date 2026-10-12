"use client";
import { useCallback, useEffect, useState } from "react";
import type { ActionItem, Decision, Meeting, TranscriptLine } from "@/types/meeting";
import { mockMeeting } from "@/lib/mockData";
import { endMeetingRequest, fetchMeeting, updateActionStatus } from "@/lib/api";

export function useMeeting() {
  const [meeting, setMeeting] = useState<Meeting>(mockMeeting);
  const [seconds, setSeconds] = useState(mockMeeting.seconds);
  const [listening, setListening] = useState(true);
  const [ended, setEnded] = useState(false);

  useEffect(() => {
    fetchMeeting()
      .then((m) => { setMeeting(m); setSeconds(m.seconds); })
      .catch((e) => console.warn("Backend unreachable, using mock data", e));
  }, []);

  useEffect(() => {
    if (!listening) return;
    const t = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [listening]);

  const toggleAction = useCallback((id: number) => {
    const current = meeting.actions.find((a) => a.id === id);
    if (!current) return;
    const next: ActionItem["status"] = current.status === "Done" ? "Pending" : "Done";
    setMeeting((m) => ({
      ...m,
      actions: m.actions.map((a) => (a.id === id ? { ...a, status: next } : a)),
    }));
    updateActionStatus(id, next).catch(console.error);
  }, [meeting.actions]);

  const endMeeting = useCallback(() => {
    if (!window.confirm("End this meeting?")) return;
    setListening(false);
    setEnded(true);
    endMeetingRequest({ ...meeting, seconds }).catch(console.error);
  }, [meeting, seconds]);

  // Call these from a WebSocket / SSE handler when the backend pushes data
  const addTranscriptLine = (l: TranscriptLine) =>
    setMeeting((m) => ({ ...m, transcript: [...m.transcript, l] }));
  const addAction = (a: ActionItem) =>
    setMeeting((m) => ({ ...m, actions: [...m.actions, a] }));
  const addDecision = (d: Decision) =>
    setMeeting((m) => ({ ...m, decisions: [...m.decisions, d] }));

  return {
    meeting, seconds, listening, ended,
    togglePause: () => setListening((l) => !l),
    toggleAction, endMeeting, addTranscriptLine, addAction, addDecision,
  };
}
