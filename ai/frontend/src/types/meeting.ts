export type ParticipantStatus = "Speaking" | "Listening";
export type ActionStatus = "Pending" | "Done";

export interface Participant { name: string; status: ParticipantStatus }
export interface TranscriptLine { time: string; speaker: string; text: string }
export interface ActionItem {
  id: number; action: string; owner: string; deadline: string; status: ActionStatus;
}
export interface Decision { id: number; decision: string; by: string; time: string }

export interface Meeting {
  title: string;
  date: string;
  timeRange: string;
  participants: Participant[];
  transcript: TranscriptLine[];
  actions: ActionItem[];
  decisions: Decision[];
  seconds: number; // elapsed duration
}

export type TabKey = "transcript" | "actions" | "decisions" | "timeline";
