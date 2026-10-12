"use client";
import { useMeeting } from "@/hooks/useMeeting";
import Sidebar from "./Sidebar";
import TopBar from "./TopBar";
import ListeningBar from "./ListeningBar";
import TranscriptCard from "./TranscriptCard";
import ActionItemsCard from "./ActionItemsCard";
import DecisionsCard from "./DecisionsCard";
import TabsPanel from "./TabsPanel";
import { LiveSummary, MeetingInfo, MeetingSummary, Participants } from "./SidePanels";

export default function Dashboard() {
  const { meeting, seconds, listening, ended, togglePause, toggleAction, endMeeting } = useMeeting();

  return (
    <div className="app">
      <Sidebar />
      <main className="main">
        <TopBar
          title={meeting.title} date={meeting.date} timeRange={meeting.timeRange}
          count={meeting.participants.length} ended={ended} onEnd={endMeeting}
        />
        <div className="grid">
          <section className="col">
            <ListeningBar listening={listening} ended={ended} onToggle={togglePause} />
            <TranscriptCard lines={meeting.transcript} />
            <ActionItemsCard items={meeting.actions} onToggle={toggleAction} />
            <DecisionsCard items={meeting.decisions} />
            <TabsPanel meeting={meeting} />
          </section>
          <aside className="col">
            <MeetingInfo meeting={meeting} ended={ended} />
            <Participants meeting={meeting} />
            <LiveSummary meeting={meeting} seconds={seconds} />
            <MeetingSummary meeting={meeting} />
          </aside>
        </div>
      </main>
    </div>
  );
}
