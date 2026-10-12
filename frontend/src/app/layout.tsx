import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MeetingLens – Decision Extractor",
  description: "Live meeting transcript, action items and decisions.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
