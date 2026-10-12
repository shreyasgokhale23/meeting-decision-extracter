# MeetingLens – Decision Extractor

Responsive meeting dashboard (live transcript, action items, decisions) built with **Next.js 14 (App Router), React 18 and TypeScript**.

## Run locally
```bash
npm install
cp .env.example .env.local
npm run dev        # http://localhost:3000
```

## Project structure
```
src/
├─ app/
│  ├─ layout.tsx, page.tsx, globals.css
│  └─ api/
│     ├─ meetings/current/route.ts   # GET meeting (mock) – connect your DB here
│     └─ ai/extract/route.ts         # POST transcript -> actions/decisions (AI stub)
├─ components/                       # UI pieces (Dashboard composes them)
├─ hooks/useMeeting.ts               # all meeting state + actions
├─ lib/api.ts                        # every backend call lives here
├─ lib/mockData.ts, lib/utils.ts
└─ types/meeting.ts                  # shared TypeScript types
```

## Connecting backend & AI
1. Set `NEXT_PUBLIC_USE_MOCK=false` in `.env.local`.
2. Point `NEXT_PUBLIC_API_BASE` to your server (default `/api` = the Next.js routes).
3. Implement the endpoints used in `src/lib/api.ts`:
   `GET /meetings/current`, `PATCH /actions/:id`, `POST /meetings/end`, `POST /ai/extract`.
4. For live data, push into `addTranscriptLine`, `addAction`, `addDecision` (returned by `useMeeting`) from a WebSocket/SSE handler.
5. Put AI calls in `src/app/api/ai/extract/route.ts` and keep `AI_API_KEY` server-side only.
