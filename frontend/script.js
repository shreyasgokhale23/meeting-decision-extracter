/* ===== Config: change this when your backend is ready ===== */
const API_BASE = "http://localhost:5000/api"; // e.g. Express/Flask server
const USE_MOCK = true;                        // set false to use the backend

/* ===== State (mock data – replace with backend data) ===== */
const state = {
  participants: [
    { name: "Rahul", status: "Speaking" },
    { name: "Sneha", status: "Listening" },
    { name: "Amit",  status: "Listening" },
  ],
  transcript: [
    { time: "10:12:15", speaker: "Rahul", text: "I'll complete the backend by Friday." },
    { time: "10:12:27", speaker: "Sneha", text: "I'll prepare the UI tomorrow." },
    { time: "10:12:42", speaker: "Amit",  text: "Let's use MongoDB for the database." },
    { time: "10:13:05", speaker: "Rahul", text: "Also, we need to add unit tests before the release." },
    { time: "10:13:20", speaker: "Sneha", text: "I'll handle the testing part." },
  ],
  actions: [
    { id: 1, action: "Complete backend development", owner: "Rahul", deadline: "Apr 30, 2025 (Fri)", status: "Pending" },
    { id: 2, action: "Prepare UI design",            owner: "Sneha", deadline: "Apr 29, 2025 (Tue)", status: "Pending" },
    { id: 3, action: "Add unit tests",               owner: "Rahul", deadline: "May 2, 2025 (Fri)",  status: "Pending" },
  ],
  decisions: [
    { id: 1, decision: "Use MongoDB for the database", by: "Amit", time: "10:12 AM" },
  ],
  seconds: 13 * 60 + 24,
  listening: true,
  activeTab: "transcript",
};

const $ = (id) => document.getElementById(id);
const badge = (n) => `<span class="av c-${n}">${n[0]}</span>`;
const person = (n) => `<span class="who">${badge(n)}${n}</span>`;

/* ===== Render functions ===== */
function renderTranscript() {
  $("transcript").innerHTML = state.transcript.map(m => `
    <div class="msg">
      <span class="time">${m.time}</span>${badge(m.speaker)}
      <div><b>${m.speaker}</b>${m.text}</div>
    </div>`).join("");
  $("transcript").scrollTop = $("transcript").scrollHeight;
}

function renderActions() {
  $("actionBody").innerHTML = state.actions.map((a, i) => `
    <tr>
      <td>${i + 1}</td><td>${a.action}</td><td>${person(a.owner)}</td><td>${a.deadline}</td>
      <td><span class="pill ${a.status === "Done" ? "done" : "pending"}">${a.status}</span></td>
      <td><button class="dots" title="Toggle status" onclick="toggleAction(${a.id})">⋮</button></td>
    </tr>`).join("");
}

function renderDecisions() {
  $("decisionBody").innerHTML = state.decisions.map((d, i) => `
    <tr><td>${i + 1}</td><td>${d.decision}</td><td>${person(d.by)}</td><td>${d.time}</td>
    <td><button class="dots">⋮</button></td></tr>`).join("");
}

function renderPeople() {
  $("people").innerHTML = state.participants.map(p =>
    `<li>${badge(p.name)}<b>${p.name}</b><span class="st">${p.status}</span></li>`).join("");
}

function renderTabs() {
  const t = state.activeTab, box = $("tabContent");
  if (t === "transcript") box.innerHTML = state.transcript.map(m =>
    `<div class="tab-row"><span class="time">${m.time}</span>${badge(m.speaker)}<b class="t-${m.speaker}">${m.speaker}:</b><span>${m.text}</span></div>`).join("");
  if (t === "actions") box.innerHTML = state.actions.map(a =>
    `<div class="tab-row"><b>${a.owner}</b><span>${a.action} — ${a.deadline}</span></div>`).join("");
  if (t === "decisions") box.innerHTML = state.decisions.map(d =>
    `<div class="tab-row"><span class="time">${d.time}</span><span>${d.decision} (${d.by})</span></div>`).join("");
  if (t === "timeline") box.innerHTML = [...state.transcript].map(m =>
    `<div class="tab-row"><span class="time">${m.time}</span><span>${m.speaker} spoke</span></div>`).join("");
}

function renderSummary() {
  const done = state.actions.filter(a => a.status === "Done").length;
  const short = (d) => d.replace(/ \(.*\)/, "");
  $("sAction").textContent = state.actions.length;
  $("sDecision").textContent = state.decisions.length;
  $("sSpeakers").textContent = state.participants.length;
  $("summary").innerHTML = `
    <h4>Participants</h4><p>${state.participants.map(p => p.name).join(", ")}</p>
    <h4>Action Items</h4><ol>${state.actions.map(a => `<li>${a.action} — ${a.owner} — ${short(a.deadline)}</li>`).join("")}</ol>
    <h4>Decisions</h4><ol>${state.decisions.map(d => `<li>${d.decision}</li>`).join("")}</ol>
    <h4>Pending Items</h4><p>${state.actions.length - done}</p>
    <h4>Completed Items</h4><p>${done}</p>`;
}

function renderAll() {
  renderTranscript(); renderActions(); renderDecisions();
  renderPeople(); renderTabs(); renderSummary();
}

/* ===== Interactions ===== */
function toggleAction(id) {
  const a = state.actions.find(x => x.id === id);
  a.status = a.status === "Done" ? "Pending" : "Done";
  renderActions(); renderSummary(); renderTabs();
  if (!USE_MOCK) api(`/actions/${id}`, "PATCH", { status: a.status });
}

document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  state.activeTab = btn.dataset.tab;
  renderTabs();
}));

$("pauseBtn").addEventListener("click", () => {
  state.listening = !state.listening;
  $("listenBox").classList.toggle("paused", !state.listening);
  $("pauseBtn").textContent = state.listening ? "⏸ Pause" : "▶ Resume";
  $("listenText").textContent = state.listening ? "Listening..." : "Paused";
});

$("endBtn").addEventListener("click", () => {
  if (!confirm("End this meeting?")) return;
  state.listening = false;
  $("listenBox").classList.add("paused");
  $("listenText").textContent = "Meeting ended";
  $("liveLabel").textContent = "Ended"; $("liveLabel").classList.add("off");
  $("statusPill").textContent = "Completed";
  $("endBtn").disabled = true;
  if (!USE_MOCK) api("/meetings/end", "POST", state);
});

$("downloadBtn").addEventListener("click", () => {
  const blob = new Blob([$("summary").innerText], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "meeting-summary.txt"; a.click();
});

/* ===== Waveform + timer ===== */
const wave = $("wave");
for (let i = 0; i < 45; i++) wave.appendChild(document.createElement("i"));
setInterval(() => {
  if (!state.listening) return;
  wave.querySelectorAll("i").forEach(b => b.style.height = (4 + Math.random() * 24) + "px");
}, 160);

setInterval(() => {
  if (!state.listening) return;
  state.seconds++;
  $("sDuration").textContent = `${Math.floor(state.seconds / 60)}m ${String(state.seconds % 60).padStart(2, "0")}s`;
}, 1000);

/* ===== Backend hooks ===== */
async function api(path, method = "GET", body) {
  const res = await fetch(API_BASE + path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  return res.json();
}

// Call this when your backend/WebSocket sends a new transcript line
function addTranscriptLine(line) {            // { time, speaker, text }
  state.transcript.push(line);
  renderTranscript(); renderTabs();
}
// Call this when the backend extracts a new action item / decision
function addActionItem(item)  { state.actions.push(item);   renderActions();   renderSummary(); renderTabs(); }
function addDecision(item)    { state.decisions.push(item); renderDecisions(); renderSummary(); renderTabs(); }

async function init() {
  if (!USE_MOCK) {
    try {
      const data = await api("/meetings/current");
      Object.assign(state, data);
    } catch (e) { console.warn("Backend not reachable, using mock data", e); }
  }
  renderAll();
}
init();
