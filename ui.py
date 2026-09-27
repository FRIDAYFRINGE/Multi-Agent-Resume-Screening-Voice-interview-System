"""
UI template — dark-theme single-page HTML/CSS/JS for the AI Interview Assistant.
Imported by interview_app.py and served at GET /.
"""

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Interview Assistant</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; min-height: 100vh; }

  .container { max-width: 920px; margin: 0 auto; padding: 32px 20px; }

  h1 { font-size: 1.6rem; font-weight: 600; color: #fff; margin-bottom: 4px; }
  .subtitle { font-size: 0.85rem; color: #888; margin-bottom: 32px; }

  .card { background: #1a1d27; border: 1px solid #2a2d3e; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
  .card h2 { font-size: 1rem; font-weight: 600; color: #a0a8ff; margin-bottom: 16px; }
  label { display: block; font-size: 0.8rem; color: #888; margin-bottom: 6px; margin-top: 12px; }
  input[type=text], textarea, select {
    width: 100%; background: #0f1117; border: 1px solid #2a2d3e; border-radius: 8px;
    color: #e0e0e0; padding: 10px 14px; font-size: 0.9rem; outline: none;
    transition: border-color 0.2s;
  }
  input[type=text]:focus, textarea:focus { border-color: #5c6aff; }
  textarea { resize: vertical; min-height: 100px; }

  .btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 10px 22px; border-radius: 8px; border: none;
    font-size: 0.9rem; font-weight: 500; cursor: pointer;
    transition: all 0.2s;
  }
  .btn-primary { background: #5c6aff; color: #fff; }
  .btn-primary:hover { background: #4a58ee; }
  .btn-primary:disabled { background: #2a2d3e; color: #555; cursor: not-allowed; }
  .btn-danger  { background: #ff4f4f; color: #fff; }
  .btn-danger:hover  { background: #e03e3e; }
  .btn-success { background: #27ae60; color: #fff; }
  .btn-success:hover { background: #219a52; }
  .btn-sm { padding: 6px 14px; font-size: 0.8rem; }

  /* Status bar */
  #status-bar {
    display: flex; align-items: center; gap: 10px;
    background: #1a1d27; border: 1px solid #2a2d3e;
    border-radius: 8px; padding: 10px 16px;
    margin-bottom: 20px; font-size: 0.85rem;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #888; flex-shrink: 0; }
  .dot.green { background: #27ae60; box-shadow: 0 0 6px #27ae60; }
  .dot.yellow { background: #f39c12; box-shadow: 0 0 6px #f39c12; animation: pulse 1s infinite; }
  .dot.red { background: #e74c3c; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* Pool badge */
  .pool-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1e2a3a; border: 1px solid #2a4060; border-radius: 20px;
    padding: 4px 12px; font-size: 0.78rem; color: #5fa8ff; margin-bottom: 16px;
  }
  .pool-badge .count { font-weight: 700; }

  /* File input */
  .file-input-wrap {
    background: #0f1117; border: 1px dashed #2a2d3e; border-radius: 8px;
    padding: 12px 14px; width: 100%; color: #888; font-size: 0.85rem;
  }
  .file-input-wrap input[type=file] { width: 100%; color: #e0e0e0; }

  /* Rankings panel */
  #rankings-panel { display: none; }
  .rankings-header {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;
  }
  .pool-stats { font-size: 0.82rem; color: #888; }

  .rankings-table { width: 100%; border-collapse: collapse; }
  .rankings-table th {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px;
    color: #666; padding: 8px 12px; text-align: left; border-bottom: 1px solid #2a2d3e;
  }
  .rankings-table td { padding: 12px 12px; border-bottom: 1px solid #1a1d27; font-size: 0.88rem; vertical-align: middle; }
  .rankings-table tr:last-child td { border-bottom: none; }
  .rankings-table tr:hover td { background: #1f2233; }

  .rank-num { font-weight: 700; color: #a0a8ff; font-size: 1rem; }
  .rank-1 .rank-num { color: #ffd700; }
  .rank-2 .rank-num { color: #c0c0c0; }
  .rank-3 .rank-num { color: #cd7f32; }

  .score-pill {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.78rem; font-weight: 600;
  }
  .score-high { background: #1a3a1a; color: #5fd48a; border: 1px solid #27ae60; }
  .score-mid  { background: #3a3a0f; color: #ffe080; border: 1px solid #f39c12; }
  .score-low  { background: #3a0f0f; color: #ff8080; border: 1px solid #e74c3c; }

  .rec-badge {
    font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
    border-radius: 10px; text-transform: uppercase; letter-spacing: 0.4px;
  }
  .rec-strong { background: #0f2a0f; color: #27ae60; border: 1px solid #27ae60; }
  .rec-match  { background: #1e3a5f; color: #5fa8ff; border: 1px solid #3a7abf; }
  .rec-weak   { background: #3a3a0f; color: #ffe080; border: 1px solid #f39c12; }
  .rec-no     { background: #2a1a1a; color: #888; border: 1px solid #444; }

  /* Interview panel */
  #interview-panel { display: none; }

  .progress-bar { background: #2a2d3e; border-radius: 4px; height: 4px; margin-bottom: 24px; }
  .progress-fill { height: 100%; border-radius: 4px; background: #5c6aff; transition: width 0.4s; }

  .question-meta { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; }
  .badge {
    font-size: 0.7rem; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .badge-tech   { background: #1e3a5f; color: #5fa8ff; }
  .badge-gap    { background: #3a1e1e; color: #ff8080; }
  .badge-proj   { background: #1e3a2a; color: #5fd48a; }
  .badge-sysdes { background: #2a1e3a; color: #c080ff; }
  .badge-beh    { background: #3a3a1e; color: #ffe080; }

  .question-box {
    background: #0f1117; border: 1px solid #2a2d3e; border-left: 3px solid #5c6aff;
    border-radius: 8px; padding: 18px 20px; font-size: 1rem; line-height: 1.6;
    margin-bottom: 20px; min-height: 70px;
  }

  audio { width: 100%; border-radius: 8px; margin-bottom: 16px; }

  .record-area {
    border: 2px dashed #2a2d3e; border-radius: 12px;
    padding: 28px; text-align: center; margin-bottom: 20px;
    transition: border-color 0.2s;
  }
  .record-area.recording { border-color: #ff4f4f; background: #1a0f0f; }
  .record-area.done { border-color: #27ae60; background: #0f1a0f; }

  .mic-icon { font-size: 2.5rem; margin-bottom: 8px; }
  .record-hint { font-size: 0.8rem; color: #888; margin-top: 8px; }

  #timer { font-size: 1.4rem; font-weight: 700; color: #ff4f4f; font-variant-numeric: tabular-nums; }

  .answer-box {
    background: #0f1117; border: 1px solid #2a2d3e; border-radius: 8px;
    padding: 14px 16px; min-height: 60px; font-size: 0.9rem; line-height: 1.5;
    margin-bottom: 16px; white-space: pre-wrap; color: #ccc;
  }

  .eval-row { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
  .eval-chip {
    background: #1a1d27; border: 1px solid #2a2d3e; border-radius: 8px;
    padding: 8px 14px; font-size: 0.8rem;
  }
  .eval-chip span { font-weight: 700; color: #a0a8ff; }
  .score-chip-high { border-color: #27ae60; }
  .score-chip-mid  { border-color: #f39c12; }
  .score-chip-low  { border-color: #e74c3c; }

  .hits-misses { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }
  .hits-misses div { background: #0f1117; border: 1px solid #2a2d3e; border-radius: 8px; padding: 12px 14px; }
  .hits-misses h4 { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
  .hits-misses .hits h4 { color: #27ae60; }
  .hits-misses .misses h4 { color: #e74c3c; }
  .hits-misses li { font-size: 0.82rem; color: #aaa; margin-left: 14px; margin-bottom: 4px; }

  /* STT progress bar */
  #stt-progress-wrap { height: 4px; background: #1a1d27; border-radius: 4px; margin-bottom: 12px; overflow: hidden; }
  #stt-progress-bar  { height: 100%; background: #5c6aff; width: 0%; transition: width 0.3s linear; }

  /* Report */
  #report-panel { display: none; }
  .report-header { text-align: center; margin-bottom: 28px; }
  .verdict {
    display: inline-block; font-size: 1.1rem; font-weight: 700;
    padding: 10px 28px; border-radius: 8px; margin-top: 12px;
  }
  .verdict-HIRE        { background: #1a3a1a; color: #5fd48a; border: 1px solid #27ae60; }
  .verdict-STRONG_HIRE { background: #0f2a0f; color: #27ae60; border: 1px solid #27ae60; }
  .verdict-HOLD        { background: #3a3a0f; color: #ffe080; border: 1px solid #f39c12; }
  .verdict-REJECT      { background: #3a0f0f; color: #ff8080; border: 1px solid #e74c3c; }

  .score-overall { font-size: 2.8rem; font-weight: 800; color: #a0a8ff; }
  .score-label { font-size: 0.85rem; color: #888; }

  .category-scores { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .cat-score { background: #1a1d27; border: 1px solid #2a2d3e; border-radius: 8px; padding: 14px; text-align: center; }
  .cat-score .name { font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .cat-score .val { font-size: 1.4rem; font-weight: 700; color: #a0a8ff; }

  .log { max-height: 320px; overflow-y: auto; }
  .log-entry { border-bottom: 1px solid #1a1d27; padding: 12px 0; }
  .log-entry:last-child { border-bottom: none; }
  .log-q { font-size: 0.82rem; color: #888; margin-bottom: 4px; }
  .log-a { font-size: 0.88rem; color: #ccc; margin-bottom: 6px; }
  .log-score { font-size: 0.78rem; }
  .log-score .good { color: #27ae60; } .log-score .mid { color: #f39c12; } .log-score .bad { color: #e74c3c; }
  #log-area { margin-top: 8px; }

  .back-link {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.82rem; color: #666; cursor: pointer; margin-bottom: 16px;
    background: none; border: none; padding: 0;
  }
  .back-link:hover { color: #a0a8ff; }
</style>
</head>
<body>
<div class="container">
  <h1>AI Interview Assistant</h1>
  <p class="subtitle">Multi-agent resume screening + live voice interview pipeline</p>

  <!-- Status bar -->
  <div id="status-bar">
    <div class="dot" id="status-dot"></div>
    <span id="status-text">Ready</span>
  </div>

  <!-- ── Setup panel ──────────────────────────────────────────────────── -->
  <div id="setup-panel">
    <div class="card">
      <h2>Candidate Pool</h2>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:16px;">
        <div class="pool-badge" style="margin-bottom:0;">
          Candidates in pool: <span class="count" id="pool-count">...</span>
        </div>
        <div class="pool-badge" style="margin-bottom:0;" id="mongo-badge">
          MongoDB: <span id="mongo-status">checking...</span>
        </div>
      </div>
      <div id="history-links" style="display:none;font-size:0.8rem;color:#666;margin-bottom:12px;">
        Past runs: <span id="job-history-list"></span>
      </div>
      <label>Add new resumes to pool (optional — leave blank to screen existing pool)</label>
      <div class="file-input-wrap">
        <input type="file" id="resume-file" accept=".pdf,.docx,.txt" multiple>
      </div>
      <div id="file-count" style="font-size:0.78rem;color:#666;margin-top:6px;"></div>
    </div>

    <div class="card">
      <h2>Job Description</h2>
      <textarea id="jd-input" placeholder="Paste the full job description here...">Senior Business Analyst - Agile & Enterprise Systems
Lead requirements elicitation, analysis, and documentation (BRD, FRD, SRS, Use Cases, User Stories, Acceptance Criteria).
Facilitate Agile/Scrum ceremonies (Sprint Planning, Backlog Grooming, JAD sessions) and manage product backlogs.
Model business processes and workflows using BPMN, UML, MS Visio, or Lucidchart.
Coordinate User Acceptance Testing (UAT), define test cases, and manage defect lifecycle in JIRA or HP ALM.
Perform data analysis, business rule validation, and SQL queries to support data-driven decision making.
Strong stakeholder communication, cross-functional leadership, and domain expertise.</textarea>

      <label>Required Skills (comma-separated)</label>
      <input type="text" id="skills-input" value="BRD, FRD, Agile, Scrum, JIRA, UML, MS Visio, SQL, User Stories, UAT">

      <label>Max follow-ups per question</label>
      <select id="followups-select" style="background:#0f1117;border:1px solid #2a2d3e;border-radius:8px;padding:10px 14px;width:100%;color:#e0e0e0;">
        <option value="0">0 — No follow-ups</option>
        <option value="1" selected>1 — One follow-up if answer is shallow</option>
        <option value="2">2 — Up to two follow-ups</option>
      </select>

      <div style="margin-top:20px;">
        <button class="btn btn-primary" id="screen-btn" onclick="startScreening()">
          Screen the Pool
        </button>
      </div>
    </div>
  </div>

  <!-- ── Rankings panel ──────────────────────────────────────────────── -->
  <div id="rankings-panel">

    <!-- Cached-run notice (shown when fast-path reuses a previous job) -->
    <div id="cached-notice" style="display:none;margin-bottom:14px;border-radius:10px;
         background:#0d1117;border:1px solid #2a2d3e;overflow:hidden;">
      <div style="display:flex;align-items:center;gap:14px;padding:12px 16px;">
        <div style="width:32px;height:32px;border-radius:8px;background:#1a1d27;border:1px solid #2a2d3e;
                    display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:1rem;">&#128336;</div>
        <div style="flex:1;min-width:0;">
          <div style="font-size:0.82rem;font-weight:600;color:#e0e0e0;margin-bottom:2px;">Cached rankings</div>
          <div style="font-size:0.76rem;color:#666;" id="cached-notice-text"></div>
        </div>
        <div id="pool-change-badge" style="display:none;flex-shrink:0;padding:4px 10px;border-radius:20px;
             background:#1a1400;border:1px solid #4a3a00;font-size:0.72rem;font-weight:600;color:#c8a020;
             white-space:nowrap;"></div>
      </div>
    </div>

    <!-- Uploaded candidates highlight card -->
    <div id="uploaded-section" style="display:none;margin-bottom:18px;">
      <div class="card" style="border-color:#2a5c3a;background:#0a1a10;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
          <span style="font-size:1.1rem;font-weight:700;color:#4caf50;">Your Uploaded Candidates</span>
          <span style="font-size:0.75rem;color:#888;">Qualified for immediate interview</span>
        </div>
        <div id="uploaded-cards" style="display:flex;flex-direction:column;gap:10px;"></div>
      </div>
    </div>

    <div class="card">
      <div class="rankings-header">
        <div>
          <h2 style="margin-bottom:4px;">Talent Pool Rankings</h2>
          <div class="pool-stats" id="pool-stats"></div>
        </div>
        <button class="btn btn-sm" style="background:#1a1d27;border:1px solid #2a2d3e;color:#888;"
          onclick="backToSetup()">New Search</button>
      </div>
      <div style="overflow-x:auto;">
        <table class="rankings-table" id="rankings-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Candidate</th>
              <th>Score</th>
              <th>Match Level</th>
              <th>Key Strengths</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="rankings-body"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ── Interview panel ─────────────────────────────────────────────── -->
  <div id="interview-panel">
    <button class="back-link" onclick="backToRankings()">&#8592; Back to rankings</button>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <div>
          <h2 id="candidate-name" style="margin-bottom:2px;"></h2>
          <div style="font-size:0.8rem;color:#888;" id="interview-meta"></div>
        </div>
        <div style="text-align:right;display:flex;flex-direction:column;align-items:flex-end;gap:6px;">
          <div>
            <div style="font-size:0.75rem;color:#888;">Question</div>
            <div style="font-size:1.1rem;font-weight:700;color:#a0a8ff;" id="q-counter">1 / ?</div>
          </div>
          <button class="btn btn-sm" style="background:#1a1d27;border:1px solid #2a2d3e;color:#888;"
            onclick="restartInterview()" id="restart-btn" title="Restart from Q1">&#8635; Restart</button>
        </div>
      </div>
      <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>

      <div class="question-meta">
        <span class="badge" id="q-badge">Technical</span>
        <span style="font-size:0.75rem;color:#888;" id="q-gap"></span>
      </div>
      <div class="question-box" id="question-text">Loading question...</div>

      <audio id="tts-audio" controls autoplay style="display:none;"></audio>

      <div class="record-area" id="record-area">
        <div class="mic-icon" id="mic-icon">&#127897;&#65039;</div>
        <div id="record-label">Click to start recording your answer</div>
        <div id="timer" style="display:none;">0:00</div>
        <div class="record-hint">Press the button or use spacebar</div>
      </div>

      <div style="display:flex;gap:10px;margin-bottom:16px;">
        <button class="btn btn-danger" id="record-btn" onclick="toggleRecording()">
          &#128308; Start Recording
        </button>
        <button class="btn btn-success" id="submit-btn" onclick="submitAnswer()" disabled>
          Submit Answer &rarr;
        </button>
      </div>

      <div id="answer-section" style="display:none;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <label style="margin:0;">Raw Whisper Transcript</label>
          <button class="btn btn-sm" style="background:none;border:1px solid #2a2d3e;color:#666;padding:2px 8px;font-size:0.72rem;"
            onclick="toggleRawTranscript()" id="raw-toggle-btn">Hide</button>
        </div>
        <div id="stt-progress-wrap" style="display:none;">
          <div id="stt-progress-bar"></div>
        </div>
        <div class="answer-box" id="answer-text"></div>

        <div id="eval-section" style="display:none;">
          <div class="eval-row" id="eval-chips"></div>
          <div id="brief-feedback-box" style="display:none;background:#0f1117;border:1px solid #2a2d3e;border-left:3px solid #5c6aff;border-radius:8px;padding:10px 14px;font-size:0.85rem;color:#ccc;margin-bottom:12px;line-height:1.5;">
            <span style="font-size:0.7rem;color:#5c6aff;text-transform:uppercase;letter-spacing:0.5px;">AI Evaluation Notes</span><br>
            <span id="brief-feedback-text"></span>
          </div>
          <div class="hits-misses">
            <div class="hits"><h4>Covered</h4><ul id="hits-list"></ul></div>
            <div class="misses"><h4>Missed</h4><ul id="misses-list"></ul></div>
          </div>
          <div style="display:flex;gap:10px;">
            <button class="btn btn-primary" id="next-btn" onclick="nextQuestion()">
              Next Question &rarr;
            </button>
          </div>
        </div>
      </div>

      <div id="log-area" style="display:none;">
        <div style="font-size:0.75rem;color:#888;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">
          Answered so far
        </div>
        <div class="log" id="log-entries"></div>
      </div>
    </div>
  </div>

  <!-- ── Report panel ────────────────────────────────────────────────── -->
  <div id="report-panel">
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:16px;">
      <button class="back-link" style="margin-bottom:0;" onclick="backToRankings()">&#8592; Back to rankings</button>
      <button class="btn btn-sm" style="background:#1a1d27;border:1px solid #2a2d3e;color:#888;"
        onclick="restartInterview()">&#8635; Restart Interview</button>
    </div>
    <div class="card">
      <div class="report-header">
        <div style="font-size:0.8rem;color:#888;margin-bottom:4px;">Interview Complete</div>
        <div class="score-overall" id="report-score"></div>
        <div class="score-label">/ 10 overall score</div>
        <div class="verdict" id="report-verdict"></div>
      </div>
      <div class="category-scores" id="cat-scores"></div>
      <div style="margin-bottom:20px;">
        <div style="font-size:0.8rem;color:#888;margin-bottom:8px;">Summary</div>
        <div style="font-size:0.9rem;line-height:1.6;color:#ccc;" id="report-summary"></div>
      </div>
      <div class="hits-misses" style="margin-bottom:20px;">
        <div class="hits"><h4>Top Strengths</h4><ul id="report-strengths"></ul></div>
        <div class="misses"><h4>Key Concerns</h4><ul id="report-concerns"></ul></div>
      </div>
      <div style="margin-bottom:20px;">
        <div style="font-size:0.8rem;color:#888;margin-bottom:8px;">Hiring Recommendation</div>
        <div style="font-size:0.9rem;line-height:1.6;color:#ccc;background:#0f1117;border:1px solid #2a2d3e;border-radius:8px;padding:14px;" id="report-rec"></div>
      </div>
      <div id="report-downloads" style="display:flex;gap:10px;margin-bottom:20px;">
        <button class="btn btn-primary btn-sm" onclick="downloadPDF()">&#8595; Download PDF</button>
        <button class="btn btn-sm" style="background:#1a1d27;border:1px solid #2a2d3e;color:#ccc;" onclick="downloadJSON()">&#8595; Download JSON</button>
      </div>
      <div>
        <div style="font-size:0.8rem;color:#888;margin-bottom:8px;">Full Transcript</div>
        <div class="log" id="report-log"></div>
      </div>
    </div>
  </div>

</div>

<script>
// ── Global state ──────────────────────────────────────────────────────────────
let sessionId = null;
let rankingsData = null;  // last screen response
let ws = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let timerInterval = null;
let timerSeconds = 0;
let currentQuestion = null;
let questionIdx = 0;
let totalQuestions = 0;
let transcript = [];
let currentCandidateName = '';
let currentMatchScore = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(text, color) {
  document.getElementById('status-text').textContent = text;
  document.getElementById('status-dot').className = 'dot ' + (color || '');
}

function showPanel(id) {
  ['setup-panel','rankings-panel','interview-panel','report-panel'].forEach(p => {
    document.getElementById(p).style.display = p === id ? 'block' : 'none';
  });
}

function badge(category) {
  const map = {
    technical_depth:  ['Technical Depth', 'badge-tech'],
    gap_probing:      ['Gap Probing',     'badge-gap'],
    project_specific: ['Project',         'badge-proj'],
    system_design:    ['System Design',   'badge-sysdes'],
    behavioral:       ['Behavioral',      'badge-beh'],
  };
  return map[category] || [category, ''];
}

function scoreClass(s) { return s >= 70 ? 'score-high' : s >= 45 ? 'score-mid' : 'score-low'; }
function chipClass(s)  { return s >= 7  ? 'score-chip-high' : s >= 4 ? 'score-chip-mid' : 'score-chip-low'; }

function recBadge(rec) {
  const map = {
    STRONG_MATCH:  ['STRONG MATCH', 'rec-strong'],
    MATCH:         ['MATCH',        'rec-match'],
    WEAK_MATCH:    ['WEAK MATCH',   'rec-weak'],
    NOT_QUALIFIED: ['NOT QUALIFIED','rec-no'],
  };
  return map[rec] || [rec, 'rec-no'];
}

// ── Duplicate-email prompt ────────────────────────────────────────────────────
// Resolves to 'overwrite' | 'skip' | null (cancelled).
function askDuplicate(info) {
  return new Promise(resolve => {
    const cached = info.has_cached_ranking;
    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;background:rgba(0,0,0,0.65);
      display:flex;align-items:center;justify-content:center;z-index:9999;`;
    overlay.innerHTML = `
      <div style="background:#151823;border:1px solid #2a2f42;border-radius:12px;
                  padding:26px 28px;max-width:520px;width:90%;
                  box-shadow:0 18px 50px rgba(0,0,0,0.6);">
        <div style="font-size:1.05rem;font-weight:700;color:#e8e8ee;margin-bottom:6px;">
          This candidate is already in the pool
        </div>
        <div style="font-size:0.86rem;color:#9aa0b4;line-height:1.55;margin-bottom:18px;">
          <b style="color:#c8cde0;">${esc(info.name)}</b> &lt;${esc(info.email)}&gt; is already
          stored. Re-parsing the PDF costs about 30 seconds of model time;
          skipping reuses the resume already on file.
          ${cached
            ? `<br><span style="color:#5fa86a;">A score for this job description is already cached.</span>`
            : `<br><span style="color:#c9a227;">No score for this job description yet — it will still be evaluated.</span>`}
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          <button id="dup-skip" class="btn btn-primary btn-sm"
                  style="flex:1;min-width:180px;padding:9px 14px;">
            Skip re-parse — go to ranking
          </button>
          <button id="dup-over" class="btn btn-sm"
                  style="flex:1;min-width:180px;padding:9px 14px;background:#242838;
                         border:1px solid #3a4056;color:#c8cde0;">
            Re-parse &amp; overwrite (~30s)
          </button>
        </div>
        <button id="dup-cancel"
                style="margin-top:14px;background:none;border:none;color:#6c7288;
                       font-size:0.8rem;cursor:pointer;padding:0;">Cancel</button>
      </div>`;
    document.body.appendChild(overlay);
    const done = v => { overlay.remove(); resolve(v); };
    overlay.querySelector('#dup-skip').onclick   = () => done('skip');
    overlay.querySelector('#dup-over').onclick   = () => done('overwrite');
    overlay.querySelector('#dup-cancel').onclick = () => done(null);
  });
}

// ── Screening ─────────────────────────────────────────────────────────────────
async function startScreening() {
  const jd = document.getElementById('jd-input').value.trim();
  if (!jd) { alert('Please enter a job description.'); return; }

  const files = document.getElementById('resume-file').files;
  const fileCount = files.length;

  // Single upload only: check the email before spending anything on parsing.
  // Batch uploads always run the normal path for every file.
  let dedupMode = 'overwrite';
  if (fileCount === 1) {
    setStatus('Checking whether this candidate is already in the pool...', 'yellow');
    document.getElementById('screen-btn').disabled = true;
    try {
      const fd = new FormData();
      fd.append('resume', files[0]);
      fd.append('jd', jd);
      const r = await fetch('/api/check-duplicate', { method: 'POST', body: fd });
      const info = await r.json();
      if (r.ok && info.exists) {
        const choice = await askDuplicate(info);
        if (choice === null) {
          setStatus('Cancelled.', '');
          document.getElementById('screen-btn').disabled = false;
          return;
        }
        dedupMode = choice;
      }
    } catch (e) {
      // Pre-flight is an optimisation, never a gate — fall through to a
      // normal full parse if it fails for any reason.
      console.warn('duplicate pre-check failed, continuing:', e);
    }
  }

  const msg = fileCount > 0
    ? (dedupMode === 'skip'
        ? 'Reusing stored resume — running screening pipeline...'
        : `Parsing ${fileCount} resume${fileCount > 1 ? 's' : ''} + running screening pipeline...`)
    : 'Running screening pipeline...';
  setStatus(msg, 'yellow');
  document.getElementById('screen-btn').disabled = true;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) formData.append('resumes', files[i]);
  formData.append('jd', jd);
  formData.append('skills', document.getElementById('skills-input').value);
  formData.append('max_follow_ups', document.getElementById('followups-select').value);
  formData.append('dedup_mode', dedupMode);

  try {
    const resp = await fetch('/api/screen', { method: 'POST', body: formData });
    const data = await resp.json();

    document.getElementById('screen-btn').disabled = false;

    if (!resp.ok || data.error || !Array.isArray(data.rankings)) {
      // FastAPI errors use `detail`; the pipeline uses `error` + `detail`.
      const msg = data.error || data.detail || (resp.status + ' ' + resp.statusText);
      setStatus('Error: ' + msg, 'red');
      console.error('Screening failed:', resp.status, data);
      return;
    }

    rankingsData = data;
    showRankings(data);
    setStatus('Screening complete — ' + data.rankings.length + ' candidates evaluated', 'green');
  } catch (e) {
    setStatus('Error: ' + e.message, 'red');
    document.getElementById('screen-btn').disabled = false;
  }
}

// ── Rankings display ──────────────────────────────────────────────────────────
function showRankings(data) {
  showPanel('rankings-panel');
  const sessions  = data.sessions || {};
  const uploadSet = new Set(data.uploaded_cids || []);
  const recommended = Object.keys(sessions).length;
  const jobId = data.reused_job_id || data.job_id || '';

  document.getElementById('pool-stats').textContent =
    `Pool size: ${data.pool_size} candidates · Evaluated: ${data.rankings.length} · Recommended for interview: ${recommended}`;

  // ── Cached-run notice ──
  const notice = document.getElementById('cached-notice');
  if (data.reused_job_id) {
    const d = data.reused_job_date
      ? new Date(data.reused_job_date).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
      : 'previous run';
    document.getElementById('cached-notice-text').textContent =
      `From ${d} · only newly uploaded resumes re-evaluated`;
    const badge = document.getElementById('pool-change-badge');
    if (data.pool_changed) {
      badge.textContent = `Pool ${data.old_pool_size} → ${data.pool_size}`;
      badge.style.display = 'block';
    } else {
      badge.style.display = 'none';
    }
    notice.style.display = 'block';
  } else {
    notice.style.display = 'none';
  }

  // ── Uploaded candidates highlight section ──
  const uploadedWithSession = (data.rankings || []).filter(r =>
    uploadSet.has(r.candidate_id) && sessions[r.candidate_id]
  );
  const uploadedSection = document.getElementById('uploaded-section');
  if (uploadedWithSession.length > 0) {
    uploadedSection.style.display = '';
    const cards = document.getElementById('uploaded-cards');
    cards.innerHTML = '';
    uploadedWithSession.forEach(r => {
      const sess  = sessions[r.candidate_id];
      const sc    = r.match_score != null ? Math.round(r.match_score) : '?';
      const level = r.match_level || '—';
      const levelColor = level === 'Strong Fit' ? '#4caf50' : '#ff9800';
      const strengths = (r.strengths || []).slice(0, 2).join('; ') || '—';
      cards.innerHTML += `
        <div style="background:#0d1f14;border:1px solid #2a5c3a;border-radius:8px;
                    padding:14px 18px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
          <div style="flex:1;min-width:160px;">
            <div style="font-weight:700;font-size:0.95rem;">${esc(r.candidate_name || r.candidate_id)}</div>
            <div style="font-size:0.72rem;color:#666;">${esc(r.candidate_id)}</div>
          </div>
          <span class="score-pill ${scoreClass(sc)}">${sc}/100</span>
          <span style="color:${levelColor};font-weight:600;font-size:0.85rem;">${esc(level)}</span>
          <div style="font-size:0.8rem;color:#aaa;flex:2;min-width:140px;">${esc(strengths)}</div>
          <button class="btn btn-primary btn-sm"
            style="background:#1a7a3a;border-color:#2a9a4a;font-weight:700;white-space:nowrap;"
            onclick="startInterview('${sess.session_id}','${esc(r.candidate_name)}',${r.match_score||0},${sess.total_questions})">
            Interview Now →
          </button>
        </div>`;
    });
  } else {
    uploadedSection.style.display = 'none';
  }

  // ── Full talent pool table ──
  const tbody = document.getElementById('rankings-body');
  tbody.innerHTML = '';

  (data.rankings || []).forEach(r => {
    const sess    = sessions[r.candidate_id];
    const sc      = r.match_score != null ? Math.round(r.match_score) : '?';
    const level   = r.match_level || '—';
    const [recLabel, recCls] = recBadge(r.overall_recommendation);
    const strengths = (r.strengths || []).slice(0, 2).join('; ') || '—';
    const rankCls = r.rank <= 3 ? `rank-${r.rank}` : '';
    const scCls   = scoreClass(sc);
    const isNew   = uploadSet.has(r.candidate_id);

    let actionCell;
    if (sess) {
      const btnStyle = isNew ? 'background:#1a7a3a;border-color:#2a9a4a;font-weight:700;' : '';
      const btnLabel = isNew ? 'Interview Now →' : 'Interview';
      actionCell = `<button class="btn btn-primary btn-sm" style="${btnStyle}"
           onclick="startInterview('${sess.session_id}','${esc(r.candidate_name)}',${r.match_score||0},${sess.total_questions})">
           ${btnLabel}</button>`;
    } else if (jobId && (level === 'Strong Fit' || level === 'Good Fit')) {
      actionCell = `<button class="btn btn-primary btn-sm"
           onclick="prepareAndInterview('${r.candidate_id}','${jobId}',${r.match_score||0},'${esc(r.candidate_name)}',this)">
           Interview</button>`;
    } else {
      actionCell = `<span style="color:#555;font-size:0.8rem;">—</span>`;
    }

    const newBadge = isNew
      ? `<span style="font-size:0.65rem;font-weight:700;padding:2px 6px;border-radius:8px;
                      background:#1e3a5f;color:#5fa8ff;border:1px solid #3a7abf;margin-left:6px;">NEW</span>`
      : '';

    tbody.innerHTML += `
      <tr class="${rankCls}">
        <td><span class="rank-num">${r.rank}</span></td>
        <td>
          <div style="font-weight:600;">${esc(r.candidate_name || r.candidate_id)}${newBadge}</div>
          <div style="font-size:0.75rem;color:#666;">${esc(r.candidate_id)}</div>
        </td>
        <td><span class="score-pill ${scCls}">${sc}/100</span></td>
        <td><span class="rec-badge ${recCls}">${esc(level)}</span></td>
        <td style="font-size:0.8rem;color:#aaa;max-width:220px;">${esc(strengths)}</td>
        <td>${actionCell}</td>
      </tr>`;
  });
}

function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function backToSetup() {
  showPanel('setup-panel');
  fetchPoolCount();
  setStatus('Ready', '');
}

function backToRankings() {
  if (ws) { ws.close(); ws = null; }
  if (rankingsData) {
    showRankings(rankingsData);
    setStatus('Screening complete', 'green');
  } else {
    showPanel('setup-panel');
  }
  transcript = [];
}

// ── Lazy IQ: prepare interview on demand, then start ──────────────────────────
async function prepareAndInterview(candidateId, jobId, matchScore, candidateName, btn) {
  setStatus('Preparing interview questions...', 'yellow');
  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Preparing...';

  const fd = new FormData();
  fd.append('candidate_id', candidateId);
  fd.append('job_id', jobId);
  fd.append('match_score', matchScore);
  const fu = document.getElementById('followups-select');
  fd.append('max_follow_ups', fu ? fu.value : 1);

  try {
    const r = await fetch('/api/prepare-interview', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.error) {
      setStatus('Error: ' + d.error, 'red');
      btn.disabled = false;
      btn.textContent = orig;
      return;
    }
    startInterview(d.session_id, d.candidate_name || candidateName, matchScore, d.total_questions);
  } catch (e) {
    setStatus('Error: ' + e.message, 'red');
    btn.disabled = false;
    btn.textContent = orig;
  }
}

// ── Start interview for a specific candidate ──────────────────────────────────
function startInterview(sid, candidateName, matchScore, totalQ) {
  sessionId = sid;
  totalQuestions = totalQ;
  transcript = [];
  currentCandidateName = candidateName;
  currentMatchScore = matchScore;

  showPanel('interview-panel');
  document.getElementById('candidate-name').textContent = candidateName;
  document.getElementById('interview-meta').textContent =
    `Match score: ${Math.round(matchScore)}/100 · ${totalQ} questions`;

  // Reset recording UI
  document.getElementById('answer-section').style.display = 'none';
  document.getElementById('eval-section').style.display = 'none';
  document.getElementById('log-area').style.display = 'none';
  document.getElementById('log-entries').innerHTML = '';
  document.getElementById('submit-btn').disabled = true;
  document.getElementById('record-btn').innerHTML = '&#128308; Start Recording';
  document.getElementById('record-area').className = 'record-area';
  document.getElementById('tts-audio').style.display = 'none';
  audioChunks = [];

  connectWS();
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
  if (ws) { ws.close(); }
  ws = new WebSocket(`ws://${location.host}/ws/${sessionId}`);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'question')     handleQuestion(msg);
    if (msg.type === 'tts_audio')    playTTS(msg.audio_b64);
    if (msg.type === 'transcription')handleTranscription(msg.text);
    if (msg.type === 'evaluation')   handleEvaluation(msg);
    if (msg.type === 'report')       handleReport(msg);
    if (msg.type === 'status')       setStatus(msg.text, msg.color || '');
  };

  ws.onopen = () => {
    setStatus('Connected — interview starting', 'green');
    ws.send(JSON.stringify({ type: 'start' }));
  };

  ws.onerror = () => setStatus('Connection error', 'red');
}

// ── Question display ──────────────────────────────────────────────────────────
function handleQuestion(msg) {
  currentQuestion = msg;
  questionIdx = msg.idx;
  document.getElementById('q-counter').textContent = `${msg.idx + 1} / ${totalQuestions}`;
  document.getElementById('progress-fill').style.width = `${(msg.idx / totalQuestions) * 100}%`;

  const [label, cls] = badge(msg.category);
  const b = document.getElementById('q-badge');
  b.textContent = label; b.className = 'badge ' + cls;

  document.getElementById('q-gap').textContent = msg.gap ? `Gap: ${msg.gap}` : '';
  document.getElementById('question-text').textContent = msg.question;

  document.getElementById('answer-section').style.display = 'none';
  document.getElementById('eval-section').style.display = 'none';
  document.getElementById('brief-feedback-box').style.display = 'none';
  document.getElementById('submit-btn').disabled = true;
  document.getElementById('record-btn').innerHTML = '&#128308; Start Recording';
  document.getElementById('record-area').className = 'record-area';
  document.getElementById('mic-icon').textContent = '🎙️';
  document.getElementById('record-label').textContent = 'Click to start recording your answer';
  document.getElementById('timer').style.display = 'none';
  audioChunks = [];
  setStatus(`Question ${msg.idx + 1} of ${totalQuestions}`, 'green');
}

// ── TTS ───────────────────────────────────────────────────────────────────────
function playTTS(b64) {
  const audio = document.getElementById('tts-audio');
  audio.style.display = 'block';
  audio.src = 'data:audio/mpeg;base64,' + b64;
  audio.play().catch(() => {});
}

// ── Recording ─────────────────────────────────────────────────────────────────
async function toggleRecording() {
  if (isRecording) stopRecording(); else await startRecording();
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    audioChunks = [];

    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      document.getElementById('submit-btn').disabled = false;
    };

    mediaRecorder.start(250);
    isRecording = true;

    document.getElementById('record-btn').innerHTML = '&#9209; Stop Recording';
    document.getElementById('record-area').className = 'record-area recording';
    document.getElementById('mic-icon').textContent = '🔴';
    document.getElementById('record-label').textContent = 'Recording... speak your answer';
    document.getElementById('timer').style.display = 'block';

    timerSeconds = 0;
    timerInterval = setInterval(() => {
      timerSeconds++;
      const m = Math.floor(timerSeconds / 60);
      const s = String(timerSeconds % 60).padStart(2, '0');
      document.getElementById('timer').textContent = `${m}:${s}`;
    }, 1000);
  } catch (e) {
    setStatus('Mic access denied: ' + e.message, 'red');
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  isRecording = false;
  clearInterval(timerInterval);
  document.getElementById('record-btn').innerHTML = '&#128308; Record Again';
  document.getElementById('record-area').className = 'record-area done';
  document.getElementById('mic-icon').textContent = '✅';
  document.getElementById('record-label').textContent = 'Recording saved — submit when ready';
}

// ── Submit answer ─────────────────────────────────────────────────────────────
function encodeWav(audioBuffer) {
  const numChannels = 1;
  const sampleRate = audioBuffer.sampleRate;
  const samples = audioBuffer.getChannelData(0);
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  const dataLen = pcm.byteLength;
  const buf = new ArrayBuffer(44 + dataLen);
  const view = new DataView(buf);
  const write = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
  write(0, 'RIFF'); view.setUint32(4, 36 + dataLen, true);
  write(8, 'WAVE'); write(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); write(36, 'data');
  view.setUint32(40, dataLen, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], { type: 'audio/wav' });
}

async function submitAnswer() {
  if (!audioChunks.length) return;
  setStatus('Processing audio...', 'yellow');
  document.getElementById('submit-btn').disabled = true;
  document.getElementById('answer-section').style.display = 'block';
  document.getElementById('answer-text').textContent = 'Processing...';

  const webmBlob = new Blob(audioChunks, { type: 'audio/webm' });
  let uploadBlob = webmBlob;
  let uploadName = 'answer.webm';
  try {
    const arrayBuf = await webmBlob.arrayBuffer();
    const actx = new AudioContext();
    const audioBuf = await actx.decodeAudioData(arrayBuf);
    await actx.close();
    uploadBlob = encodeWav(audioBuf);
    uploadName = 'answer.wav';
  } catch (e) {
    console.warn('WAV conversion failed, sending raw webm:', e);
  }

  const durationSec = timerSeconds;
  const estSec = Math.max(10, Math.round(durationSec * 1.5));
  setStatus(`Uploading & transcribing (~${estSec}s for ${durationSec}s audio)...`, 'yellow');
  document.getElementById('answer-text').textContent = `Transcribing ${durationSec}s of audio — please wait...`;

  // Animated progress bar inside answer area
  const pWrap = document.getElementById('stt-progress-wrap');
  const pBar  = document.getElementById('stt-progress-bar');
  pWrap.style.display = 'block';
  pBar.style.width = '0%';
  let pct = 0;
  const pInterval = setInterval(() => {
    pct = Math.min(pct + (100 / estSec), 92);
    pBar.style.width = pct + '%';
  }, 1000);
  window._sttProgressInterval = pInterval;

  const formData = new FormData();
  formData.append('audio', uploadBlob, uploadName);
  formData.append('session_id', sessionId);
  formData.append('question_idx', questionIdx);

  let elapsed = 0;
  const pulse = setInterval(() => {
    elapsed += 5;
    setStatus(`Transcribing... (${elapsed}s elapsed, audio was ${durationSec}s)`, 'yellow');
  }, 5000);

  try {
    const resp = await fetch('/api/submit-answer', { method: 'POST', body: formData });
    clearInterval(pulse);
    const data = await resp.json();
    if (data.error) { setStatus('Error: ' + data.error, 'red'); return; }
  } catch (e) {
    clearInterval(pulse);
    setStatus('Error: ' + e.message, 'red');
  }
}

function handleTranscription(text) {
  clearInterval(window._sttProgressInterval);
  const pBar = document.getElementById('stt-progress-bar');
  pBar.style.width = '100%';
  setTimeout(() => {
    document.getElementById('stt-progress-wrap').style.display = 'none';
    pBar.style.width = '0%';
  }, 500);
  document.getElementById('answer-text').style.display = '';
  document.getElementById('raw-toggle-btn').textContent = 'Hide';
  document.getElementById('answer-text').textContent = text || '[No speech detected]';
  setStatus('Evaluating...', 'yellow');
}

function handleEvaluation(msg) {
  const ev = msg.evaluation;
  const cc = chipClass(ev.score);

  document.getElementById('eval-chips').innerHTML = `
    <div class="eval-chip ${cc}">Score <span>${ev.score}/10</span></div>
    <div class="eval-chip">Depth <span>${ev.depth}</span></div>
    ${ev.needs_follow_up ? '<div class="eval-chip" style="border-color:#f39c12;">Follow-up queued</div>' : ''}
  `;

  document.getElementById('hits-list').innerHTML = (ev.hits || []).map(h => `<li>${esc(h)}</li>`).join('');
  document.getElementById('misses-list').innerHTML = (ev.misses || []).map(m => `<li>${esc(m)}</li>`).join('');

  const fbBox = document.getElementById('brief-feedback-box');
  if (ev.brief_feedback) {
    document.getElementById('brief-feedback-text').textContent = ev.brief_feedback;
    fbBox.style.display = 'block';
  } else {
    fbBox.style.display = 'none';
  }

  document.getElementById('eval-section').style.display = 'block';

  transcript.push({ question: currentQuestion.question, answer: document.getElementById('answer-text').textContent, score: ev.score });
  updateLog();

  document.getElementById('next-btn').textContent = ev.needs_follow_up ? 'Answer Follow-up →' : 'Next Question →';
  setStatus(`Score: ${ev.score}/10 · ${ev.depth}`, 'green');
}

function updateLog() {
  if (!transcript.length) return;
  document.getElementById('log-area').style.display = 'block';
  document.getElementById('log-entries').innerHTML = transcript.map((t, i) => {
    const cls = t.score >= 7 ? 'good' : t.score >= 4 ? 'mid' : 'bad';
    return `<div class="log-entry">
      <div class="log-q">Q${i+1}: ${esc(t.question.slice(0,90))}...</div>
      <div class="log-a">${esc(t.answer.slice(0,120))}...</div>
      <div class="log-score"><span class="${cls}">Score: ${t.score}/10</span></div>
    </div>`;
  }).join('');
}

function nextQuestion() {
  if (!ws) return;
  ws.send(JSON.stringify({ type: 'next' }));
  document.getElementById('eval-section').style.display = 'none';
  document.getElementById('answer-section').style.display = 'none';
}

// ── Final report ──────────────────────────────────────────────────────────────
function handleReport(msg) {
  const r = msg.report;
  showPanel('report-panel');

  document.getElementById('report-score').textContent = r.overall_score?.toFixed(1);
  const v = document.getElementById('report-verdict');
  v.textContent = (r.verdict || '').replace('_', ' ');
  v.className = 'verdict verdict-' + r.verdict;
  document.getElementById('report-summary').textContent = r.summary;
  document.getElementById('report-rec').textContent = r.recommendation;

  const cats = r.category_scores || {};
  document.getElementById('cat-scores').innerHTML = Object.entries(cats).map(([k, val]) =>
    `<div class="cat-score"><div class="name">${k.replace('_',' ')}</div><div class="val">${val?.toFixed(1)}</div></div>`
  ).join('');

  document.getElementById('report-strengths').innerHTML = (r.top_strengths || []).map(s => `<li>${esc(s)}</li>`).join('');
  document.getElementById('report-concerns').innerHTML = (r.key_concerns || []).map(c => `<li>${esc(c)}</li>`).join('');

  document.getElementById('report-log').innerHTML = transcript.map((t, i) => {
    const cls = t.score >= 7 ? 'good' : t.score >= 4 ? 'mid' : 'bad';
    return `<div class="log-entry">
      <div class="log-q">Q${i+1}: ${esc(t.question)}</div>
      <div class="log-a">${esc(t.answer)}</div>
      <div class="log-score"><span class="${cls}">Score: ${t.score}/10</span></div>
    </div>`;
  }).join('');

  setStatus('Interview complete', 'green');
}

// ── Raw transcript toggle ─────────────────────────────────────────────────────
function toggleRawTranscript() {
  const el = document.getElementById('answer-text');
  const btn = document.getElementById('raw-toggle-btn');
  if (el.style.display === 'none') { el.style.display = ''; btn.textContent = 'Hide'; }
  else { el.style.display = 'none'; btn.textContent = 'Show'; }
}

// ── Report downloads ──────────────────────────────────────────────────────────
function downloadPDF() {
  if (sessionId) window.open(`/api/report/${sessionId}/pdf`, '_blank');
}

async function downloadJSON() {
  if (!sessionId) return;
  const r = await fetch(`/api/report/${sessionId}`);
  const data = await r.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(blob),
    download: `interview_report_${sessionId.slice(0, 8)}.json`
  });
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Restart interview ─────────────────────────────────────────────────────────
async function restartInterview() {
  if (!sessionId) return;
  if (!confirm('Restart this interview from question 1? Current progress will be cleared.')) return;
  const r = await fetch(`/api/session/${sessionId}/restart`, { method: 'POST' });
  const d = await r.json();
  if (d.error) { setStatus('Error: ' + d.error, 'red'); return; }
  startInterview(sessionId, currentCandidateName, currentMatchScore, d.total_questions);
}

// Spacebar shortcut for recording
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && document.getElementById('interview-panel').style.display !== 'none') {
    e.preventDefault();
    toggleRecording();
  }
});

// Show selected file count
document.getElementById('resume-file').addEventListener('change', function() {
  const n = this.files.length;
  const el = document.getElementById('file-count');
  el.textContent = n > 0 ? `${n} file${n > 1 ? 's' : ''} selected` : '';
});

// Load pool count + mongo status on page load
async function fetchPoolCount() {
  try {
    const r = await fetch('/api/pool-count');
    const d = await r.json();
    document.getElementById('pool-count').textContent = d.count;
    const ms = document.getElementById('mongo-status');
    if (d.mongo) {
      ms.textContent = 'connected';
      ms.style.color = '#27ae60';
      loadJobHistory();
    } else {
      ms.textContent = 'not running';
      ms.style.color = '#e74c3c';
    }
  } catch (e) {
    document.getElementById('pool-count').textContent = '?';
  }
}

async function loadJobHistory() {
  try {
    const r = await fetch('/api/jobs');
    const d = await r.json();
    if (!d.jobs || !d.jobs.length) return;
    document.getElementById('history-links').style.display = 'block';
    document.getElementById('job-history-list').innerHTML = d.jobs.slice(0, 5).map(j => {
      const date = new Date(j.run_at).toLocaleString();
      const jd = (j.job_description || '').slice(0, 40);
      return `<a href="#" style="color:#5fa8ff;margin-right:12px;"
        onclick="loadPastJob('${j._id}');return false;"
        title="${esc(j.job_description)}">${esc(jd)}… (${date})</a>`;
    }).join('');
  } catch (e) {}
}

async function loadPastJob(jobId) {
  setStatus('Loading past screening run...', 'yellow');
  try {
    const r = await fetch('/api/jobs/' + jobId);
    const data = await r.json();
    if (data.error) { setStatus('Error: ' + data.error, 'red'); return; }
    // Re-show rankings — no live sessions, but lazy Interview via job_id
    rankingsData = { rankings: data.rankings, sessions: {}, pool_size: data.pool_size, job_id: jobId };
    showRankings(rankingsData);
    setStatus('Past run loaded — ' + new Date(data.run_at).toLocaleString(), 'green');
  } catch (e) {
    setStatus('Error: ' + e.message, 'red');
  }
}

fetchPoolCount();
showPanel('setup-panel');
</script>
</body>
</html>
"""
