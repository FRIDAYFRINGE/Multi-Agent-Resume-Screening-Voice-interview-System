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

  .container { max-width: 860px; margin: 0 auto; padding: 32px 20px; }

  h1 { font-size: 1.6rem; font-weight: 600; color: #fff; margin-bottom: 4px; }
  .subtitle { font-size: 0.85rem; color: #888; margin-bottom: 32px; }

  /* Setup card */
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

  /* Interview panel */
  #interview-panel { display: none; }

  .progress-bar { background: #2a2d3e; border-radius: 4px; height: 4px; margin-bottom: 24px; }
  .progress-fill { height: 100%; border-radius: 4px; background: #5c6aff; transition: width 0.4s; }

  .question-meta { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; }
  .badge {
    font-size: 0.7rem; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .badge-tech { background: #1e3a5f; color: #5fa8ff; }
  .badge-gap  { background: #3a1e1e; color: #ff8080; }
  .badge-proj { background: #1e3a2a; color: #5fd48a; }
  .badge-sysdes { background: #2a1e3a; color: #c080ff; }
  .badge-beh  { background: #3a3a1e; color: #ffe080; }

  .question-box {
    background: #0f1117; border: 1px solid #2a2d3e; border-left: 3px solid #5c6aff;
    border-radius: 8px; padding: 18px 20px; font-size: 1rem; line-height: 1.6;
    margin-bottom: 20px; min-height: 70px;
  }

  .tts-controls { display: flex; gap: 10px; margin-bottom: 20px; }
  audio { width: 100%; border-radius: 8px; margin-bottom: 16px; }

  /* Recording */
  .record-area {
    border: 2px dashed #2a2d3e; border-radius: 12px;
    padding: 28px; text-align: center; margin-bottom: 20px;
    transition: border-color 0.2s;
  }
  .record-area.recording { border-color: #ff4f4f; background: #1a0f0f; }
  .record-area.done { border-color: #27ae60; background: #0f1a0f; }

  .mic-icon { font-size: 2.5rem; margin-bottom: 8px; }
  .record-hint { font-size: 0.8rem; color: #888; margin-top: 8px; }

  /* Timer */
  #timer { font-size: 1.4rem; font-weight: 700; color: #ff4f4f; font-variant-numeric: tabular-nums; }

  /* Transcript */
  .answer-box {
    background: #0f1117; border: 1px solid #2a2d3e; border-radius: 8px;
    padding: 14px 16px; min-height: 60px; font-size: 0.9rem; line-height: 1.5;
    margin-bottom: 16px; white-space: pre-wrap; color: #ccc;
  }

  /* Score */
  .eval-row { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
  .eval-chip {
    background: #1a1d27; border: 1px solid #2a2d3e; border-radius: 8px;
    padding: 8px 14px; font-size: 0.8rem;
  }
  .eval-chip span { font-weight: 700; color: #a0a8ff; }
  .score-high  { border-color: #27ae60; }
  .score-mid   { border-color: #f39c12; }
  .score-low   { border-color: #e74c3c; }

  .hits-misses { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }
  .hits-misses div { background: #0f1117; border: 1px solid #2a2d3e; border-radius: 8px; padding: 12px 14px; }
  .hits-misses h4 { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
  .hits-misses .hits h4 { color: #27ae60; }
  .hits-misses .misses h4 { color: #e74c3c; }
  .hits-misses li { font-size: 0.82rem; color: #aaa; margin-left: 14px; margin-bottom: 4px; }

  /* Report */
  #report-panel { display: none; }
  .report-header { text-align: center; margin-bottom: 28px; }
  .verdict {
    display: inline-block; font-size: 1.1rem; font-weight: 700;
    padding: 10px 28px; border-radius: 8px; margin-top: 12px;
  }
  .verdict-HIRE { background: #1a3a1a; color: #5fd48a; border: 1px solid #27ae60; }
  .verdict-STRONG_HIRE { background: #0f2a0f; color: #27ae60; border: 1px solid #27ae60; }
  .verdict-HOLD { background: #3a3a0f; color: #ffe080; border: 1px solid #f39c12; }
  .verdict-REJECT { background: #3a0f0f; color: #ff8080; border: 1px solid #e74c3c; }

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

  <!-- Setup panel -->
  <div id="setup-panel">
    <div class="card">
      <h2>Resume</h2>
      <label>Upload PDF</label>
      <input type="file" id="resume-file" accept=".pdf,.docx,.txt"
        style="background:#0f1117;border:1px solid #2a2d3e;border-radius:8px;padding:8px 12px;width:100%;color:#e0e0e0;">
    </div>

    <div class="card">
      <h2>Job Description</h2>
      <textarea id="jd-input" placeholder="Paste the full job description here...">Senior AI/ML Engineer - Agentic Systems
Design and build multi-agent LLM systems (LangGraph, CrewAI).
Implement RAG pipelines with vector databases (ChromaDB, FAISS).
Build production FastAPI backends. Work with LoRA/QLoRA fine-tuning
and RLHF. Deploy with Docker and CI/CD. Strong Python and PyTorch.
Required: 1+ year production AI experience.</textarea>

      <label>Required Skills (comma-separated)</label>
      <input type="text" id="skills-input" value="LangGraph, ChromaDB, FastAPI, RAG, PyTorch">

      <label>Max follow-ups per question</label>
      <select id="followups-select" style="background:#0f1117;border:1px solid #2a2d3e;border-radius:8px;padding:10px 14px;width:100%;color:#e0e0e0;">
        <option value="0">0 — No follow-ups</option>
        <option value="1" selected>1 — One follow-up if answer is shallow</option>
        <option value="2">2 — Up to two follow-ups</option>
      </select>

      <div style="margin-top:20px;">
        <button class="btn btn-primary" id="screen-btn" onclick="startScreening()">
          Run Screening Pipeline
        </button>
      </div>
    </div>
  </div>

  <!-- Interview panel -->
  <div id="interview-panel">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <div>
          <h2 id="candidate-name" style="margin-bottom:2px;"></h2>
          <div style="font-size:0.8rem;color:#888;" id="interview-meta"></div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.75rem;color:#888;">Question</div>
          <div style="font-size:1.1rem;font-weight:700;color:#a0a8ff;" id="q-counter">1 / ?</div>
        </div>
      </div>
      <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>

      <!-- Question -->
      <div class="question-meta">
        <span class="badge" id="q-badge">Technical</span>
        <span style="font-size:0.75rem;color:#888;" id="q-gap"></span>
      </div>
      <div class="question-box" id="question-text">Loading question...</div>

      <!-- TTS audio -->
      <audio id="tts-audio" controls autoplay style="display:none;"></audio>

      <!-- Recording controls -->
      <div class="record-area" id="record-area">
        <div class="mic-icon" id="mic-icon">🎙️</div>
        <div id="record-label">Click to start recording your answer</div>
        <div id="timer" style="display:none;">0:00</div>
        <div class="record-hint">Press the button or use spacebar</div>
      </div>

      <div style="display:flex;gap:10px;margin-bottom:16px;">
        <button class="btn btn-danger" id="record-btn" onclick="toggleRecording()">
          🔴 Start Recording
        </button>
        <button class="btn btn-success" id="submit-btn" onclick="submitAnswer()" disabled>
          Submit Answer →
        </button>
      </div>

      <!-- Answer transcription -->
      <div id="answer-section" style="display:none;">
        <label>Transcription</label>
        <div class="answer-box" id="answer-text"></div>

        <!-- Evaluation -->
        <div id="eval-section" style="display:none;">
          <div class="eval-row" id="eval-chips"></div>
          <div class="hits-misses">
            <div class="hits"><h4>Covered</h4><ul id="hits-list"></ul></div>
            <div class="misses"><h4>Missed</h4><ul id="misses-list"></ul></div>
          </div>
          <div style="display:flex;gap:10px;">
            <button class="btn btn-primary" id="next-btn" onclick="nextQuestion()">
              Next Question →
            </button>
          </div>
        </div>
      </div>

      <!-- Log -->
      <div id="log-area" style="display:none;">
        <div style="font-size:0.75rem;color:#888;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">
          Answered so far
        </div>
        <div class="log" id="log-entries"></div>
      </div>
    </div>
  </div>

  <!-- Report panel -->
  <div id="report-panel">
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
      <div>
        <div style="font-size:0.8rem;color:#888;margin-bottom:8px;">Full Transcript</div>
        <div class="log" id="report-log"></div>
      </div>
    </div>
  </div>

</div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let sessionId = null;
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

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(text, color) {
  document.getElementById('status-text').textContent = text;
  const dot = document.getElementById('status-dot');
  dot.className = 'dot ' + (color || '');
}

function badge(category) {
  const map = {
    technical_depth: ['Technical Depth', 'badge-tech'],
    gap_probing:     ['Gap Probing',     'badge-gap'],
    project_specific:['Project',         'badge-proj'],
    system_design:   ['System Design',   'badge-sysdes'],
    behavioral:      ['Behavioral',      'badge-beh'],
  };
  return map[category] || [category, ''];
}

// ── Screening ─────────────────────────────────────────────────────────────────
async function startScreening() {
  const file = document.getElementById('resume-file').files[0];
  if (!file) { alert('Please upload a resume file.'); return; }

  const jd = document.getElementById('jd-input').value.trim();
  if (!jd) { alert('Please enter a job description.'); return; }

  setStatus('Parsing resume...', 'yellow');
  document.getElementById('screen-btn').disabled = true;

  const formData = new FormData();
  formData.append('resume', file);
  formData.append('jd', jd);
  formData.append('skills', document.getElementById('skills-input').value);
  formData.append('max_follow_ups', document.getElementById('followups-select').value);

  try {
    const resp = await fetch('/api/screen', { method: 'POST', body: formData });
    const data = await resp.json();

    if (data.error) { setStatus('Error: ' + data.error, 'red'); return; }

    sessionId = data.session_id;
    setStatus('Screening complete — starting interview', 'green');

    // Switch to interview UI
    document.getElementById('setup-panel').style.display = 'none';
    document.getElementById('interview-panel').style.display = 'block';

    document.getElementById('candidate-name').textContent = data.candidate_name;
    document.getElementById('interview-meta').textContent =
      `Match score: ${data.match_score}/100 · ${data.total_questions} questions`;

    totalQuestions = data.total_questions;
    connectWS();
  } catch (e) {
    setStatus('Error: ' + e.message, 'red');
    document.getElementById('screen-btn').disabled = false;
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws/${sessionId}`);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'question')   handleQuestion(msg);
    if (msg.type === 'tts_audio')  playTTS(msg.audio_b64);
    if (msg.type === 'transcription') handleTranscription(msg.text);
    if (msg.type === 'evaluation') handleEvaluation(msg);
    if (msg.type === 'report')     handleReport(msg);
    if (msg.type === 'status')     setStatus(msg.text, msg.color || '');
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

  // Reset UI
  document.getElementById('answer-section').style.display = 'none';
  document.getElementById('eval-section').style.display = 'none';
  document.getElementById('submit-btn').disabled = true;
  document.getElementById('record-btn').textContent = '🔴 Start Recording';
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
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
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

    document.getElementById('record-btn').textContent = '⏹ Stop Recording';
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
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  isRecording = false;
  clearInterval(timerInterval);
  document.getElementById('record-btn').textContent = '🔴 Record Again';
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
  document.getElementById('answer-text').textContent = 'Transcribing...';

  // Convert WebM → WAV in browser so server can read it without ffmpeg
  const webmBlob = new Blob(audioChunks, { type: 'audio/webm' });
  let uploadBlob = webmBlob;
  let uploadName = 'answer.webm';
  try {
    const arrayBuf = await webmBlob.arrayBuffer();
    const actx = new AudioContext();  // native rate — Whisper resamples internally
    const audioBuf = await actx.decodeAudioData(arrayBuf);
    await actx.close();
    uploadBlob = encodeWav(audioBuf);
    uploadName = 'answer.wav';
  } catch (e) {
    console.warn('WAV conversion failed, sending raw webm:', e);
  }

  setStatus('Transcribing...', 'yellow');
  const formData = new FormData();
  formData.append('audio', uploadBlob, uploadName);
  formData.append('session_id', sessionId);
  formData.append('question_idx', questionIdx);

  const resp = await fetch('/api/submit-answer', { method: 'POST', body: formData });
  const data = await resp.json();

  if (data.error) { setStatus('Error: ' + data.error, 'red'); return; }
  // transcription + evaluation arrive via WS
}

function handleTranscription(text) {
  document.getElementById('answer-text').textContent = text || '[No speech detected]';
  setStatus('Evaluating...', 'yellow');
}

function handleEvaluation(msg) {
  const ev = msg.evaluation;
  const scoreClass = ev.score >= 7 ? 'score-high' : ev.score >= 4 ? 'score-mid' : 'score-low';

  document.getElementById('eval-chips').innerHTML = `
    <div class="eval-chip ${scoreClass}">Score <span>${ev.score}/10</span></div>
    <div class="eval-chip">Depth <span>${ev.depth}</span></div>
    ${ev.needs_follow_up ? '<div class="eval-chip" style="border-color:#f39c12;">Follow-up queued</div>' : ''}
  `;

  document.getElementById('hits-list').innerHTML = (ev.hits || []).map(h => `<li>${h}</li>`).join('');
  document.getElementById('misses-list').innerHTML = (ev.misses || []).map(m => `<li>${m}</li>`).join('');
  document.getElementById('eval-section').style.display = 'block';

  // Add to log
  transcript.push({ question: currentQuestion.question, answer: document.getElementById('answer-text').textContent, score: ev.score });
  updateLog();

  // Change next button label if follow-up
  document.getElementById('next-btn').textContent = ev.needs_follow_up ? 'Answer Follow-up →' : 'Next Question →';
  setStatus(`Score: ${ev.score}/10 · ${ev.depth}`, 'green');
}

function updateLog() {
  if (transcript.length === 0) return;
  document.getElementById('log-area').style.display = 'block';
  document.getElementById('log-entries').innerHTML = transcript.map((t, i) => {
    const cls = t.score >= 7 ? 'good' : t.score >= 4 ? 'mid' : 'bad';
    return `<div class="log-entry">
      <div class="log-q">Q${i+1}: ${t.question.slice(0,90)}...</div>
      <div class="log-a">${t.answer.slice(0,120)}...</div>
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
  document.getElementById('interview-panel').style.display = 'none';
  document.getElementById('report-panel').style.display = 'block';

  document.getElementById('report-score').textContent = r.overall_score?.toFixed(1);
  const v = document.getElementById('report-verdict');
  v.textContent = r.verdict?.replace('_', ' ');
  v.className = 'verdict verdict-' + r.verdict;
  document.getElementById('report-summary').textContent = r.summary;
  document.getElementById('report-rec').textContent = r.recommendation;

  const cats = r.category_scores || {};
  document.getElementById('cat-scores').innerHTML = Object.entries(cats).map(([k, v]) =>
    `<div class="cat-score"><div class="name">${k.replace('_',' ')}</div><div class="val">${v?.toFixed(1)}</div></div>`
  ).join('');

  document.getElementById('report-strengths').innerHTML = (r.top_strengths || []).map(s => `<li>${s}</li>`).join('');
  document.getElementById('report-concerns').innerHTML = (r.key_concerns || []).map(c => `<li>${c}</li>`).join('');

  document.getElementById('report-log').innerHTML = transcript.map((t, i) => {
    const cls = t.score >= 7 ? 'good' : t.score >= 4 ? 'mid' : 'bad';
    return `<div class="log-entry">
      <div class="log-q">Q${i+1}: ${t.question}</div>
      <div class="log-a">${t.answer}</div>
      <div class="log-score"><span class="${cls}">Score: ${t.score}/10</span></div>
    </div>`;
  }).join('');

  setStatus('Interview complete', 'green');
}

// Spacebar shortcut
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && document.getElementById('interview-panel').style.display !== 'none') {
    e.preventDefault();
    toggleRecording();
  }
});
</script>
</body>
</html>
"""
