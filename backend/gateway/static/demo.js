import { WsClient } from './ws.js';
import {
  API, api, escapeHtml, mmss, riskClass, showToast, initToast,
} from './app.js';

// OpenCV BGR tuple'ları → CSS rgb(...). BGR'de ilk değer Mavi, son Kırmızı.
const CLASS_COLORS = {
  arac: 'rgb(0,140,255)',
  insan: 'rgb(127,255,0)',
  palet: 'rgb(255,215,0)',
  baret: 'rgb(0,191,255)',
  yelek: 'rgb(50,205,50)',
  yangin: 'rgb(255,0,0)',
  duman: 'rgb(180,180,180)',
  unknown: 'rgb(200,200,200)',
};

const RELATION_COLORS = {
  near: 'rgb(255,70,0)',
  wearing: 'rgb(0,255,0)',
  carrying: 'rgb(0,255,255)',
};

const STEP_ORDER = ['ingest', 'perception', 'vlm', 'decision'];

const el = {
  dropZone: document.getElementById('drop-zone'),
  uploadSection: document.getElementById('upload-section'),
  videoInput: document.getElementById('video-input'),
  progressSection: document.getElementById('progress-section'),
  progressFill: document.getElementById('progress-fill'),
  progressSteps: document.getElementById('progress-steps'),
  videoSection: document.getElementById('video-section'),
  video: document.getElementById('demo-video'),
  canvas: document.getElementById('overlay-canvas'),
  videoMeta: document.getElementById('video-meta'),
  decisionSummary: document.getElementById('decision-summary'),
  summaryRisk: document.getElementById('summary-risk'),
  summaryConfidence: document.getElementById('summary-confidence'),
  summaryText: document.getElementById('summary-text'),
  summaryReasoning: document.getElementById('summary-reasoning'),
  summaryActions: document.getElementById('summary-actions'),
  detailAnalysis: document.getElementById('detail-analysis'),
  tabButtons: document.querySelectorAll('.tab-btn'),
  tabContents: document.querySelectorAll('.tab-content'),
  yoloList: document.getElementById('yolo-list'),
  vlmSummary: document.getElementById('vlm-summary'),
  vlmFlags: document.getElementById('vlm-flags'),
  ragList: document.getElementById('rag-list'),
  eventTimeline: document.getElementById('event-timeline'),
  demoStatus: document.getElementById('demo-status'),
  toast: document.getElementById('toast'),
};
initToast(el.toast);

let currentJobId = null;
let uploadedFile = null;
let currentEvents = [];
let vlmInterpretation = null;
let decisionPayload = null;
let wsClient = null;
let completedSteps = new Set();

// ---------------------------------------------------------------------------
// MM:SS ↔ saniye dönüşümleri
// ---------------------------------------------------------------------------

function parseMmss(ts) {
  if (typeof ts !== 'string' || !ts.includes(':')) return 0;
  const [m, s] = ts.split(':');
  return (parseInt(m, 10) || 0) * 60 + (parseInt(s, 10) || 0);
}

function formatConfidence(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return `%${(n * 100).toFixed(0)}`;
}

// ---------------------------------------------------------------------------
// Upload + sürükle-bırak
// ---------------------------------------------------------------------------

function initUpload() {
  el.dropZone.addEventListener('click', () => el.videoInput.click());

  el.videoInput.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) startUpload(file);
  });

  ['dragenter', 'dragover'].forEach((name) => {
    el.uploadSection.addEventListener(name, (e) => {
      e.preventDefault();
      el.uploadSection.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach((name) => {
    el.uploadSection.addEventListener(name, (e) => {
      e.preventDefault();
      el.uploadSection.classList.remove('drag-over');
    });
  });

  el.uploadSection.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) startUpload(file);
    else if (file) showToast('Lütfen bir video dosyası yükleyin.', true);
  });
}

async function startUpload(file) {
  uploadedFile = file;
  currentJobId = null;
  currentEvents = [];
  vlmInterpretation = null;
  decisionPayload = null;
  completedSteps = new Set();
  resetUI();

  el.demoStatus.textContent = 'Yükleniyor…';
  const formData = new FormData();
  formData.append('video', file);
  formData.append('camera_id', 'demo_upload');

  try {
    const res = await fetch(`${API}/analyses/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch { /* ignore */ }
      throw new Error(detail);
    }
    const data = await res.json();
    currentJobId = data.job_id;
    el.demoStatus.textContent = `Analiz başlatıldı: ${currentJobId.slice(0, 8)}`;
    showToast('Video yüklendi; analiz başlatıldı.');
    el.progressSection.hidden = false;
    startWatching(currentJobId);
  } catch (err) {
    el.demoStatus.textContent = 'Yükleme başarısız';
    showToast(`Yükleme hatası: ${err.message}`, true);
  }
}

function resetUI() {
  el.progressFill.style.width = '0%';
  el.progressSection.hidden = true;
  el.videoSection.hidden = true;
  el.decisionSummary.hidden = true;
  el.detailAnalysis.hidden = true;
  el.canvas.getContext('2d')?.clearRect(0, 0, el.canvas.width, el.canvas.height);
  el.yoloList.innerHTML = '';
  el.vlmSummary.textContent = '';
  el.vlmFlags.innerHTML = '';
  el.ragList.innerHTML = '';
  el.eventTimeline.innerHTML = '';
  el.summaryActions.innerHTML = '';
  el.summaryText.textContent = '';
  el.summaryReasoning.textContent = '';
  el.summaryRisk.textContent = '—';
  el.summaryRisk.className = 'risk-badge';
  el.summaryConfidence.textContent = 'Güven: —';
  el.tabButtons.forEach((btn) => btn.classList.remove('active'));
  el.tabButtons[0]?.classList.add('active');
  el.tabContents.forEach((tc) => tc.classList.remove('active'));
  el.tabContents[0]?.classList.add('active');
  updateStepUI();
}

// ---------------------------------------------------------------------------
// Pipeline progress + WebSocket
// ---------------------------------------------------------------------------

function updateStepUI() {
  const total = STEP_ORDER.length;
  const done = Array.from(completedSteps);
  const progress = Math.min(100, Math.round((done.length / total) * 100));
  el.progressFill.style.width = `${progress}%`;

  el.progressSteps.querySelectorAll('span').forEach((span) => {
    const step = span.dataset.step;
    span.classList.remove('done', 'active');
    if (completedSteps.has(step)) {
      span.classList.add('done');
      span.textContent = span.textContent.replace(/^⏳\s*/, '✓ ');
    } else if (step === nextPendingStep()) {
      span.classList.add('active');
      if (!span.textContent.startsWith('⏳') && !span.textContent.startsWith('✓')) {
        span.textContent = '⏳ ' + span.textContent.replace(/^[✓⏳]\s*/, '');
      }
    }
  });
}

function nextPendingStep() {
  return STEP_ORDER.find((s) => !completedSteps.has(s));
}

function startWatching(jobId) {
  if (wsClient) {
    wsClient.disconnect();
  }

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  wsClient = new WsClient(`${proto}://${location.host}/ws`);
  wsClient.connect();
  wsClient.onMessage(handleWsMessage);

  // Güvenlik ağı: decision.final kaçırılırsa polling ile sonucu çek
  const pollInterval = setInterval(async () => {
    if (decisionPayload || !currentJobId) {
      clearInterval(pollInterval);
      return;
    }
    try {
      const analysis = await api(`/analyses/${currentJobId}`);
      if (analysis) {
        const events = await api(`/analyses/${currentJobId}/events`);
        finalize(analysis, events || []);
        clearInterval(pollInterval);
      }
    } catch { /* henüz hazır değil */ }
  }, 2500);
}

function handleWsMessage(msg) {
  const { stream, data } = msg || {};
  if (!data || data.job_id !== currentJobId) return;

  if (stream === 'frame.chunk') {
    completedSteps.add('ingest');
    updateStepUI();
  } else if (stream === 'event.detected') {
    completedSteps.add('ingest');
    completedSteps.add('perception');
    currentEvents.push({ stream, data });
    updateStepUI();
  } else if (stream === 'vlm.interpreted') {
    completedSteps.add('ingest');
    completedSteps.add('perception');
    completedSteps.add('vlm');
    vlmInterpretation = data.interpretation || {};
    updateStepUI();
  } else if (stream === 'decision.final') {
    completedSteps.add('ingest');
    completedSteps.add('perception');
    completedSteps.add('vlm');
    completedSteps.add('decision');
    decisionPayload = data;
    updateStepUI();
    // Events tablosundan tüm kayıtları çek ve ekranı doldur
    fetchAndFinalize();
  }
}

async function fetchAndFinalize() {
  if (!currentJobId) return;
  try {
    const [analysis, events] = await Promise.all([
      api(`/analyses/${currentJobId}`),
      api(`/analyses/${currentJobId}/events`).catch(() => ({ events: [] })),
    ]);
    finalize(analysis || decisionPayload, events.events || []);
  } catch (err) {
    showToast(`Sonuçlar alınamadı: ${err.message}`, true);
  }
}

// ---------------------------------------------------------------------------
// Nihai ekranı doldur
// ---------------------------------------------------------------------------

function finalize(analysis, events) {
  if (!analysis) return;

  // Yerel video kaynağı (upload edilen dosya)
  if (uploadedFile) {
    el.video.src = URL.createObjectURL(uploadedFile);
  }
  el.videoSection.hidden = false;
  el.decisionSummary.hidden = false;
  el.detailAnalysis.hidden = false;
  el.demoStatus.textContent = 'Analiz tamamlandı';

  renderDecision(analysis);
  renderEvents(events);
  renderYolo(events);
  renderVlm(events, analysis);
  renderRag(analysis);
}

function renderDecision(analysis) {
  const risk = analysis.risk || '—';
  el.summaryRisk.textContent = risk;
  el.summaryRisk.className = `risk-badge ${riskClass(risk)}`;
  el.summaryConfidence.textContent = `Güven: ${formatConfidence(analysis.confidence)}`;
  el.summaryText.textContent = analysis.summary || 'Özet üretilmedi.';
  el.summaryReasoning.textContent = analysis.reasoning || 'Gerekçe kaydı yok.';

  const actions = analysis.actions || [];
  el.summaryActions.innerHTML = actions.length
    ? actions.map((a) => `<li>${escapeHtml(a)}</li>`).join('')
    : '<li class="meta">Aksiyon önerisi yok.</li>';
}

function renderEvents(events) {
  const eventRows = events
    .filter((e) => e.stream === 'event.detected')
    .map((e) => e.data);

  el.eventTimeline.innerHTML = eventRows.length
    ? eventRows.map((ev) => `
        <li class="timeline-item severity-${ev.severity || 'low'}">
          <span class="timeline-time">${escapeHtml(ev.timestamp || '00:00')}</span>
          <span class="timeline-type">${escapeHtml(ev.event_type)}</span>
          <span class="timeline-sev ${ev.severity || 'low'}">${escapeHtml(ev.severity || 'Belirsiz')}</span>
          <span class="meta">${escapeHtml(ev.description || '')}</span>
          <span class="meta">Güven: ${formatConfidence(ev.confidence)}</span>
        </li>`).join('')
    : '<li class="empty-state">Tespit edilen olay yok.</li>';
}

function renderYolo(events) {
  const snapshots = events
    .filter((e) => e.stream === 'event.detected')
    .map((e) => e.data)
    .filter((ev) => ev.snapshot?.detections?.length);

  // Tüm unique tespit sınıflarını ve sayılarını topla
  const counts = {};
  const examples = [];
  snapshots.forEach((ev) => {
    ev.snapshot.detections.forEach((det) => {
      const cls = det.class || 'unknown';
      counts[cls] = (counts[cls] || 0) + 1;
      if (examples.length < 12) {
        examples.push({
          cls,
          confidence: det.confidence,
          trackId: det.track_id,
          timestamp: ev.timestamp,
        });
      }
    });
  });

  let html = '';
  if (Object.keys(counts).length) {
    html += `<li class="timeline-item"><span class="timeline-time">SINIF</span><span class="timeline-type">Adet</span><span class="timeline-sev low">Renk</span><span class="meta">Örnek track</span></li>`;
    Object.entries(counts).forEach(([cls, count]) => {
      const color = CLASS_COLORS[cls] || CLASS_COLORS.unknown;
      html += `
        <li class="timeline-item">
          <span class="timeline-type">${escapeHtml(cls.toUpperCase())}</span>
          <span class="timeline-time">${count}</span>
          <span class="timeline-sev low" style="background:${color};color:#000">${escapeHtml(cls.toUpperCase())}</span>
          <span class="meta">YOLO tespiti</span>
        </li>`;
    });
    html += `<li class="timeline-item" style="margin-top:0.6rem"><span class="timeline-time">ZAMAN</span><span class="timeline-type">NESNE</span><span class="timeline-sev low">GÜVEN</span><span class="meta">TRACK ID</span></li>`;
    examples.forEach((ex) => {
      html += `
        <li class="timeline-item">
          <span class="timeline-time">${escapeHtml(ex.timestamp)}</span>
          <span class="timeline-type">${escapeHtml(ex.cls.toUpperCase())}</span>
          <span class="timeline-sev low">${formatConfidence(ex.confidence)}</span>
          <span class="meta">#${ex.trackId ?? '—'}</span>
        </li>`;
    });
  } else {
    html = '<li class="empty-state">YOLO snapshot kaydı yok.</li>';
  }
  el.yoloList.innerHTML = html;
}

function renderVlm(events, analysis) {
  // VLM yorumunu event.detected / decision.final öncesinde gelen vlm.interpreted'dan al
  const vlmEvents = events.filter((e) => e.stream === 'vlm.interpreted').map((e) => e.data);
  const interp = vlmEvents[0]?.interpretation || vlmInterpretation || analysis.vlm_interpretation || {};

  const summary = interp.summary_tr || interp.summary || interp.scene_description || '';
  el.vlmSummary.textContent = summary || 'Kanal B yorumu alınamadı.';

  const flags = interp.risk_flags_tr || interp.risk_flags || [];
  if (flags.length) {
    el.vlmFlags.innerHTML = flags
      .map((f) => `<li class="timeline-item severity-medium"><span class="timeline-type">⚠ ${escapeHtml(f)}</span></li>`)
      .join('');
  } else if (interp.risk_events?.length) {
    el.vlmFlags.innerHTML = interp.risk_events
      .map((r) => `<li class="timeline-item severity-medium"><span class="timeline-type">⚠ ${escapeHtml(r.description_tr || r.description || '')}</span></li>`)
      .join('');
  } else {
    el.vlmFlags.innerHTML = '<li class="empty-state">Risk bayrağı üretilmedi.</li>';
  }
}

function renderRag(analysis) {
  const results = analysis.results || [];
  const ragRows = results.filter((r) => r?.evidence?.rag?.supports);

  if (!ragRows.length) {
    el.ragList.innerHTML = '<li class="empty-state">RAG eşleşmesi bulunamadı.</li>';
    return;
  }

  el.ragList.innerHTML = ragRows.map((r) => {
    const obs = (r.evidence.rag.observations || []).map((o) => `<span class="meta">• ${escapeHtml(o)}</span>`).join('');
    return `
      <li class="timeline-item severity-${r.severity || 'low'}">
        <span class="timeline-time">${escapeHtml(r.time || '00:00')}</span>
        <span class="timeline-type">${escapeHtml(r.event_type || 'RAG')}</span>
        <span class="timeline-sev ${r.severity || 'low'}">${escapeHtml(r.severity || 'Düşük')}</span>
        <span class="meta">${escapeHtml(r.hazard_mechanism || r.event || '')}</span>
        ${obs ? `<div style="grid-column:1/-1">${obs}</div>` : ''}
      </li>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Canvas overlay
// ---------------------------------------------------------------------------

function resizeCanvas() {
  if (!el.video.videoWidth) return;
  const { clientWidth, clientHeight } = el.video;
  if (el.canvas.width !== clientWidth || el.canvas.height !== clientHeight) {
    el.canvas.width = clientWidth;
    el.canvas.height = clientHeight;
  }
}

function findNearestSnapshot(timeSec) {
  const events = currentEvents
    .filter((e) => e.stream === 'event.detected')
    .map((e) => e.data)
    .filter((ev) => ev.snapshot?.detections?.length);

  let best = null;
  let bestDelta = Infinity;
  events.forEach((ev) => {
    const evSec = parseMmss(ev.timestamp);
    const delta = Math.abs(evSec - timeSec);
    if (delta < bestDelta) {
      bestDelta = delta;
      best = ev.snapshot;
    }
  });
  return bestDelta <= 1.5 ? best : null;
}

function drawOverlay() {
  resizeCanvas();
  const ctx = el.canvas.getContext('2d');
  ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);

  if (!el.video.videoWidth || !el.video.videoHeight) return;

  const snapshot = findNearestSnapshot(el.video.currentTime);
  if (!snapshot) return;

  const scaleX = el.canvas.width / el.video.videoWidth;
  const scaleY = el.canvas.height / el.video.videoHeight;

  // Sahne grafiği çizgileri
  const centers = {};
  (snapshot.detections || []).forEach((det) => {
    const [x1, y1, x2, y2] = det.bbox || [0, 0, 0, 0];
    centers[`${det.class}_${det.track_id}`] = {
      x: ((x1 + x2) / 2) * scaleX,
      y: ((y1 + y2) / 2) * scaleY,
    };
  });

  (snapshot.scene_graph?.edges || []).forEach((edge) => {
    const src = centers[edge.source];
    const tgt = centers[edge.target];
    if (!src || !tgt) return;
    ctx.beginPath();
    ctx.moveTo(src.x, src.y);
    ctx.lineTo(tgt.x, tgt.y);
    ctx.strokeStyle = RELATION_COLORS[edge.relation] || RELATION_COLORS.near;
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  // Bounding box'lar
  (snapshot.detections || []).forEach((det) => {
    const [x1, y1, x2, y2] = det.bbox || [0, 0, 0, 0];
    const sx1 = x1 * scaleX;
    const sy1 = y1 * scaleY;
    const sx2 = x2 * scaleX;
    const sy2 = y2 * scaleY;
    const color = CLASS_COLORS[det.class] || CLASS_COLORS.unknown;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

    const label = `${(det.class || 'unknown').toUpperCase()} ${formatConfidence(det.confidence)}`;
    ctx.font = '12px system-ui, sans-serif';
    const tm = ctx.measureText(label);
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(sx1, Math.max(0, sy1 - 16), tm.width + 8, 16);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, sx1 + 4, Math.max(12, sy1 - 4));
  });
}

// ---------------------------------------------------------------------------
// Tab geçişleri
// ---------------------------------------------------------------------------

function initTabs() {
  el.tabButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      el.tabButtons.forEach((b) => b.classList.remove('active'));
      el.tabContents.forEach((c) => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`tab-${target}`)?.classList.add('active');
    });
  });
}

// ---------------------------------------------------------------------------
// Başlatma
// ---------------------------------------------------------------------------

function init() {
  initUpload();
  initTabs();

  el.video.addEventListener('timeupdate', drawOverlay);
  el.video.addEventListener('resize', drawOverlay);
  window.addEventListener('resize', drawOverlay);
}

init();
