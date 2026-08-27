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
let pollInterval = null;
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

function stopPolling() {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}

function startWatching(jobId) {
  if (wsClient) {
    wsClient.close();
    wsClient = null;
  }
  stopPolling();

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  wsClient = new WsClient(`${proto}://${location.host}/ws`);
  wsClient.connect();
  wsClient.onMessage(handleWsMessage);

  // Güvenlik ağı: decision.final kaçırılırsa polling ile sonucu çek.
  // jobId kapanışta tutulur; önceki bir upload'ın zamanlayıcısı yeni işi
  // yanlışlıkla tamamlayamaz.
  pollInterval = setInterval(async () => {
    if (decisionPayload || currentJobId !== jobId) {
      stopPolling();
      return;
    }
    try {
      const analysis = await api(`/analyses/${jobId}`);
      if (analysis) {
        const response = await api(`/analyses/${jobId}/events`);
        finalize(analysis, response.events || []);
        stopPolling();
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

  // WebSocket kaçırılırsa REST yedeğinden gelen olayları da overlay ve
  // zaman çizelgesinin kullandığı kanonik koleksiyona al.
  currentEvents = Array.isArray(events) ? events : [];

  // Yerel video kaynağı (upload edilen dosya)
  if (uploadedFile) {
    el.video.src = URL.createObjectURL(uploadedFile);
    el.video.load();
    el.video.play().catch(e => console.log('Oynatma hatası:', e));
  }
  el.videoSection.hidden = false;
  el.decisionSummary.hidden = false;
  el.demoStatus.textContent = 'Analiz tamamlandı';

  renderDecision(analysis);
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
// Başlatma
// ---------------------------------------------------------------------------

function init() {
  initUpload();

  el.video.addEventListener('timeupdate', drawOverlay);
  el.video.addEventListener('resize', drawOverlay);
  window.addEventListener('resize', drawOverlay);
}

init();
