import { WsClient } from './ws.js';

const RISK_MAP = {
  high: 'high',
  yüksek: 'high',
  yuksek: 'high',
  medium: 'medium',
  orta: 'medium',
  low: 'low',
  düşük: 'low',
  dusuk: 'low',
};

const RISK_LABEL = {
  high: 'Yüksek',
  medium: 'Orta',
  low: 'Düşük',
};

const gridEl = document.getElementById('camera-grid');
const notificationsEl = document.getElementById('notifications-list');
const actionsEl = document.getElementById('actions-list');
const selectedCameraEl = document.getElementById('selected-camera');
const modal = document.getElementById('modal');
const modalVideo = document.getElementById('modal-video');
const modalTitle = document.getElementById('modal-title');
const modalRisk = document.getElementById('modal-risk');
const modalProgress = document.getElementById('modal-progress');
const modalSummary = document.getElementById('modal-summary');
const modalEvents = document.getElementById('modal-events');
const modalActions = document.getElementById('modal-actions');
const toastEl = document.getElementById('toast');

const cameras = new Map();
let selectedCameraId = null;
let toastTimer = null;

function normalizeRisk(value) {
  if (!value) return 'low';
  const key = String(value).toLowerCase().replace(/\s+/g, '');
  for (const [k, v] of Object.entries(RISK_MAP)) {
    if (key.includes(k)) return v;
  }
  return 'low';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 3000);
}

function createCameraState(item) {
  return {
    camera_id: item.camera_id,
    label: item.active?.camera_label || item.camera_id,
    risk: normalizeRisk(item.active?.risk),
    video_file: item.active?.video_file || '',
    duration_sec: item.active?.duration_sec || 0,
    current_event: item.active?.current_event || '',
    progress_percent: item.active?.progress_percent || 0,
    summary: '',
    events: [],
    actions: [],
    riskSegment: null,
  };
}

function getOrCreateCamera(item) {
  if (cameras.has(item.camera_id)) {
    const existing = cameras.get(item.camera_id);
    return { ...existing, ...createCameraState(item) };
  }
  return createCameraState(item);
}

function renderCell(camera) {
  let cell = document.getElementById(`cell-${camera.camera_id}`);
  if (!cell) {
    cell = document.createElement('div');
    cell.id = `cell-${camera.camera_id}`;
    cell.className = 'camera-cell';
    cell.innerHTML = `
      <video autoplay muted loop></video>
      <div class="camera-label">
        <span class="label-text"></span>
        <span class="risk-badge">—</span>
      </div>
    `;
    cell.addEventListener('click', () => openModal(camera.camera_id));
    gridEl.appendChild(cell);
  }

  const video = cell.querySelector('video');
  const labelText = cell.querySelector('.label-text');
  const badge = cell.querySelector('.risk-badge');

  video.src = `/api/v1/pseudolive/videos/${camera.camera_id}`;
  labelText.textContent = `${camera.label} (${camera.camera_id})`;
  badge.textContent = RISK_LABEL[camera.risk] || '—';
  badge.className = `risk-badge ${camera.risk}`;

  cell.classList.remove('risk-high', 'risk-medium', 'risk-low');
  cell.classList.add(`risk-${camera.risk}`);

  return cell;
}

async function loadCameras() {
  try {
    const res = await fetch('/api/v1/pseudolive/cameras');
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    const ids = new Set();
    for (const item of data) {
      ids.add(item.camera_id);
      const camera = getOrCreateCamera(item);
      cameras.set(camera.camera_id, camera);
      renderCell(camera);
    }
    for (const [id, camera] of cameras) {
      if (!ids.has(id)) {
        const cell = document.getElementById(`cell-${id}`);
        if (cell) cell.remove();
        cameras.delete(id);
      }
    }
  } catch (err) {
    console.error('Kameralar yüklenemedi:', err);
  }
}

function setTemporaryRisk(cameraId, risk) {
  const camera = cameras.get(cameraId);
  if (!camera) return;
  camera.risk = normalizeRisk(risk);
  renderCell(camera);
  setTimeout(() => {
    const current = cameras.get(cameraId);
    if (!current) return;
    current.risk = 'low';
    renderCell(current);
  }, 6000);
}

function addNotification(stream, data) {
  const empty = notificationsEl.querySelector('.empty-state');
  if (empty) empty.remove();
  const card = document.createElement('div');
  card.className = 'notification-card';
  const title = data.headline || data.current_event || stream;
  card.innerHTML = `
    <div><strong>${escapeHtml(stream)}</strong> — ${escapeHtml(title)}</div>
    <div class="meta">Kamera: ${escapeHtml(data.camera_id || '-')}</div>
  `;
  notificationsEl.prepend(card);
  if (notificationsEl.children.length > 50) {
    notificationsEl.lastElementChild.remove();
  }
}

function renderActionsPanel() {
  actionsEl.innerHTML = '';
  if (!selectedCameraId || !cameras.has(selectedCameraId)) {
    selectedCameraEl.textContent = 'Kamera seçilmedi.';
    return;
  }
  const camera = cameras.get(selectedCameraId);
  selectedCameraEl.textContent = `Seçili: ${camera.label} (${camera.camera_id})`;

  const notifyBtn = document.createElement('button');
  notifyBtn.className = 'btn btn-danger';
  notifyBtn.textContent = 'Saha Ekibine Bildirim Gönder';
  notifyBtn.addEventListener('click', () => sendFieldAlert(camera));
  actionsEl.appendChild(notifyBtn);

  for (const action of camera.actions || []) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-secondary';
    btn.textContent = action;
    btn.addEventListener('click', () => showToast(`Tetiklendi: ${action}`));
    actionsEl.appendChild(btn);
  }
}

async function sendFieldAlert(camera) {
  const riskLabel = RISK_LABEL[camera.risk] || 'Orta';
  const body = {
    camera_id: camera.camera_id,
    risk: riskLabel,
    headline: camera.current_event || 'Riskli Durum',
    summary: camera.summary || 'Süpervizör tarafından bildirim gönderildi.',
    actions: camera.actions.length ? camera.actions : ['İncele', 'Müdahale Et'],
    risk_segment: camera.riskSegment || { start_sec: 0, end_sec: 10, event_type: camera.current_event || 'genel' },
    target_roles: ['sağlık', 'temizlik', 'teknisyen'],
  };
  try {
    const res = await fetch('/api/v1/field-alerts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(res.statusText);
    showToast('Saha ekibine bildirim gönderildi');
  } catch (err) {
    showToast('Bildirim gönderilemedi: ' + err.message);
  }
}

function openModal(cameraId) {
  selectedCameraId = cameraId;
  renderActionsPanel();
  const camera = cameras.get(cameraId);
  if (!camera) return;
  modalTitle.textContent = `${camera.label} (${camera.camera_id})`;
  modalVideo.src = `/api/v1/pseudolive/videos/${camera.camera_id}`;
  modalRisk.textContent = RISK_LABEL[camera.risk] || '—';
  modalRisk.className = `risk-badge ${camera.risk}`;
  modalProgress.textContent = `İlerleme: ${camera.progress_percent}% • Süre: ${camera.duration_sec}s`;
  modalSummary.textContent = camera.summary || 'Henüz özet yok.';
  modalEvents.innerHTML = camera.events.length
    ? camera.events.map((e) => `<li>${escapeHtml(e)}</li>`).join('')
    : '<li class="empty-state">Henüz olay kaydı yok.</li>';
  modalActions.innerHTML = '';
  if (camera.actions.length) {
    for (const action of camera.actions) {
      const btn = document.createElement('button');
      btn.className = 'btn btn-secondary';
      btn.textContent = action;
      btn.addEventListener('click', () => showToast(`Tetiklendi: ${action}`));
      modalActions.appendChild(btn);
    }
  } else {
    modalActions.innerHTML = '<span class="meta">Eylem yok.</span>';
  }
  modal.classList.add('open');
}

function closeModal() {
  modal.classList.remove('open');
  modalVideo.pause();
  modalVideo.src = '';
}

function handleWsMessage(msg) {
  const { stream, data } = msg || {};
  if (!data || !data.camera_id) return;

  addNotification(stream, data);

  if (stream === 'event.detected') {
    const camera = cameras.get(data.camera_id);
    if (camera) {
      camera.current_event = data.event_type || data.current_event || camera.current_event;
      camera.events.unshift(data.event_type || JSON.stringify(data));
      if (camera.events.length > 20) camera.events.pop();
      if (data.risk) setTemporaryRisk(data.camera_id, data.risk);
    }
  } else if (stream === 'decision.final') {
    const camera = cameras.get(data.camera_id);
    if (camera) {
      camera.summary = data.summary || camera.summary;
      camera.actions = Array.isArray(data.actions) ? data.actions : camera.actions;
      camera.current_event = data.current_event || camera.current_event;
      if (data.risk_segment) camera.riskSegment = data.risk_segment;
      if (data.risk) setTemporaryRisk(data.camera_id, data.risk);
    }
  } else if (stream === 'notification.push') {
    if (data.risk) setTemporaryRisk(data.camera_id, data.risk);
  }
  if (selectedCameraId === data.camera_id && modal.classList.contains('open')) {
    openModal(selectedCameraId);
  }
}

function init() {
  loadCameras();
  setInterval(loadCameras, 5000);

  const ws = new WsClient(`ws://${location.host}/ws`);
  ws.connect();
  ws.onMessage(handleWsMessage);

  document.getElementById('modal-close').addEventListener('click', closeModal);
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });
}

init();
