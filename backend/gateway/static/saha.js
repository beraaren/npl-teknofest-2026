import { WsClient } from './ws.js';

const containerEl = document.getElementById('alarms-container');
const selectorEl = document.getElementById('role-selector');
const toastEl = document.getElementById('toast');

let currentRole = '';
let toastTimer = null;

const ROLE_LABELS = {
  '': 'Tümü',
  temizlik: 'Temizlik Görevlisi',
  'sağlık': 'Sağlık Ekibi',
  teknisyen: 'Teknisyen',
};

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

function playBeep() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'square';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.1, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.2);
  } catch (err) {
    console.error('Beep çalınamadı:', err);
  }
}

function renderAlarmCard(alert) {
  const id = `alarm-${alert.id || Math.random().toString(36).slice(2)}`;
  const roles = Array.isArray(alert.target_roles) ? alert.target_roles.join(', ') : '-';
  const actions = Array.isArray(alert.actions) && alert.actions.length
    ? alert.actions.map((a) => `<button class="btn btn-secondary">${escapeHtml(a)}</button>`).join('')
    : '<span class="meta">Eylem yok.</span>';

  const card = document.createElement('div');
  card.className = 'alarm-card';
  card.id = id;
  card.innerHTML = `
    <div class="alarm-banner">⚠ ALARM</div>
    <h3>${escapeHtml(alert.headline || 'Bildirim')}</h3>
    <p class="meta">${escapeHtml(alert.summary || '')}</p>
    <video id="video-${id}" controls></video>
    <div class="meta" style="margin-bottom:0.5rem;">
      <strong>Kamera:</strong> ${escapeHtml(alert.camera_id || '-')} |
      <strong>Risk:</strong> ${escapeHtml(alert.risk || '-')} |
      <strong>Hedef Roller:</strong> ${escapeHtml(roles)}
    </div>
    <div class="actions-list">${actions}</div>
  `;

  const video = card.querySelector('video');
  video.src = `/api/v1/pseudolive/videos/${alert.camera_id}`;

  const segment = alert.risk_segment || { start_sec: 0, end_sec: 0 };
  const startSec = Number(segment.start_sec) || 0;
  const endSec = Number(segment.end_sec) || 0;
  const eventType = segment.event_type || alert.current_event || 'genel';

  card.querySelector('h3').textContent = `${escapeHtml(alert.headline || 'Bildirim')} — ${escapeHtml(eventType)}`;

  if (startSec > 0 || endSec > 0) {
    video.addEventListener('loadedmetadata', () => {
      video.currentTime = startSec;
      video.play().catch(() => {});
    }, { once: true });
    video.addEventListener('timeupdate', () => {
      if (endSec > 0 && video.currentTime >= endSec) {
        video.pause();
      }
    });
  } else {
    video.play().catch(() => {});
  }

  const actionButtons = card.querySelectorAll('.actions-list button');
  actionButtons.forEach((btn) => {
    btn.addEventListener('click', () => showToast(`Tetiklendi: ${btn.textContent}`));
  });

  return card;
}

async function loadAlerts(role = '') {
  currentRole = role;
  const url = role ? `/api/v1/field-alerts?role=${encodeURIComponent(role)}` : '/api/v1/field-alerts';
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    containerEl.innerHTML = '';
    const alerts = Array.isArray(data) ? data : [];
    if (!alerts.length) {
      containerEl.innerHTML = '<div class="empty-state">Bu rol için bekleyen alarm yok.</div>';
      return;
    }
    for (const alert of alerts.slice().reverse()) {
      containerEl.appendChild(renderAlarmCard(alert));
    }
  } catch (err) {
    containerEl.innerHTML = `<div class="empty-state">Alarmlar yüklenemedi: ${escapeHtml(err.message)}</div>`;
  }
}

function handleWsMessage(msg) {
  const { stream, data } = msg || {};
  if (stream === 'field.alert' && data) {
    const empty = containerEl.querySelector('.empty-state');
    if (empty) empty.remove();
    containerEl.prepend(renderAlarmCard(data));
    playBeep();
    if (containerEl.children.length > 50) {
      containerEl.lastElementChild.remove();
    }
  }
}

function init() {
  selectorEl.addEventListener('click', (e) => {
    if (e.target.tagName !== 'BUTTON') return;
    for (const btn of selectorEl.querySelectorAll('button')) {
      btn.classList.remove('active');
      btn.classList.add('btn-secondary');
    }
    e.target.classList.add('active');
    e.target.classList.remove('btn-secondary');
    loadAlerts(e.target.dataset.role);
  });

  loadAlerts();

  const ws = new WsClient(`ws://${location.host}/ws`);
  ws.connect();
  ws.onMessage(handleWsMessage);
}

init();
