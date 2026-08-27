import { WsClient } from './ws.js';
import { api, escapeHtml, mmss, showToast, initToast, runOnce } from './app.js';

const RISK_CLASS = { 'Yüksek': 'high', 'Orta': 'medium', 'Düşük': 'low' };
const STATUS_LABEL = { atandi: 'Bekliyor', goruldu: 'Görüldü', devam_ediyor: 'İntikal Edildi', tamamlandi: 'Bitti' };
/** Olayın kaç saniye öncesinden oynatmaya başlanacağı (bağlam için). */
const PRE_ROLL_SEC = 3;

const el = {
  selector: document.getElementById('role-selector'),
  statusFilter: document.getElementById('status-filter'),
  refreshBtn: document.getElementById('refresh-btn'),
  container: document.getElementById('assignments-container'),
  roleStatus: document.getElementById('role-status'),
  toast: document.getElementById('toast'),
};
initToast(el.toast);

let currentRole = '';
let roleLabels = new Map();
/** Araç katalogu; kart başına Önerilen/Manuel aksiyon listelerini kurar. */
let toolCatalog = [];

function toolLabel(toolName) {
  const tool = toolCatalog.find((t) => t.tool_name === toolName);
  return tool?.description || toolName;
}

function playAlertSound() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 880;
    gain.gain.value = 0.15;
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    setTimeout(() => { osc.stop(); ctx.close(); }, 220);
  } catch { /* ses API'si yoksa sessizce geç */ }
}

/**
 * Bir atamayı kart olarak oluşturur.
 *
 * Video, atamadaki `event_seconds` değerinin biraz öncesinden başlar ve
 * `event_end_seconds` biliniyorsa (backend'in ürettiği duration > 0 olduğu
 * durumlarda) tam o anda durur; süre bilinmiyorsa (event_end_seconds === 0)
 * klip sonuna kadar oynar — süre uydurulmaz.
 */
function renderAssignmentCard(row) {
  const riskCls = RISK_CLASS[row.risk] || 'low';
  const card = document.createElement('div');
  card.className = `alarm-card status-${row.status}`;
  card.id = `assignment-${row.id}`;

  const actions = (row.actions || []).length
    ? row.actions.map((a) => `<li>${escapeHtml(a)}</li>`).join('')
    : '<li class="meta">Aksiyon önerisi yok.</li>';

  card.innerHTML = `
    <div class="alarm-banner">⚠ GÖREV #${row.id} · ${escapeHtml(STATUS_LABEL[row.status] || row.status)}</div>
    <div class="card-head">
      <h3>${escapeHtml(row.headline || 'Görev')}</h3>
      <span class="risk-badge ${riskCls}">${escapeHtml(row.risk || '—')}</span>
    </div>

    <div class="meta card-facts">
      <span><strong>Kamera:</strong> ${escapeHtml(row.camera_id || '-')}</span>
      <span><strong>Olay:</strong> ${escapeHtml(row.event_type || '-')}</span>
      <span><strong>Olay anı:</strong> ${escapeHtml(row.event_timestamp || mmss(row.event_seconds))}</span>
      <span><strong>Atandı:</strong> ${escapeHtml(row.created_at || '')}</span>
    </div>

    ${row.note ? `<p class="note-box">📌 Not: ${escapeHtml(row.note)}</p>` : ''}

    <h4>🧠 Ajan Olay Özeti</h4>
    <p class="summary-box">${escapeHtml(row.summary || 'Özet yok.')}</p>

    <video controls playsinline preload="metadata"></video>
    <div class="meta clip-hint">
      ${row.event_end_seconds > row.event_seconds
        ? `Klip olayın süresine (${mmss(row.event_seconds)}–${mmss(row.event_end_seconds)}) konumlandı ve sonunda durur.`
        : `Klip olayın anına (${escapeHtml(row.event_timestamp || mmss(row.event_seconds))}) konumlandı.`}
    </div>

    <h4>✅ Yapılacaklar</h4>
    <ul class="action-advice">${actions}</ul>

    <h4>⚡ Önerilen Aksiyonlar</h4>
    <div class="actions-list card-suggested-actions"></div>

    <h4>🛠 Manuel Aksiyon</h4>
    <div class="manual-action-panel card-manual-panel">
      <select class="manual-tool-select"></select>
      <button class="btn btn-secondary manual-run-btn">Çalıştır</button>
    </div>

    <div class="actions-list card-buttons">
      <button class="btn btn-secondary" data-status="goruldu">Gördüm</button>
      <button class="btn btn-secondary" data-status="devam_ediyor">İntikal Edildi</button>
      <button class="btn btn-danger" data-status="tamamlandi">Bitti</button>
    </div>

    <details>
      <summary class="meta">Ajanın gerekçesi</summary>
      <p class="meta">${escapeHtml(row.reasoning || 'Gerekçe kaydı yok.')}</p>
    </details>
  `;

  // Video: atanan olayın anına konumlanır, biliniyorsa bitişte durur.
  const video = card.querySelector('video');
  if (row.analysis_slug) {
    video.src = `${API()}/library/videos/${encodeURIComponent(row.analysis_slug)}`;
    const start = Math.max(0, (Number(row.event_seconds) || 0) - PRE_ROLL_SEC);
    const end = Number(row.event_end_seconds) || 0;
    video.addEventListener('loadedmetadata', () => {
      try {
        video.currentTime = Math.min(start, Math.max(0, video.duration - 0.2));
      } catch { /* seek mümkün değil */ }
    }, { once: true });
    if (end > (Number(row.event_seconds) || 0)) {
      video.addEventListener('timeupdate', () => {
        if (video.currentTime >= end) video.pause();
      });
    }
  } else {
    video.replaceWith(Object.assign(document.createElement('div'), {
      className: 'empty-state',
      textContent: 'Bu görev için video bulunamadı.',
    }));
  }

  // Önerilen aksiyonlar: assignments tablosu hangi aracın tetiklediğini
  // taşımaz (bkz. store.py); bu ekibe atama yapan aracı katalogdan
  // (assigns_role alanı) geriye eşleyerek birincil öneriyi kuruyoruz. Bu araç
  // zaten tetiklenmiş olabilir (idempotency backend'de zaten var), tekrar
  // tıklamak zararsızdır.
  const suggestedBox = card.querySelector('.card-suggested-actions');
  suggestedBox.innerHTML = '';
  const suggestedTool = toolCatalog.find((t) => t.enabled && t.assigns_role === row.role);
  if (suggestedTool) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-secondary';
    btn.textContent = suggestedTool.description || suggestedTool.tool_name;
    btn.addEventListener('click', () => runTool(btn, row, suggestedTool.tool_name));
    suggestedBox.appendChild(btn);
  } else {
    suggestedBox.innerHTML = '<span class="meta">Bu görev için önerilen araç yok.</span>';
  }

  const manualSelect = card.querySelector('.manual-tool-select');
  manualSelect.innerHTML = toolCatalog
    .filter((t) => t.enabled)
    .map((t) => `<option value="${escapeHtml(t.tool_name)}">${escapeHtml(t.description || t.tool_name)}</option>`)
    .join('');
  card.querySelector('.manual-run-btn').addEventListener('click', (e) => {
    runTool(e.target, row, manualSelect.value, true);
  });

  // Durum butonları
  card.querySelectorAll('button[data-status]').forEach((btn) => {
    btn.addEventListener('click', () => updateStatus(row, btn.dataset.status, btn));
  });

  if (row.status === 'tamamlandi') {
    card.querySelectorAll('button[data-status]').forEach((b) => { b.disabled = true; });
  }

  return card;
}

function API() { return '/api/v1'; }

async function updateStatus(row, status, btn) {
  try {
    await runOnce(btn, async () => {
      const updated = await api(`/assignments/${row.id}`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      });
      showToast(`Görev #${updated.id}: ${STATUS_LABEL[updated.status] || updated.status}`);
      await loadAssignments();
      return { label: STATUS_LABEL[status] || status };
    });
  } catch (err) {
    showToast(`Güncellenemedi: ${err.message}`, true);
  }
}

async function runTool(btn, row, toolName, isManual = false) {
  if (!toolName) return;
  try {
    await runOnce(btn, async () => {
      const res = await api('/tools/execute', {
        method: 'POST',
        body: JSON.stringify({
          tool_name: toolName,
          params: {
            location: row.camera_id || 'saha',
            urgency: row.risk || 'Orta',
            reason: (row.headline || '').slice(0, 100),
          },
          camera_id: row.camera_id,
          analysis_slug: row.analysis_slug,
          triggered_by: 'field',
        }),
      });
      showToast(res.already_executed
        ? `${res.tool_name} zaten çalıştırılmıştı.`
        : `${res.tool_name}: ${res.mock_result || res.status}`);
      return { label: isManual ? 'Çalıştırıldı ✓' : `✓ ${toolLabel(toolName)}` };
    });
  } catch (err) {
    showToast(`Araç çalıştırılamadı: ${err.message}`, true);
  }
}

// ---------------------------------------------------------------------------
// Yükleme
// ---------------------------------------------------------------------------

async function loadAssignments() {
  if (!currentRole) {
    el.container.innerHTML = '<div class="empty-state">Görevlerinizi görmek için yukarıdan ekibinizi seçin.</div>';
    el.roleStatus.textContent = 'Ekip seçin';
    return;
  }

  const params = new URLSearchParams({ role: currentRole });
  if (el.statusFilter.value) params.set('status', el.statusFilter.value);

  try {
    const rows = await api(`/assignments?${params.toString()}`);
    el.container.innerHTML = '';
    el.roleStatus.textContent = `${roleLabels.get(currentRole) || currentRole} · ${rows.length} görev`;

    if (!rows.length) {
      el.container.innerHTML =
        '<div class="empty-state">Ekibinize atanmış bekleyen görev yok. '
        + 'Süpervizör bir olayı atadığında burada anında görünür.</div>';
      return;
    }
    for (const row of rows) {
      el.container.appendChild(renderAssignmentCard(row));
    }
  } catch (err) {
    el.container.innerHTML = `<div class="empty-state">Görevler yüklenemedi: ${escapeHtml(err.message)}</div>`;
  }
}

async function loadRoles() {
  try {
    const rolesList = await api('/roles');
    roleLabels = new Map(rolesList.map((r) => [r.role, r.label]));
    el.selector.innerHTML = rolesList
      .map((r, i) =>
        `<button class="btn ${i === 0 ? 'active' : 'btn-secondary'}" data-role="${escapeHtml(r.role)}">${escapeHtml(r.label)}</button>`)
      .join('');
    currentRole = rolesList.length ? rolesList[0].role : '';
  } catch (err) {
    el.selector.innerHTML = `<span class="empty-state">Roller yüklenemedi: ${escapeHtml(err.message)}</span>`;
  }
}

async function loadToolCatalog() {
  try {
    toolCatalog = await api('/tools');
  } catch {
    toolCatalog = [];
  }
}

function handleWsMessage(msg) {
  const { stream, data } = msg || {};
  if (!data) return;

  if (stream === 'assignment.created' && data.role === currentRole) {
    playAlertSound();
    showToast(`Yeni görev: ${data.headline || data.event_type}`);
    loadAssignments();
  } else if (stream === 'assignment.updated' && data.role === currentRole) {
    loadAssignments();
  }
}

function init() {
  el.selector.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-role]');
    if (!btn) return;
    for (const b of el.selector.querySelectorAll('button')) {
      b.classList.remove('active');
      b.classList.add('btn-secondary');
    }
    btn.classList.add('active');
    btn.classList.remove('btn-secondary');
    currentRole = btn.dataset.role;
    loadAssignments();
  });

  el.statusFilter.addEventListener('change', loadAssignments);
  el.refreshBtn.addEventListener('click', loadAssignments);

  Promise.all([loadRoles(), loadToolCatalog()]).then(loadAssignments);

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WsClient(`${proto}://${location.host}/ws`);
  ws.connect();
  ws.onMessage(handleWsMessage);
}

init();
