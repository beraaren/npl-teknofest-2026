/**
 * Saha ekibi ekranı — yalnızca kendi ekibine atanan görevleri gösterir.
 *
 * MOCK KALDIRILDI
 * Eskiden bu ekran, atama olmadan da tüm uyarıları listeliyor ve canlı kamera
 * akışını gösteriyordu; ajanın olay özeti hiç görünmüyordu, aksiyon butonları
 * ise yalnızca ekranda bir bildirim gösterip kayboluyordu.
 *
 * ŞİMDİ
 * - Görevler süpervizörün yaptığı atamadan gelir (`/assignments?role=`), yani
 *   ekip yalnızca kendisine ait olanı görür.
 * - Her görev, karar ajanının yazdığı olay özetini taşır; süpervizörün gördüğü
 *   metnin aynısıdır (sunucu tarafında tek kaynaktan okunur).
 * - Video, olayın geçtiği saniyeye konumlanır; ekip doğrudan kritik anı görür.
 * - "Gördüm" / "Tamamlandı" işaretleri ve araç çalıştırmaları sunucuya yazılır,
 *   yani kalıcıdır ve süpervizör ekranı da bunu görür.
 */
import { WsClient } from './ws.js';

const API = '/api/v1';

const RISK_CLASS = { 'Yüksek': 'high', 'Orta': 'medium', 'Düşük': 'low' };
const STATUS_LABEL = { atandi: 'Bekliyor', goruldu: 'Görüldü', tamamlandi: 'Tamamlandı' };
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

let currentRole = '';
let roleLabels = new Map();
let toastTimer = null;

// ---------------------------------------------------------------------------
// Yardımcılar
// ---------------------------------------------------------------------------

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

function mmss(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

function showToast(message, isError = false) {
  el.toast.textContent = message;
  el.toast.classList.toggle('toast-error', isError);
  el.toast.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.remove('show'), 3200);
}

async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch { /* gövde JSON değil */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

function playAlertSound() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'square';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.08, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.25);
  } catch { /* ses engellenebilir */ }
}

// ---------------------------------------------------------------------------
// Görev kartı
// ---------------------------------------------------------------------------

/**
 * Bir atamayı kart olarak oluşturur.
 *
 * Video, atamadaki `event_seconds` değerinin biraz öncesinden başlar; ekip
 * olayı bağlamıyla görür. Klip, olaydan birkaç saniye sonra durur ki dikkat
 * kritik anda kalsın.
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

    ${row.note ? `<p class="note-box">📌 Süpervizör notu: ${escapeHtml(row.note)}</p>` : ''}

    <h4>🧠 Ajan Olay Özeti</h4>
    <p class="summary-box">${escapeHtml(row.summary || 'Özet yok.')}</p>

    <video controls playsinline preload="metadata"></video>
    <div class="meta clip-hint">Klip olayın geçtiği ana (${escapeHtml(row.event_timestamp || mmss(row.event_seconds))}) konumlandı.</div>

    <h4>✅ Yapılacaklar</h4>
    <ul class="action-advice">${actions}</ul>

    <div class="actions-list card-buttons">
      <button class="btn btn-secondary" data-status="goruldu">Gördüm</button>
      <button class="btn btn-danger" data-status="tamamlandi">Tamamlandı</button>
      <button class="btn btn-secondary" data-tool="call_health_team">Sağlık Ekibi Çağır</button>
      <button class="btn btn-secondary" data-tool="secure_area">Alanı Güvene Al</button>
      <button class="btn btn-secondary" data-tool="record_incident">Olayı Kaydet</button>
    </div>

    <details>
      <summary class="meta">Ajanın gerekçesi</summary>
      <p class="meta">${escapeHtml(row.reasoning || 'Gerekçe kaydı yok.')}</p>
    </details>
  `;

  // Video: atanan olayın anına konumlanır.
  const video = card.querySelector('video');
  if (row.analysis_slug) {
    video.src = `${API}/library/videos/${encodeURIComponent(row.analysis_slug)}`;
    const start = Math.max(0, (Number(row.event_seconds) || 0) - PRE_ROLL_SEC);
    video.addEventListener('loadedmetadata', () => {
      try {
        video.currentTime = Math.min(start, Math.max(0, video.duration - 0.2));
      } catch { /* seek mümkün değil */ }
    }, { once: true });
  } else {
    video.replaceWith(Object.assign(document.createElement('div'), {
      className: 'empty-state',
      textContent: 'Bu görev için video bulunamadı.',
    }));
  }

  // Durum butonları
  card.querySelectorAll('button[data-status]').forEach((btn) => {
    btn.addEventListener('click', () => updateStatus(row, btn.dataset.status, btn));
  });
  // Araç butonları — sunucuda gerçekten çalışır
  card.querySelectorAll('button[data-tool]').forEach((btn) => {
    btn.addEventListener('click', () => runTool(row, btn.dataset.tool, btn));
  });

  if (row.status === 'tamamlandi') {
    card.querySelectorAll('button[data-status]').forEach((b) => { b.disabled = true; });
  }

  return card;
}

async function updateStatus(row, status, btn) {
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = `${original} …`;
  try {
    const updated = await api(`/assignments/${row.id}`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
    });
    showToast(`Görev #${updated.id}: ${STATUS_LABEL[updated.status] || updated.status}`);
    await loadAssignments();
  } catch (err) {
    showToast(`Güncellenemedi: ${err.message}`, true);
    btn.textContent = original;
    btn.disabled = false;
  }
}

async function runTool(row, toolName, btn) {
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = `${original} …`;
  try {
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
    showToast(`${res.tool_name}: ${res.mock_result || res.status}`);
    btn.classList.add('btn-done');
    btn.textContent = `✓ ${original}`;
  } catch (err) {
    showToast(`Araç çalıştırılamadı: ${err.message}`, true);
    btn.textContent = original;
  } finally {
    btn.disabled = false;
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
    const roles = await api('/roles');
    roleLabels = new Map(roles.map((r) => [r.role, r.label]));
    el.selector.innerHTML = roles
      .map((r, i) =>
        `<button class="btn ${i === 0 ? 'active' : 'btn-secondary'}" data-role="${escapeHtml(r.role)}">${escapeHtml(r.label)}</button>`)
      .join('');
    currentRole = roles.length ? roles[0].role : '';
  } catch (err) {
    el.selector.innerHTML = `<span class="empty-state">Roller yüklenemedi: ${escapeHtml(err.message)}</span>`;
  }
}

function handleWsMessage(msg) {
  const { stream, data } = msg || {};
  if (!data) return;

  // Yalnızca bu ekibi ilgilendiren atamalar dikkate alınır.
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

  loadRoles().then(loadAssignments);

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WsClient(`${proto}://${location.host}/ws`);
  ws.connect();
  ws.onMessage(handleWsMessage);
}

init();
