import { WsClient } from './ws.js';
import { API, api, escapeHtml, mmss, riskClass, showToast, initToast, runOnce } from './app.js';

const STATUS_LABEL = { atandi: 'Bekliyor', goruldu: 'Görüldü', devam_ediyor: 'İntikal Edildi', tamamlandi: 'Bitti' };
/** Durum ilerleme sırası: bir görev geri gidemez, bu yüzden mevcut durumun
 * rankından düşük/eşit statü butonları kilitlenir. Eskiden yalnızca
 * "tamamlandi" durumunda tüm butonlar kilitleniyordu; "Gördüm"e basılıp
 * durum "goruldu" olduktan sonra "Gördüm" butonu hâlâ tıklanabilir kalıyordu. */
const STATUS_RANK = { atandi: 0, goruldu: 1, devam_ediyor: 2, tamamlandi: 3 };
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
let toolCatalog = [];
/**
 * assignment_id -> Set(tool_name): bu oturumda bu görev için çalıştırılmış
 * araçlar. loadAssignments() periyodik olarak (WS mesajı, filtre değişimi,
 * "Yenile") TÜM kartları sıfırdan yeniden çiziyor; bu Set olmadan her
 * yeniden çizimde "✓ Çalıştırıldı" kilidi kaybolup buton yeniden tıklanabilir
 * görünürdü.
 */
const executedTools = new Map();

function toolLabel(toolName) {
  const found = toolCatalog.find((t) => t.tool_name === toolName);
  return found?.description || toolName;
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
 * durumlarda) tam o anda durur; süre bilinmiyorsa (event_end_seconds === 0
 * veya event_seconds'a eşit) klip sonuna kadar oynar — süre uydurulmaz.
 */
function renderAssignmentCard(row) {
  const card = document.createElement('div');
  card.className = `alarm-card status-${row.status}`;
  card.id = `assignment-${row.id}`;

  const actionsHtml = (row.actions || []).length
    ? row.actions.map((a) => `<li>${escapeHtml(a)}</li>`).join('')
    : '<li class="meta">Aksiyon önerisi yok.</li>';

  const hasKnownEnd = Number(row.event_end_seconds) > Number(row.event_seconds);
  const clipHint = hasKnownEnd
    ? `Klip olayın süresine (${mmss(row.event_seconds)}–${mmss(row.event_end_seconds)}) konumlandı ve sonunda durur.`
    : `Klip olayın anına (${escapeHtml(row.event_timestamp || mmss(row.event_seconds))}) konumlandı.`;

  card.innerHTML = `
    <div class="alarm-banner">⚠ GÖREV #${row.id} · ${escapeHtml(STATUS_LABEL[row.status] || row.status)}</div>
    <div class="card-head">
      <h3>${escapeHtml(row.headline || 'Görev')}</h3>
      <span class="risk-badge ${riskClass(row.risk)}">${escapeHtml(row.risk || '—')}</span>
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
    <div class="meta clip-hint">${clipHint}</div>

    <h4>✅ Yapılacaklar</h4>
    <ul class="action-advice">${actionsHtml}</ul>

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

  setupCardVideo(card, row);
  setupSuggestedAction(card, row);
  setupManualAction(card, row);
  setupStatusButtons(card, row);

  return card;
}

function setupCardVideo(card, row) {
  const video = card.querySelector('video');
  if (!row.analysis_slug) {
    video.replaceWith(Object.assign(document.createElement('div'), {
      className: 'empty-state',
      textContent: 'Bu görev için video bulunamadı.',
    }));
    return;
  }
  video.src = `${API}/library/videos/${encodeURIComponent(row.analysis_slug)}`;
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
}

function setupSuggestedAction(card, row) {
  const box = card.querySelector('.card-suggested-actions');
  const locked = executedTools.get(row.id) || new Set();
  // assignments tablosu hangi aracın bu görevi tetikledigini taşımaz (bkz.
  // store.py); bu ekibe atama yapan aracı katalogdan (assigns_role alanı)
  // geriye eşleyerek birincil öneriyi kuruyoruz.
  const suggestedTool = toolCatalog.find((t) => t.enabled && t.assigns_role === row.role);
  if (!suggestedTool) {
    box.innerHTML = '<span class="meta">Bu görev için önerilen araç yok.</span>';
    return;
  }
  const btn = document.createElement('button');
  btn.className = 'btn btn-secondary';
  if (locked.has(suggestedTool.tool_name)) {
    btn.disabled = true;
    btn.dataset.locked = '1';
    btn.classList.add('btn-done');
    btn.textContent = `✓ ${suggestedTool.description || suggestedTool.tool_name}`;
  } else {
    btn.textContent = suggestedTool.description || suggestedTool.tool_name;
    btn.addEventListener('click', () => executeSuggestedTool(btn, row, suggestedTool.tool_name));
  }
  box.appendChild(btn);
}

function setupManualAction(card, row) {
  const select = card.querySelector('.manual-tool-select');
  select.innerHTML = toolCatalog
    .filter((t) => t.enabled)
    .map((t) => `<option value="${escapeHtml(t.tool_name)}">${escapeHtml(t.description || t.tool_name)}</option>`)
    .join('');
  card.querySelector('.manual-run-btn').addEventListener('click', (e) => {
    runManualTool(e.currentTarget, row, select.value);
  });
}

function setupStatusButtons(card, row) {
  const currentRank = STATUS_RANK[row.status] ?? 0;
  card.querySelectorAll('button[data-status]').forEach((btn) => {
    const target = btn.dataset.status;
    const targetRank = STATUS_RANK[target] ?? 0;
    if (targetRank <= currentRank) {
      btn.disabled = true;
      if (targetRank === currentRank) {
        btn.classList.add('btn-done');
        btn.textContent = `✓ ${STATUS_LABEL[target] || target}`;
      }
    } else {
      btn.addEventListener('click', () => updateStatus(row, target, btn));
    }
  });
}

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

async function executeSuggestedTool(btn, row, toolName) {
  try {
    await runOnce(btn, async () => {
      const res = await execTool(row, toolName);
      markToolExecuted(row.id, toolName);
      showToast(res.already_executed
        ? `${res.tool_name} zaten çalıştırılmıştı.`
        : `${res.tool_name}: ${res.mock_result || res.status}`);
      return { label: `✓ ${toolLabel(toolName)}` };
    });
  } catch (err) {
    showToast(`Araç çalıştırılamadı: ${err.message}`, true);
  }
}

async function runManualTool(btn, row, toolName) {
  if (!toolName) return;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = `${original} …`;
  try {
    const res = await execTool(row, toolName);
    markToolExecuted(row.id, toolName);
    showToast(res.already_executed
      ? `${res.tool_name} zaten çalıştırılmıştı.`
      : `${res.tool_name}: ${res.mock_result || res.status}`);
    btn.textContent = 'Çalıştırıldı ✓';
    setTimeout(() => { btn.textContent = original; }, 1200);
  } catch (err) {
    showToast(`Araç çalıştırılamadı: ${err.message}`, true);
    btn.textContent = original;
  } finally {
    btn.disabled = false;
  }
}

function markToolExecuted(assignmentId, toolName) {
  if (!executedTools.has(assignmentId)) executedTools.set(assignmentId, new Set());
  executedTools.get(assignmentId).add(toolName);
}

function execTool(row, toolName) {
  return api('/tools/execute', {
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
