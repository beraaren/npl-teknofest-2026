import { WsClient } from './ws.js';
import {
  API, api, escapeHtml, mmss, riskClass, showToast, initToast,
  SEVERITY_LABEL, SEVERITY_CLASS, runOnce,
} from './app.js';

/** Modal açıkken ilerleme satırı bu aralıkla tazelenir. */
const PROGRESS_TICK_MS = 300;
/** Kamera duvarı bu aralıkla yoklanır (risk penceresi geçişlerinin akıcı
 * görünmesi için eski 4000ms'lik aralıktan düşürüldü). */
const POLL_MS = 1000;
/** Video konumu sunucudan bu kadar saniye saparsa yeniden hizalanır. */
const DRIFT_TOLERANCE_SEC = 2.5;

const el = {
  grid: document.getElementById('camera-grid'),
  gridEmpty: document.getElementById('grid-empty'),
  libraryStatus: document.getElementById('library-status'),
  notifications: document.getElementById('notifications-list'),
  modal: document.getElementById('modal'),
  modalVideo: document.getElementById('modal-video'),
  modalTitle: document.getElementById('modal-title'),
  modalRisk: document.getElementById('modal-risk'),
  modalConfidence: document.getElementById('modal-confidence'),
  modalProgress: document.getElementById('modal-progress'),
  modalSummary: document.getElementById('modal-summary'),
  modalReasoning: document.getElementById('modal-reasoning'),
  modalEvents: document.getElementById('modal-events'),
  suggestedActions: document.getElementById('suggested-actions'),
  manualTool: document.getElementById('manual-tool'),
  manualNote: document.getElementById('manual-note'),
  manualRunBtn: document.getElementById('manual-run-btn'),
  manualResult: document.getElementById('manual-result'),
  assignRole: document.getElementById('assign-role'),
  assignEvent: document.getElementById('assign-event'),
  assignBtn: document.getElementById('assign-btn'),
  assignExisting: document.getElementById('assign-existing'),
  feedbackBadge: document.getElementById('feedback-badge'),
  fbBtnCorrect: document.getElementById('fb-btn-correct'),
  fbBtnToggleEdit: document.getElementById('fb-btn-toggle-edit'),
  fbEditPanel: document.getElementById('fb-edit-panel'),
  fbType: document.getElementById('fb-type'),
  fbCorrectRisk: document.getElementById('fb-correct-risk'),
  fbCorrectSummary: document.getElementById('fb-correct-summary'),
  fbNotes: document.getElementById('fb-notes'),
  fbBtnSubmitCorrection: document.getElementById('fb-btn-submit-correction'),
  fbBtnCancelEdit: document.getElementById('fb-btn-cancel-edit'),
  toast: document.getElementById('toast'),
};
initToast(el.toast);

/** camera_id -> sunucudan gelen son durum objesi (loadCameras tarafından yazılır). */
const cameraState = new Map();
/** camera_id -> son görülen cycle_index; döngü değişince kilit hafızası sıfırlanır. */
const lastCycle = new Map();
/**
 * camera_id -> Set(tool_name): bu tarayıcı oturumunda o kamerada
 * çalıştırılmış araçlar. Modal, WS mesajı geldiğinde ~500ms'lik debounce ile
 * tazelenir (scheduleModalRefresh); bu tazeleme aksiyon bölümünü sıfırdan
 * çiziyor. Bu Set olmadan, az önce "✓" ile kilitlenmiş bir buton her
 * tazelemede yeniden tıklanabilir görünürdü — backend duplicate kayıt
 * açmasa da operatör aynı butona defalarca basma hissi yaşardı.
 */
const executedTools = new Map();

let toolCatalog = [];
let openCameraId = null;
let currentAnalysis = null;
let progressTimer = null;

// ---------------------------------------------------------------------------
// Kamera duvarı
// ---------------------------------------------------------------------------

function ensureCell(cameraId) {
  let cell = document.getElementById(`cell-${cameraId}`);
  if (cell) return cell;
  cell = document.createElement('div');
  cell.id = `cell-${cameraId}`;
  cell.className = 'camera-cell';
  cell.innerHTML = `
    <video autoplay muted playsinline></video>
    <div class="camera-label">
      <span class="label-text"></span>
      <span class="risk-badge">—</span>
    </div>
    <div class="cell-event"></div>
  `;
  cell.addEventListener('click', () => openModal(cameraId));
  el.grid.appendChild(cell);
  return cell;
}

function renderCell(cameraId, active) {
  const cell = ensureCell(cameraId);
  const video = cell.querySelector('video');
  const labelText = cell.querySelector('.label-text');
  const badge = cell.querySelector('.risk-badge');
  const eventLine = cell.querySelector('.cell-event');

  const prevCycle = lastCycle.get(cameraId);
  const cycleChanged = prevCycle === undefined || prevCycle !== active.cycle_index;
  lastCycle.set(cameraId, active.cycle_index);

  if (cycleChanged) {
    // Yeni video döngüsü başladı: önceki videonun "çalıştırıldı" kilitleri
    // artık anlamsız (farklı analiz, farklı olaylar) — temizle.
    executedTools.delete(cameraId);
    video.src = `${API}/pseudolive/videos/${cameraId}?c=${active.cycle_index}`;
    video.addEventListener('loadedmetadata', () => {
      syncVideoTime(video, active.position_sec);
      video.play().catch(() => { /* otomatik oynatma tarayıcı tarafından engellenebilir */ });
    }, { once: true });
  } else {
    syncVideoTime(video, active.position_sec);
  }

  labelText.textContent = `${active.camera_label} · ${active.video_name || ''}`;
  labelText.title = active.video_name || '';

  // Genel risk rozeti: analizin TAMAMININ verdiği nihai karar. Çerçeveden
  // bağımsız, her zaman görünen bir özet bilgidir ("bu video genel olarak
  // yüksek riskli" anlamına gelir).
  const generalCls = riskClass(active.risk);
  badge.textContent = active.risk || '—';
  badge.className = `risk-badge ${generalCls}`;

  // Kamera çerçevesi: SADECE şu an oynatma konumunda aktif bir risk penceresi
  // varsa (active_window dolu) renklenir — "şu an tam bu saniyede aktif bir
  // tehlike var" anlamına gelir. Pencere yoksa hiçbir .window-* sınıfı
  // eklenmez, çerçeve şeffaftır. Backend bu pencereyi olayın gerçek
  // [timestamp_sec, timestamp_sec+duration] aralığından hesaplar; süre
  // bilinmiyorsa (duration=0) pencere hiç açılmaz.
  cell.classList.remove('window-high', 'window-medium', 'window-low');
  if (active.active_window) {
    cell.classList.add(`window-${riskClass(active.active_window.risk)}`);
  }

  const ev = active.current_event;
  eventLine.textContent = ev
    ? `⚠ ${ev.timestamp} · ${ev.event_type}`
    : `${active.fired_count}/${active.total_events} uyarı · ${mmss(active.position_sec)}/${mmss(active.duration_sec)}`;
  eventLine.className = `cell-event${ev ? ' cell-event-active' : ''}`;
}

function removeCell(cameraId) {
  document.getElementById(`cell-${cameraId}`)?.remove();
  cameraState.delete(cameraId);
  lastCycle.delete(cameraId);
  executedTools.delete(cameraId);
}

/** Video konumunu sunucunun sanal zamanına hizalar (sapma yeterince büyükse). */
function syncVideoTime(video, positionSec) {
  const target = Number(positionSec) || 0;
  if (!Number.isFinite(video.duration) || video.duration <= 0) return;
  if (Math.abs(video.currentTime - target) > DRIFT_TOLERANCE_SEC) {
    try {
      video.currentTime = Math.min(target, Math.max(0, video.duration - 0.2));
    } catch { /* henüz seek edilebilir değil */ }
  }
}

async function loadCameras() {
  try {
    const rows = await api('/pseudolive/cameras');
    const activeRows = rows.filter((r) => r.active);

    el.gridEmpty.hidden = activeRows.length > 0;
    el.libraryStatus.textContent = activeRows.length
      ? `${activeRows.length} kamera yayında`
      : 'Analiz kütüphanesi boş';

    const seen = new Set();
    for (const row of activeRows) {
      seen.add(row.camera_id);
      cameraState.set(row.camera_id, row.active);
      renderCell(row.camera_id, row.active);
    }
    for (const cameraId of [...cameraState.keys()]) {
      if (!seen.has(cameraId)) removeCell(cameraId);
    }
  } catch (err) {
    el.libraryStatus.textContent = `Kameralar yüklenemedi: ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// Bildirim akışı
// ---------------------------------------------------------------------------

function addNotification(stream, data) {
  el.notifications.querySelector('.empty-state')?.remove();

  const severity = data.severity || (data.risk === 'Yüksek' ? 'high' : data.risk === 'Orta' ? 'medium' : 'low');
  const cls = SEVERITY_CLASS[severity] || 'low';

  let title;
  let detail;
  if (stream === 'event.detected') {
    title = `${data.event_type || 'olay'} @ ${data.timestamp || mmss(data.seconds)}`;
    detail = data.description || '';
  } else if (stream === 'notification.push') {
    title = data.headline || 'Riskli durum';
    detail = data.summary || '';
  } else if (stream === 'assignment.created') {
    title = `Görev atandı: ${data.role || ''}`;
    detail = data.headline || data.event_type || '';
  } else if (stream === 'tool.executed') {
    if (data.already_executed) return; // tekrar deneme kartı üretilmez
    title = `Araç çalıştı: ${data.tool_name || ''}`;
    detail = data.mock_result || data.status || '';
  } else {
    // decision.final: döngü başına ayrı bir kart üretilmez, hücre
    // güncellemesi yeterlidir; aksi hâlde her döngüde gereksiz kart birikir.
    return;
  }

  const card = document.createElement('div');
  card.className = `notification-card severity-${cls}`;
  card.innerHTML = `
    <div class="notif-head">
      <span class="notif-severity ${cls}">${SEVERITY_LABEL[severity] || ''}</span>
      <span class="notif-clock">${escapeHtml(new Date().toLocaleTimeString('tr-TR'))}</span>
    </div>
    <div class="notif-title">${escapeHtml(title)}</div>
    ${detail ? `<div class="meta">${escapeHtml(detail)}</div>` : ''}
  `;
  el.notifications.prepend(card);

  const cards = el.notifications.querySelectorAll('.notification-card');
  if (cards.length > 60) cards[cards.length - 1].remove();
}

// ---------------------------------------------------------------------------
// Kamera detay penceresi (modal)
// ---------------------------------------------------------------------------

async function openModal(cameraId) {
  openCameraId = cameraId;
  currentAnalysis = null;
  el.modal.classList.add('open');
  try {
    const [detail, roles] = await Promise.all([
      api(`/pseudolive/cameras/${cameraId}`),
      loadRoleOptions(),
    ]);
    if (!toolCatalog.length) {
      toolCatalog = await api('/tools').catch(() => []);
      fillManualToolSelect();
    }
    renderModal(cameraId, detail.active, detail.analysis);
  } catch (err) {
    showToast(`Kamera açılamadı: ${err.message}`, true);
  }
}

function fillManualToolSelect() {
  el.manualTool.innerHTML = toolCatalog
    .filter((t) => t.enabled)
    .map((t) => `<option value="${escapeHtml(t.tool_name)}">${escapeHtml(t.description || t.tool_name)}</option>`)
    .join('');
}

function toolLabel(toolName) {
  const found = toolCatalog.find((t) => t.tool_name === toolName);
  return found?.description || toolName;
}

function renderModal(cameraId, active, analysis) {
  if (!analysis) return;

  const isNewAnalysis = !currentAnalysis || currentAnalysis.slug !== analysis.slug;
  currentAnalysis = analysis;

  el.modalTitle.textContent = `${active.camera_label} · ${analysis.video_name || ''}`;

  const cls = riskClass(analysis.risk);
  el.modalRisk.textContent = analysis.risk || '—';
  el.modalRisk.className = `risk-badge ${cls}`;
  el.modalConfidence.textContent = `Güven: ${(Number(analysis.confidence) || 0).toFixed(2)}`;
  el.modalProgress.textContent =
    `${mmss(active.position_sec)} / ${mmss(active.duration_sec)} · ${active.fired_count}/${active.total_events} uyarı`;

  el.modalSummary.textContent = analysis.summary || 'Özet üretilmedi.';
  el.modalReasoning.textContent = analysis.reasoning || 'Gerekçe kaydı yok.';

  // Video kaynağı sadece kamera fiilen değiştiğinde yeniden yüklenir; aynı
  // kamera için tekrarlanan renderModal çağrıları (WS tazelemesi) videoyu
  // yeniden başlatmaz.
  if (el.modalVideo.dataset.slug !== analysis.slug) {
    el.modalVideo.dataset.camera = cameraId;
    el.modalVideo.dataset.slug = analysis.slug;
    el.modalVideo.src = `${API}/library/videos/${analysis.slug}`;
    el.modalVideo.addEventListener('loadedmetadata', () => {
      syncVideoTime(el.modalVideo, active.position_sec);
      el.modalVideo.play().catch(() => {});
    }, { once: true });
  }

  renderTimeline(analysis.event_timestamps || []);
  renderSuggestedActions(analysis, cameraId);
  fillAssignEventSelect(analysis.event_timestamps || []);

  el.assignBtn.dataset.slug = analysis.slug || '';
  el.assignBtn.dataset.camera = cameraId;
  el.assignBtn.disabled = !analysis.slug;

  if (isNewAnalysis) {
    delete el.assignBtn.dataset.locked;
    el.assignBtn.classList.remove('btn-done');
    el.assignBtn.textContent = 'Ekibe Ata';
    el.fbCorrectRisk.value = analysis.risk || 'Düşük';
    el.fbCorrectSummary.value = analysis.summary || '';
    el.fbNotes.value = '';
    el.fbEditPanel.hidden = true;
    refreshFeedbackBadge(analysis.slug);
  }
  refreshAssignmentSummary(analysis.slug);

  startProgressWatch();
}

function renderTimeline(stamps) {
  el.modalEvents.innerHTML = stamps.length
    ? stamps.map((s) => `
        <li class="timeline-item severity-${SEVERITY_CLASS[s.severity] || 'low'}" data-seconds="${s.timestamp_sec ?? s.seconds}">
          <span class="timeline-time">${escapeHtml(s.timestamp)}</span>
          <span class="timeline-type">${escapeHtml(s.event_type)}</span>
          <span class="timeline-sev ${SEVERITY_CLASS[s.severity] || 'low'}">${SEVERITY_LABEL[s.severity] || ''}</span>
          <span class="meta">${escapeHtml(s.event || s.vlm_detail || '')}</span>
        </li>`).join('')
    : '<li class="empty-state">Bu videoda uyarı üretilmedi.</li>';

  el.modalEvents.querySelectorAll('.timeline-item').forEach((item) => {
    item.addEventListener('click', () => {
      const secs = Number(item.dataset.seconds) || 0;
      try {
        el.modalVideo.currentTime = secs;
        el.modalVideo.play().catch(() => {});
        showToast(`${mmss(secs)} anına gidildi`);
      } catch { /* seek mümkün değil */ }
    });
  });
}

function fillAssignEventSelect(stamps) {
  el.assignEvent.innerHTML = stamps.length
    ? stamps.map((s, i) =>
        `<option value="${i}">${escapeHtml(s.timestamp)} · ${escapeHtml(s.event_type)} (${SEVERITY_LABEL[s.severity] || ''})</option>`
      ).join('')
    : '<option value="">Olay yok</option>';
}

/**
 * Önerilen Aksiyonlar bölümü: ajanın tetiklediği araçlar
 * (triggered_mock_tools, gerçek buton) + yazdığı metin talimatlar (actions,
 * düz liste). Bu oturumda zaten çalıştırılmış araçlar (executedTools)
 * doğrudan kilitli çizilir; WS tazelemesi bu kilidi silmez.
 */
function renderSuggestedActions(analysis, cameraId) {
  const tools = analysis.triggered_mock_tools || [];
  const locked = executedTools.get(cameraId) || new Set();
  el.suggestedActions.innerHTML = '';

  for (const call of tools) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-secondary';
    btn.title = JSON.stringify(call.params || {});
    if (locked.has(call.tool_name)) {
      btn.disabled = true;
      btn.dataset.locked = '1';
      btn.classList.add('btn-done');
      btn.textContent = `✓ ${toolLabel(call.tool_name)}`;
    } else {
      btn.textContent = toolLabel(call.tool_name);
      btn.addEventListener('click', () => executeSuggestedTool(btn, call.tool_name, call.params || {}, cameraId, analysis.slug));
    }
    el.suggestedActions.appendChild(btn);
  }

  if ((analysis.actions || []).length) {
    const list = document.createElement('ul');
    list.className = 'action-advice';
    list.innerHTML = analysis.actions.map((a) => `<li>${escapeHtml(a)}</li>`).join('');
    el.suggestedActions.appendChild(list);
  }

  if (!tools.length && !(analysis.actions || []).length) {
    el.suggestedActions.innerHTML = '<span class="meta">Aksiyon önerisi üretilmedi.</span>';
  }
}

function startProgressWatch() {
  stopProgressWatch();
  progressTimer = setInterval(() => {
    if (!openCameraId || !el.modal.classList.contains('open') || !currentAnalysis) return;
    
    const dur = el.modalVideo.duration || 0;
    const currentSec = el.modalVideo.currentTime;
    
    const events = currentAnalysis.event_timestamps || [];
    const totalEvents = events.length;
    const firedCount = events.filter(e => {
       const s = Number(e.timestamp_sec ?? e.seconds) || 0;
       return s <= currentSec;
    }).length;

    el.modalProgress.textContent =
      `${mmss(currentSec)} / ${mmss(dur)} · ${firedCount}/${totalEvents} uyarı`;
  }, PROGRESS_TICK_MS);
}

function stopProgressWatch() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

function closeModal() {
  el.modal.classList.remove('open');
  openCameraId = null;
  currentAnalysis = null;
  stopProgressWatch();
  el.modalVideo.pause();
  el.modalVideo.removeAttribute('src');
  el.modalVideo.dataset.camera = '';
  el.modalVideo.dataset.slug = '';
}

// ---------------------------------------------------------------------------
// RLHF / DPO geri bildirim
// ---------------------------------------------------------------------------

function renderFeedbackBadge(feedback) {
  if (!feedback) {
    el.feedbackBadge.className = 'fb-badge pending';
    el.feedbackBadge.textContent = 'Değerlendirilmedi';
    return;
  }
  if (feedback.feedback_type === 'correct') {
    el.feedbackBadge.className = 'fb-badge correct';
    el.feedbackBadge.textContent = '✔ Karar Doğrulandı';
    return;
  }
  const typeMap = {
    false_positive: 'Yanlış Alarm',
    wrong_risk: 'Hatalı Risk',
    wrong_event: 'Hatalı Olay',
    wrong_action: 'Hatalı Aksiyon',
    other: 'Düzeltildi',
  };
  el.feedbackBadge.className = 'fb-badge corrected';
  el.feedbackBadge.textContent = `⚠️ ${typeMap[feedback.feedback_type] || 'Düzeltildi'}`;
}

async function refreshFeedbackBadge(slug) {
  if (!slug) return renderFeedbackBadge(null);
  try {
    const rows = await api(`/feedback?analysis_slug=${encodeURIComponent(slug)}&limit=1`);
    renderFeedbackBadge(rows.length ? rows[0] : null);
  } catch {
    renderFeedbackBadge(null);
  }
}

async function submitFeedback(feedbackType, isCorrection) {
  if (!currentAnalysis || !openCameraId) {
    showToast('Analiz veya kamera seçili değil', true);
    return;
  }
  const payload = {
    analysis_slug: currentAnalysis.slug,
    camera_id: openCameraId,
    feedback_type: feedbackType,
    original_risk: currentAnalysis.risk || '',
    original_summary: currentAnalysis.summary || '',
    original_output: currentAnalysis,
    corrected_risk: isCorrection ? el.fbCorrectRisk.value : (currentAnalysis.risk || ''),
    corrected_summary: isCorrection ? el.fbCorrectSummary.value.trim() : (currentAnalysis.summary || ''),
    supervisor_notes: isCorrection ? el.fbNotes.value.trim() : 'Süpervizör tarafından doğrulandı.',
    prompt_context: {
      camera_label: currentAnalysis.camera_label || openCameraId,
      video_name: currentAnalysis.video_name || '',
    },
  };
  // Backend /feedback analiz başına UPSERT yapar (bkz. store.py); ikinci bir
  // gönderim yeni satır açmaz, mevcut kararı günceller.
  try {
    const res = await api('/feedback', { method: 'POST', body: JSON.stringify(payload) });
    renderFeedbackBadge(res);
    el.fbEditPanel.hidden = true;
    showToast(isCorrection ? '🎯 Düzeltme DPO havuzuna kaydedildi' : '✔ Karar DPO havuzunda doğrulandı');
  } catch (err) {
    showToast(`Geri bildirim kaydedilemedi: ${err.message}`, true);
  }
}

// ---------------------------------------------------------------------------
// Aksiyon çalıştırma
// ---------------------------------------------------------------------------

async function executeSuggestedTool(btn, toolName, params, cameraId, slug) {
  try {
    await runOnce(btn, async () => {
      const res = await callToolExecute(toolName, params, cameraId, slug, 'supervisor');
      markToolExecuted(cameraId, toolName);
      reportToolResult(res, slug);
      return { label: `✓ ${toolLabel(toolName)}` };
    });
  } catch (err) {
    showToast(`Araç çalıştırılamadı: ${err.message}`, true);
  }
}

async function runManualAction() {
  if (!currentAnalysis || !openCameraId) {
    showToast('Analiz veya kamera seçili değil', true);
    return;
  }
  const toolName = el.manualTool.value;
  if (!toolName) return;
  const cameraId = openCameraId;
  const slug = currentAnalysis.slug;
  const note = el.manualNote.value.trim();

  const original = el.manualRunBtn.textContent;
  el.manualRunBtn.disabled = true;
  el.manualRunBtn.textContent = `${original} …`;
  try {
    const params = {
      location: cameraId,
      urgency: currentAnalysis.risk || 'Orta',
      reason: note || (currentAnalysis.headline || ''),
    };
    const res = await callToolExecute(toolName, params, cameraId, slug, 'supervisor');
    el.manualResult.textContent = res.already_executed
      ? `${res.tool_name} zaten çalıştırılmıştı.`
      : `${res.tool_name}: ${res.mock_result || res.status}`;
    if (res.assignment) {
      showToast(`Görev oluşturuldu: ${res.assignment.role}`);
      refreshAssignmentSummary(slug);
    }
    // Manuel panel farklı bir araçla yeniden kullanılabilir olmalı; bu
    // yüzden kalıcı kilitlenmez, yalnızca çalıştırma anında geçici disable.
    el.manualRunBtn.textContent = 'Çalıştırıldı ✓';
    setTimeout(() => { el.manualRunBtn.textContent = original; }, 1200);
  } catch (err) {
    showToast(`Araç çalıştırılamadı: ${err.message}`, true);
    el.manualRunBtn.textContent = original;
  } finally {
    el.manualRunBtn.disabled = false;
  }
}

function callToolExecute(toolName, params, cameraId, slug, triggeredBy) {
  return api('/tools/execute', {
    method: 'POST',
    body: JSON.stringify({
      tool_name: toolName,
      params,
      camera_id: cameraId,
      analysis_slug: slug,
      triggered_by: triggeredBy,
    }),
  });
}

function markToolExecuted(cameraId, toolName) {
  if (!executedTools.has(cameraId)) executedTools.set(cameraId, new Set());
  executedTools.get(cameraId).add(toolName);
}

function reportToolResult(res, slug) {
  if (res.already_executed) {
    showToast(`${res.tool_name} zaten çalıştırılmıştı.`);
    return;
  }
  showToast(`${res.tool_name}: ${res.mock_result || res.status}`);
  if (res.assignment) {
    showToast(`Görev oluşturuldu: ${res.assignment.role}`);
    refreshAssignmentSummary(slug);
  }
}

async function refreshAssignmentSummary(slug) {
  if (!slug) {
    el.assignExisting.textContent = '';
    return;
  }
  try {
    const rows = await api(`/assignments?analysis_slug=${encodeURIComponent(slug)}`);
    el.assignExisting.innerHTML = rows.length
      ? `Bu olay için ${rows.length} atama: ` + rows
          .map((r) => `<span class="status-badge status-${escapeHtml(r.status)}">${escapeHtml(r.role)} · ${escapeHtml(r.status)}</span>`)
          .join(' ')
      : '<span class="meta">Bu olay için henüz atama yok.</span>';
  } catch {
    el.assignExisting.textContent = '';
  }
}

async function submitAssignment() {
  const slug = el.assignBtn.dataset.slug;
  const cameraId = el.assignBtn.dataset.camera;
  const role = el.assignRole.value;
  if (!slug || !role) {
    showToast('Analiz veya ekip seçilmedi', true);
    return;
  }
  try {
    await runOnce(el.assignBtn, async () => {
      const eventIndex = el.assignEvent.value === '' ? null : Number(el.assignEvent.value);
      const row = await api('/assignments', {
        method: 'POST',
        body: JSON.stringify({ analysis_slug: slug, role, camera_id: cameraId, event_index: eventIndex }),
      });
      showToast(row.duplicate
        ? `Bu görev zaten açık: ${row.role} · ${row.event_type}`
        : `Görev atandı: ${row.role} · ${row.event_type} @ ${row.event_timestamp}`);
      await refreshAssignmentSummary(slug);
      return { label: '✓ Atandı' };
    });
  } catch (err) {
    showToast(`Atama başarısız: ${err.message}`, true);
  }
}

async function loadRoleOptions() {
  try {
    const rolesList = await api('/roles');
    el.assignRole.innerHTML = rolesList
      .map((r) => `<option value="${escapeHtml(r.role)}">${escapeHtml(r.label)}</option>`)
      .join('');
    return rolesList;
  } catch {
    el.assignRole.innerHTML = '<option value="">Roller yüklenemedi</option>';
    return [];
  }
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

/**
 * Modal açıkken her WS mesajında modalin tamamını yeniden çekmek ağır ve
 * gereksizdir; bunun yerine ~500ms'lik debounce ile tek bir tazeleme
 * planlanır. Aksiyon butonlarının kilit durumu executedTools Set'inde
 * ayrıca tutulduğu için bu tazeleme onları silmez (renderSuggestedActions
 * kilidi Set'ten okuyup yeniden uygular).
 */
let modalRefreshTimer = null;
function scheduleModalRefresh() {
  if (modalRefreshTimer) return;
  modalRefreshTimer = setTimeout(async () => {
    modalRefreshTimer = null;
    if (!openCameraId || !el.modal.classList.contains('open')) return;
    try {
      const detail = await api(`/pseudolive/cameras/${openCameraId}`);
      if (currentAnalysis && detail.analysis && currentAnalysis.slug !== detail.analysis.slug) {
        return; // Kamera bir sonraki videoya geçti, modaldaki şu anki analiz görünümünü bozma
      }
      renderModal(openCameraId, detail.active, detail.analysis);
    } catch { /* modal kapanmış olabilir */ }
  }, 500);
}

function handleWsMessage(msg) {
  const { stream, data } = msg || {};
  if (!data) return;

  addNotification(stream, data);

  if (stream === 'event.detected' && data.camera_id) {
    const cam = cameraState.get(data.camera_id);
    if (cam) {
      cam.current_event = data;
      cam.fired_count = (cam.fired_count || 0) + 1;
      renderCell(data.camera_id, cam);
    }
  } else if (stream === 'decision.final' && data.camera_id) {
    const cam = cameraState.get(data.camera_id);
    if (cam) {
      cam.risk = data.risk || cam.risk;
      cam.summary = data.summary || cam.summary;
      cam.current_event = null;
      cam.fired_count = 0;
      renderCell(data.camera_id, cam);
    }
  } else if (stream === 'tool.executed' && data.camera_id && data.tool_name && !data.already_executed) {
    // Başka bir ekrandan (örn. saha) tetiklenen araç da bu kameranın kilit
    // kümesine işlenir; modal daha sonra açıldığında araç yeniden
    // tıklanabilir görünmez.
    markToolExecuted(data.camera_id, data.tool_name);
  }

  if (openCameraId && data.camera_id === openCameraId && el.modal.classList.contains('open')) {
    scheduleModalRefresh();
  }
}

// ---------------------------------------------------------------------------
// Başlatma
// ---------------------------------------------------------------------------

async function init() {
  await loadCameras();
  setInterval(loadCameras, POLL_MS);

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WsClient(`${proto}://${location.host}/ws`);
  ws.connect();
  ws.onMessage(handleWsMessage);

  el.assignBtn.addEventListener('click', submitAssignment);
  el.manualRunBtn.addEventListener('click', runManualAction);

  el.fbBtnCorrect.addEventListener('click', () => submitFeedback('correct', false));
  el.fbBtnToggleEdit.addEventListener('click', () => {
    el.fbEditPanel.hidden = !el.fbEditPanel.hidden;
  });
  el.fbBtnCancelEdit.addEventListener('click', () => {
    el.fbEditPanel.hidden = true;
  });
  el.fbBtnSubmitCorrection.addEventListener('click', () => {
    submitFeedback(el.fbType.value || 'other', true);
  });

  document.getElementById('modal-close').addEventListener('click', closeModal);
  el.modal.addEventListener('click', (e) => {
    if (e.target === el.modal) closeModal();
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });
}

init();
