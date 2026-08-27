import { WsClient } from './ws.js';
import {
  api, escapeHtml, mmss, riskClass, showToast, initToast,
  SEVERITY_LABEL, SEVERITY_CLASS, runOnce,
} from './app.js';

const API = '/api/v1';

/** Video konumu sunucudan bu kadar saniye saparsa yeniden hizalanır. */
const DRIFT_TOLERANCE_SEC = 2.5;
/** Modal içindeki video zamanı bu aralıkla kontrol edilir (risk penceresi). */
const WINDOW_CHECK_MS = 300;

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

/** camera_id -> sunucudan gelen son durum. */
const cameras = new Map();
/** Araç katalogu (assigns_role dahil), modal açılışında ve manuel panelde kullanılır. */
let toolCatalog = [];
let openCameraId = null;
let currentAnalysis = null;
/** Modal açıkken video zamanını izleyip risk çerçevesini güncelleyen interval. */
let modalWindowTimer = null;

// ---------------------------------------------------------------------------
// Kamera duvarı
// ---------------------------------------------------------------------------

function renderCell(cameraId, active) {
  let cell = document.getElementById(`cell-${cameraId}`);
  if (!cell) {
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
  }

  const video = cell.querySelector('video');
  const labelText = cell.querySelector('.label-text');
  const badge = cell.querySelector('.risk-badge');
  const eventLine = cell.querySelector('.cell-event');

  const prev = cameras.get(cameraId);
  const cycleChanged = !prev || prev.cycle_index !== active.cycle_index;

  if (cycleChanged) {
    // Yeni video döngüsü: kaynağı yenile ve sunucunun konumuna hizala.
    video.src = `${API}/pseudolive/videos/${cameraId}?c=${active.cycle_index}`;
    video.addEventListener('loadedmetadata', () => {
      syncVideo(video, active.position_sec);
      video.play().catch(() => { /* otomatik oynatma engellenebilir */ });
    }, { once: true });
  } else {
    syncVideo(video, active.position_sec);
  }

  labelText.textContent = `${active.camera_label} · ${active.video_name || ''}`;
  labelText.title = active.video_name || '';

  // Genel risk rozeti: analizin tamamının verdiği karar (çerçeveden bağımsız,
  // her zaman görünür bir özet bilgi).
  const cls = riskClass(active.risk);
  badge.textContent = active.risk || '—';
  badge.className = `risk-badge ${cls}`;

  // Kamera çerçevesi: SADECE şu an oynatma konumunda aktif bir risk penceresi
  // varsa (active_window dolu) yanar, o pencerenin riskine göre renklenir.
  // Eskiden bu çerçeve analizin GENEL riskine bağlıydı ve video boyunca
  // kesintisiz yanıyordu; artık backend'in ürettiği [start_sec, end_sec]
  // aralığı dışında çerçeve nötrdür (bkz. replay.py: active_risk_window).
  cell.classList.remove('window-high', 'window-medium', 'window-low');
  const win = active.active_window;
  if (win) {
    cell.classList.add(`window-${riskClass(win.risk)}`);
  }

  const ev = active.current_event;
  eventLine.textContent = ev
    ? `⚠ ${ev.timestamp} · ${ev.event_type}`
    : `${active.fired_count}/${active.total_events} uyarı · ${mmss(active.position_sec)}/${mmss(active.duration_sec)}`;
  eventLine.className = `cell-event${ev ? ' cell-event-active' : ''}`;
}

/** Video konumunu sunucunun sanal zamanına hizalar (sapma büyükse). */
function syncVideo(video, positionSec) {
  const target = Number(positionSec) || 0;
  if (!Number.isFinite(video.duration) || video.duration <= 0) return;
  if (Math.abs(video.currentTime - target) > DRIFT_TOLERANCE_SEC) {
    try {
      video.currentTime = Math.min(target, Math.max(0, video.duration - 0.2));
    } catch { /* seek henüz mümkün değil */ }
  }
}

async function loadCameras() {
  try {
    const data = await api('/pseudolive/cameras');
    const active = data.filter((item) => item.active);

    el.gridEmpty.hidden = active.length > 0;
    el.libraryStatus.textContent = active.length
      ? `${active.length} kamera yayında`
      : 'Analiz kütüphanesi boş';

    const seen = new Set();
    for (const item of active) {
      seen.add(item.camera_id);
      renderCell(item.camera_id, item.active);
      cameras.set(item.camera_id, item.active);
    }
    for (const id of [...cameras.keys()]) {
      if (!seen.has(id)) {
        document.getElementById(`cell-${id}`)?.remove();
        cameras.delete(id);
      }
    }
  } catch (err) {
    el.libraryStatus.textContent = `Kameralar yüklenemedi: ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// Bildirimler
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
    title = `Araç çalıştı: ${data.tool_name || ''}`;
    detail = data.mock_result || data.status || '';
  } else if (stream === 'decision.final') {
    // Süpervizör paneli döngü başına ayrıca bir kart göstermez; hücre
    // güncellemesi yeterlidir, aksi hâlde her döngüde gereksiz kart birikir.
    return;
  } else {
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

  // Liste sınırsız büyümesin; en fazla 60 kart tutulur.
  const cards = el.notifications.querySelectorAll('.notification-card');
  if (cards.length > 60) cards[cards.length - 1].remove();
}

// ---------------------------------------------------------------------------
// Kamera detay penceresi (modal)
// ---------------------------------------------------------------------------

async function openModal(cameraId) {
  openCameraId = cameraId;
  el.modal.classList.add('open');
  try {
    const data = await api(`/pseudolive/cameras/${cameraId}`);
    if (!toolCatalog.length) {
      toolCatalog = await api('/tools').catch(() => []);
      populateManualToolSelect();
    }
    await loadRoles();
    renderModal(cameraId, data.active, data.analysis);
  } catch (err) {
    showToast(`Kamera açılamadı: ${err.message}`, true);
  }
}

function populateManualToolSelect() {
  el.manualTool.innerHTML = toolCatalog
    .filter((t) => t.enabled)
    .map((t) => `<option value="${escapeHtml(t.tool_name)}">${escapeHtml(t.description || t.tool_name)}</option>`)
    .join('');
}

function toolLabel(toolName) {
  const tool = toolCatalog.find((t) => t.tool_name === toolName);
  return tool?.description || toolName;
}

function renderModal(cameraId, active, analysis) {
  if (!analysis) return;

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

  el.fbCorrectRisk.value = analysis.risk || 'Düşük';
  el.fbCorrectSummary.value = analysis.summary || '';
  el.fbNotes.value = '';
  el.fbEditPanel.hidden = true;
  refreshFeedbackFor(analysis.slug);
  refreshAssignmentsFor(analysis.slug);

  if (el.modalVideo.dataset.camera !== cameraId) {
    el.modalVideo.dataset.camera = cameraId;
    el.modalVideo.src = `${API}/pseudolive/videos/${cameraId}`;
    el.modalVideo.addEventListener('loadedmetadata', () => {
      syncVideo(el.modalVideo, active.position_sec);
      el.modalVideo.play().catch(() => {});
    }, { once: true });
  }

  // Olay zaman çizelgesi: satıra tıklayınca video o ana gider.
  const stamps = analysis.event_timestamps || [];
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

  renderSuggestedActions(analysis, cameraId);

  // Atama paneli: hangi olayın atanacağı seçilebilir.
  el.assignEvent.innerHTML = stamps.length
    ? stamps.map((s, i) =>
        `<option value="${i}">${escapeHtml(s.timestamp)} · ${escapeHtml(s.event_type)} (${SEVERITY_LABEL[s.severity] || ''})</option>`
      ).join('')
    : '<option value="">Olay yok</option>';
  el.assignBtn.dataset.slug = analysis.slug || '';
  el.assignBtn.dataset.camera = cameraId;
  el.assignBtn.disabled = !analysis.slug;
  delete el.assignBtn.dataset.locked;
  el.assignBtn.classList.remove('btn-done');
  el.assignBtn.textContent = 'Ekibe Ata';

  startModalWindowWatch();
}

/**
 * Önerilen Aksiyonlar bölümünü doldurur: ajanın tetiklediği araçlar
 * (triggered_mock_tools) + yazdığı metin talimatlar (actions). Her araç
 * butonu gerçekten /tools/execute'a gider; tekrar tıklanamaz (runOnce).
 */
function renderSuggestedActions(analysis, cameraId) {
  const tools = analysis.triggered_mock_tools || [];
  el.suggestedActions.innerHTML = '';
  if (tools.length) {
    for (const call of tools) {
      const btn = document.createElement('button');
      btn.className = 'btn btn-secondary';
      btn.textContent = toolLabel(call.tool_name);
      btn.title = JSON.stringify(call.params || {});
      btn.addEventListener('click', () => runTool(btn, call.tool_name, call.params || {}, cameraId, analysis.slug, 'supervisor'));
      el.suggestedActions.appendChild(btn);
    }
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

/** Modal açıkken video zamanını izleyip modal başlığındaki risk rozetini
 * (ve ilerleme metnini) günceller. Kamera duvarındaki çerçeve zaten
 * loadCameras() döngüsüyle tazeleniyor; burada ek olarak modal içindeki
 * "ilerleme" satırı canlı tutulur. */
function startModalWindowWatch() {
  stopModalWindowWatch();
  modalWindowTimer = setInterval(() => {
    if (!openCameraId || !el.modal.classList.contains('open')) return;
    const cam = cameras.get(openCameraId);
    if (!cam) return;
    el.modalProgress.textContent =
      `${mmss(el.modalVideo.currentTime)} / ${mmss(cam.duration_sec)} · ${cam.fired_count}/${cam.total_events} uyarı`;
  }, WINDOW_CHECK_MS);
}

function stopModalWindowWatch() {
  if (modalWindowTimer) {
    clearInterval(modalWindowTimer);
    modalWindowTimer = null;
  }
}

// ---------------------------------------------------------------------------
// RLHF / DPO Geri Bildirim
// ---------------------------------------------------------------------------

function updateFeedbackUI(feedback) {
  if (!feedback) {
    el.feedbackBadge.className = 'fb-badge pending';
    el.feedbackBadge.textContent = 'Değerlendirilmedi';
    return;
  }
  if (feedback.feedback_type === 'correct') {
    el.feedbackBadge.className = 'fb-badge correct';
    el.feedbackBadge.textContent = '✔ Karar Doğrulandı';
  } else {
    el.feedbackBadge.className = 'fb-badge corrected';
    const typeMap = {
      false_positive: 'Yanlış Alarm',
      wrong_risk: 'Hatalı Risk',
      wrong_event: 'Hatalı Olay',
      wrong_action: 'Hatalı Aksiyon',
      other: 'Düzeltildi',
    };
    el.feedbackBadge.textContent = `⚠️ ${typeMap[feedback.feedback_type] || 'Düzeltildi'}`;
  }
}

async function refreshFeedbackFor(slug) {
  if (!slug) {
    updateFeedbackUI(null);
    return;
  }
  try {
    const rows = await api(`/feedback?analysis_slug=${encodeURIComponent(slug)}&limit=1`);
    updateFeedbackUI(rows.length ? rows[0] : null);
  } catch {
    updateFeedbackUI(null);
  }
}

async function submitFeedback(feedbackType, isCorrection = false) {
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
      geometric_signals: currentAnalysis.metadata?.geometric_signals || [],
    },
  };
  // Not: /feedback artık analiz başına UPSERT yapar (backend store.py); bu
  // yüzden burada ek bir "tekrar gönderme" koruması gerekmez, ama arayüz
  // tarafında da butonu geçici olarak kilitleriz ki art arda tıklamada
  // gereksiz istek gitmesin.
  try {
    const res = await api('/feedback', { method: 'POST', body: JSON.stringify(payload) });
    updateFeedbackUI(res);
    el.fbEditPanel.hidden = true;
    showToast(isCorrection ? '🎯 Düzeltme DPO havuzuna kaydedildi' : '✔ Karar DPO havuzunda doğrulandı');
  } catch (err) {
    showToast(`Geri bildirim kaydedilemedi: ${err.message}`, true);
  }
}

// ---------------------------------------------------------------------------
// Aksiyon çalıştırma (Önerilen + Manuel ortak yol)
// ---------------------------------------------------------------------------

async function runTool(btn, toolName, params, cameraId, slug, triggeredBy) {
  try {
    await runOnce(btn, async () => {
      const res = await api('/tools/execute', {
        method: 'POST',
        body: JSON.stringify({
          tool_name: toolName,
          params,
          camera_id: cameraId,
          analysis_slug: slug,
          triggered_by: triggeredBy,
        }),
      });
      if (res.already_executed) {
        showToast(`${res.tool_name} zaten çalıştırılmıştı.`);
      } else {
        showToast(`${res.tool_name}: ${res.mock_result || res.status}`);
        if (res.assignment) {
          showToast(`Görev oluşturuldu: ${res.assignment.role}`);
          refreshAssignmentsFor(slug);
        }
      }
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
  const note = el.manualNote.value.trim();
  try {
    await runOnce(el.manualRunBtn, async () => {
      const res = await api('/tools/execute', {
        method: 'POST',
        body: JSON.stringify({
          tool_name: toolName,
          params: { location: openCameraId, urgency: currentAnalysis.risk || 'Orta', reason: note || (currentAnalysis.headline || '') },
          camera_id: openCameraId,
          analysis_slug: currentAnalysis.slug,
          triggered_by: 'supervisor',
        }),
      });
      el.manualResult.textContent = res.already_executed
        ? `${res.tool_name} zaten çalıştırılmıştı.`
        : `${res.tool_name}: ${res.mock_result || res.status}`;
      if (res.assignment) {
        showToast(`Görev oluşturuldu: ${res.assignment.role}`);
        refreshAssignmentsFor(currentAnalysis.slug);
      }
      return { label: 'Çalıştırıldı ✓' };
    });
    // Manuel panel tekrar kullanılabilir olmalı (farklı araç seçilebilir);
    // yalnızca bu çalıştırma anındaki tekrar tıklamayı önlüyoruz.
    setTimeout(() => {
      el.manualRunBtn.disabled = false;
      el.manualRunBtn.classList.remove('btn-done');
      el.manualRunBtn.textContent = 'Çalıştır';
      delete el.manualRunBtn.dataset.locked;
    }, 1500);
  } catch (err) {
    showToast(`Araç çalıştırılamadı: ${err.message}`, true);
  }
}

async function refreshAssignmentsFor(slug) {
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
      await refreshAssignmentsFor(slug);
      return { label: '✓ Atandı' };
    });
  } catch (err) {
    showToast(`Atama başarısız: ${err.message}`, true);
  }
}

async function loadRoles() {
  try {
    const rolesList = await api('/roles');
    el.assignRole.innerHTML = rolesList
      .map((r) => `<option value="${escapeHtml(r.role)}">${escapeHtml(r.label)}</option>`)
      .join('');
  } catch {
    el.assignRole.innerHTML = '<option value="">Roller yüklenemedi</option>';
  }
}

function closeModal() {
  el.modal.classList.remove('open');
  openCameraId = null;
  stopModalWindowWatch();
  el.modalVideo.pause();
  el.modalVideo.removeAttribute('src');
  el.modalVideo.dataset.camera = '';
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

/** Modal açıkken WS mesajlarını hemen yeniden çekmek yerine biriktirip
 * debounce'lu tek bir tazeleme yapar. Eski davranış her mesajda modalin
 * tamamını yeniden çekip renderModal'ı yeniden çalıştırıyordu; bu, az önce
 * kilitlenmiş bir aksiyon butonunun "✓" durumunu DOM'dan siliyor ve operatörü
 * aynı aksiyonu tekrar tıklamaya yönlendiriyordu. */
let modalRefreshTimer = null;
function scheduleModalRefresh() {
  if (modalRefreshTimer) return;
  modalRefreshTimer = setTimeout(async () => {
    modalRefreshTimer = null;
    if (!openCameraId || !el.modal.classList.contains('open')) return;
    try {
      const d = await api(`/pseudolive/cameras/${openCameraId}`);
      // Yalnızca sayaç/olay/ilerleme alanlarını güncelle; aksiyon butonlarının
      // kilit durumunu korumak için renderModal'ın aksiyon bölümünü atlıyoruz
      // (renderModal zaten idempotent şekilde yeniden oluşturuyor, ama kilitli
      // butonlar "runOnce" ile zaten sunucu tarafında da idempotent olduğu
      // için yeniden oluşsalar da tekrar tıklamak zararsızdır — save_event
      // tekrar yazmaz, get_or_create_assignment tekrar satır açmaz).
      renderModal(openCameraId, d.active, d.analysis);
    } catch { /* modal kapanmış olabilir */ }
  }, 500);
}

function handleWsMessage(msg) {
  const { stream, data } = msg || {};
  if (!data) return;

  addNotification(stream, data);

  if (stream === 'event.detected' && data.camera_id) {
    const active = cameras.get(data.camera_id);
    if (active) {
      active.current_event = data;
      active.fired_count = (active.fired_count || 0) + 1;
      renderCell(data.camera_id, active);
    }
  } else if (stream === 'decision.final' && data.camera_id) {
    const active = cameras.get(data.camera_id);
    if (active) {
      active.risk = data.risk || active.risk;
      active.summary = data.summary || active.summary;
      active.current_event = null;
      active.fired_count = 0;
      renderCell(data.camera_id, active);
    }
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
  // Yoklama, sanal oynatma kafasını ve sayaçları tazeler; uyarılar WebSocket
  // üzerinden anında gelir. Ayrıca kamera çerçevesinin risk penceresi dışına
  // çıktığında sönmesi de bu yoklamayla gerçekleşir (server active_window'u
  // yeniden hesaplar).
  setInterval(loadCameras, 1000);

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WsClient(`${proto}://${location.host}/ws`);
  ws.connect();
  ws.onMessage(handleWsMessage);

  el.assignBtn.addEventListener('click', submitAssignment);
  el.manualRunBtn.addEventListener('click', runManualAction);

  el.fbBtnCorrect?.addEventListener('click', () => submitFeedback('correct', false));
  el.fbBtnToggleEdit?.addEventListener('click', () => {
    el.fbEditPanel.hidden = !el.fbEditPanel.hidden;
  });
  el.fbBtnCancelEdit?.addEventListener('click', () => {
    el.fbEditPanel.hidden = true;
  });
  el.fbBtnSubmitCorrection?.addEventListener('click', () => {
    const fbType = el.fbType.value || 'other';
    submitFeedback(fbType, true);
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
