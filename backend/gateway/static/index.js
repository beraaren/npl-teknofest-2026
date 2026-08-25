/**
 * Süpervizör ekranı — sözde-canlı kamera duvarı, olay uyarıları ve görev atama.
 *
 * ÇALIŞMA İLKESİ
 * Analizler önceden üretilmiştir; bu ekran canlı analiz yapmaz. Sunucu her
 * kamera için sanal bir oynatma kafası tutar ve uyarıları olayın videodaki
 * gerçek saniyesinde yayınlar. Bu dosya, video öğesini sunucunun bildirdiği
 * konuma (`position_sec`) hizalayarak canlı izlenimini kurar.
 *
 * MOCK KALDIRILDI
 * - Videolar artık gerçek kütüphane dosyaları (eskiden var olmayan bir klasör).
 * - Risk rozeti analizden gelir ve kalıcıdır (eskiden 6 saniyede sıfırlanıyordu).
 * - Aksiyon butonları sunucuda araç çalıştırır (eskiden yalnızca bildirim gösteriyordu).
 * - Atama, seçilen ekibe ve seçilen olaya yapılır (eskiden tüm rollere sabit gidiyordu).
 */
import { WsClient } from './ws.js';

const API = '/api/v1';

const RISK_CLASS = { 'Yüksek': 'high', 'Orta': 'medium', 'Düşük': 'low' };
const SEVERITY_LABEL = { critical: 'KRİTİK', high: 'YÜKSEK', medium: 'ORTA', low: 'DÜŞÜK' };
const SEVERITY_CLASS = { critical: 'high', high: 'high', medium: 'medium', low: 'low' };

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
  modalActions: document.getElementById('modal-actions'),
  assignRole: document.getElementById('assign-role'),
  assignEvent: document.getElementById('assign-event'),
  assignNote: document.getElementById('assign-note'),
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

/** camera_id -> sunucudan gelen son durum. */
const cameras = new Map();
let openCameraId = null;
let currentAnalysis = null;
let toastTimer = null;


// ---------------------------------------------------------------------------
// Yardımcılar
// ---------------------------------------------------------------------------

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

function riskClass(risk) {
  return RISK_CLASS[risk] || 'low';
}

function mmss(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

function clockTime(iso) {
  const d = iso ? new Date(iso) : new Date();
  return Number.isNaN(d.getTime())
    ? new Date().toLocaleTimeString('tr-TR')
    : d.toLocaleTimeString('tr-TR');
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

// ---------------------------------------------------------------------------
// Kamera duvarı
// ---------------------------------------------------------------------------

/**
 * Kamera hücresini oluşturur veya günceller.
 * Video kaynağı yalnızca döngü değiştiğinde yenilenir; her güncellemede
 * yeniden atanması oynatmayı baştan başlatır ve senkronu bozar.
 */
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

  const cls = riskClass(active.risk);
  badge.textContent = active.risk || '—';
  badge.className = `risk-badge ${cls}`;
  cell.classList.remove('risk-high', 'risk-medium', 'risk-low');
  cell.classList.add(`risk-${cls}`);

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

/**
 * Bildirim kartı ekler. Eskiden yalnızca akış adı ve başlık yazılıyordu;
 * artık olayın video içindeki anı, şiddeti ve kamerası görünür.
 */
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
    detail = `${data.event_type || ''} @ ${data.event_timestamp || ''}`;
  } else if (stream === 'assignment.created') {
    title = `Görev atandı: ${data.role}`;
    detail = `${data.headline || ''} @ ${data.event_timestamp || ''}`;
  } else if (stream === 'tool.executed') {
    title = `Araç çalıştı: ${data.tool_name}`;
    detail = data.mock_result || '';
  } else {
    return; // decision.final duvarı zaten güncelliyor, bildirim gürültüsü yapmaz
  }

  const card = document.createElement('div');
  card.className = `notification-card severity-${cls}`;
  card.innerHTML = `
    <div class="notif-head">
      <span class="notif-severity ${cls}">${SEVERITY_LABEL[severity] || ''}</span>
      <span class="notif-clock">${escapeHtml(clockTime(data.created_at))}</span>
    </div>
    <div class="notif-title">${escapeHtml(title)}</div>
    ${detail ? `<div class="meta">${escapeHtml(detail)}</div>` : ''}
    <div class="meta">Kamera: ${escapeHtml(data.camera_id || '-')}</div>
  `;
  if (data.camera_id) {
    card.style.cursor = 'pointer';
    card.addEventListener('click', () => openModal(data.camera_id));
  }

  el.notifications.prepend(card);
  while (el.notifications.children.length > 60) {
    el.notifications.lastElementChild.remove();
  }
}

// ---------------------------------------------------------------------------
// Kamera detayı + atama
// ---------------------------------------------------------------------------

async function openModal(cameraId) {
  openCameraId = cameraId;
  el.modal.classList.add('open');
  el.modalTitle.textContent = cameraId;
  el.modalSummary.textContent = 'Yükleniyor…';
  el.modalEvents.innerHTML = '';
  el.modalActions.innerHTML = '';

  try {
    const data = await api(`/pseudolive/cameras/${cameraId}`);
    renderModal(cameraId, data.active, data.analysis);
    await refreshAssignmentsFor(data.analysis?.slug);
    await refreshFeedbackFor(data.analysis?.slug);
  } catch (err) {
    el.modalSummary.textContent = `Detay alınamadı: ${err.message}`;
  }
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

  // Ajanın yazdığı olay özeti — saha ekibi de aynı metni görür.
  el.modalSummary.textContent = analysis.summary || 'Özet üretilmedi.';
  el.modalReasoning.textContent = analysis.reasoning || 'Gerekçe kaydı yok.';

  // Geri bildirim formu alanlarını varsayılana hazırla
  el.fbCorrectRisk.value = analysis.risk || 'Düşük';
  el.fbCorrectSummary.value = analysis.summary || '';
  el.fbNotes.value = '';
  el.fbEditPanel.hidden = true;

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
    ? stamps.map((s, i) => `
        <li class="timeline-item severity-${SEVERITY_CLASS[s.severity] || 'low'}" data-seconds="${s.seconds}">
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

  // Aksiyonlar: karar ajanının seçtiği araçlar gerçekten çalıştırılır.
  const tools = analysis.triggered_mock_tools || [];
  el.modalActions.innerHTML = '';
  if (tools.length) {
    for (const call of tools) {
      const btn = document.createElement('button');
      btn.className = 'btn btn-secondary';
      btn.textContent = call.tool_name;
      btn.title = JSON.stringify(call.params || {});
      btn.addEventListener('click', () => executeTool(btn, call, cameraId, analysis.slug));
      el.modalActions.appendChild(btn);
    }
  }
  // Ajanın metin olarak yazdığı öneriler (araç değil, operatör talimatı).
  if ((analysis.actions || []).length) {
    const list = document.createElement('ul');
    list.className = 'action-advice';
    list.innerHTML = analysis.actions.map((a) => `<li>${escapeHtml(a)}</li>`).join('');
    el.modalActions.appendChild(list);
  }
  if (!tools.length && !(analysis.actions || []).length) {
    el.modalActions.innerHTML = '<span class="meta">Aksiyon önerisi üretilmedi.</span>';
  }

  // Atama paneli: hangi olayın atanacağı seçilebilir.
  el.assignEvent.innerHTML = stamps.length
    ? stamps.map((s, i) =>
        `<option value="${i}">${escapeHtml(s.timestamp)} · ${escapeHtml(s.event_type)} (${SEVERITY_LABEL[s.severity] || ''})</option>`
      ).join('')
    : '<option value="">Olay yok</option>';
  el.assignBtn.dataset.slug = analysis.slug || '';
  el.assignBtn.dataset.camera = cameraId;
  el.assignBtn.disabled = !analysis.slug;
}

// ---------------------------------------------------------------------------
// RLHF / DPO Geri Bildirim İşleyicileri
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

  try {
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

    const res = await api('/feedback', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    updateFeedbackUI(res);
    el.fbEditPanel.hidden = true;
    showToast(isCorrection ? '🎯 Düzeltme DPO havuzuna kaydedildi' : '✔ Karar DPO havuzunda doğrulandı');
  } catch (err) {
    showToast(`Geri bildirim kaydedilemedi: ${err.message}`, true);
  }
}


async function executeTool(btn, call, cameraId, slug) {
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = `${original} …`;
  try {
    const res = await api('/tools/execute', {
      method: 'POST',
      body: JSON.stringify({
        tool_name: call.tool_name,
        params: call.params || {},
        camera_id: cameraId,
        analysis_slug: slug,
        triggered_by: 'supervisor',
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

  el.assignBtn.disabled = true;
  try {
    const eventIndex = el.assignEvent.value === '' ? null : Number(el.assignEvent.value);
    const row = await api('/assignments', {
      method: 'POST',
      body: JSON.stringify({
        analysis_slug: slug,
        role,
        camera_id: cameraId,
        event_index: eventIndex,
        note: el.assignNote.value.trim(),
      }),
    });
    showToast(`Görev atandı: ${row.role} · ${row.event_type} @ ${row.event_timestamp}`);
    el.assignNote.value = '';
    await refreshAssignmentsFor(slug);
  } catch (err) {
    showToast(`Atama başarısız: ${err.message}`, true);
  } finally {
    el.assignBtn.disabled = false;
  }
}

function closeModal() {
  el.modal.classList.remove('open');
  openCameraId = null;
  el.modalVideo.pause();
  el.modalVideo.removeAttribute('src');
  el.modalVideo.dataset.camera = '';
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function handleWsMessage(msg) {
  const { stream, data } = msg || {};
  if (!data) return;

  addNotification(stream, data);

  // Duvardaki hücreyi anında güncelle (sonraki yoklamayı beklemeden).
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
      // Yeni döngü başladı: risk ve özet değişti.
      active.risk = data.risk || active.risk;
      active.summary = data.summary || active.summary;
      active.current_event = null;
      active.fired_count = 0;
      renderCell(data.camera_id, active);
    }
  }

  if (openCameraId && data.camera_id === openCameraId && el.modal.classList.contains('open')) {
    // Detay penceresi açıkken sayaçlar/olaylar tazelenir.
    api(`/pseudolive/cameras/${openCameraId}`)
      .then((d) => renderModal(openCameraId, d.active, d.analysis))
      .catch(() => {});
  }
}

// ---------------------------------------------------------------------------
// Başlatma
// ---------------------------------------------------------------------------

async function loadRoles() {
  try {
    const roles = await api('/roles');
    el.assignRole.innerHTML = roles
      .map((r) => `<option value="${escapeHtml(r.role)}">${escapeHtml(r.label)}</option>`)
      .join('');
  } catch {
    el.assignRole.innerHTML = '<option value="">Roller yüklenemedi</option>';
  }
}

async function init() {
  await loadRoles();
  await loadCameras();
  // Yoklama, sanal oynatma kafasını ve sayaçları tazeler; uyarılar WebSocket
  // üzerinden anında gelir.
  setInterval(loadCameras, 4000);

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WsClient(`${proto}://${location.host}/ws`);
  ws.connect();
  ws.onMessage(handleWsMessage);

  el.assignBtn.addEventListener('click', submitAssignment);

  // RLHF / DPO Geri Bildirim Butonları
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
