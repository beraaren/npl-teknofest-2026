import { WsClient } from './ws.js';
import { api, escapeHtml, riskClass, showToast, initToast } from './app.js';

const el = {
  kpiEvents: document.getElementById('kpi-events'),
  kpiDecisions: document.getElementById('kpi-decisions'),
  kpiTools: document.getElementById('kpi-tools'),
  kpiNotifications: document.getElementById('kpi-notifications'),
  kpiFeedbacks: document.getElementById('kpi-feedbacks'),
  kpiAccuracy: document.getElementById('kpi-accuracy'),
  feedbacksBody: document.getElementById('feedbacks-body'),
  eventsBody: document.getElementById('events-body'),
  riskChart: document.getElementById('risk-chart'),
  suggestBtn: document.getElementById('suggest-btn'),
  queryInput: document.getElementById('query-text'),
  commonConditions: document.getElementById('common-conditions'),
  conditionCost: document.getElementById('condition-cost'),
  suggestionsList: document.getElementById('suggestions-list'),
};
initToast(document.getElementById('toast'));

/** WS'ten gelen son 50 olay; öneri sorgusu için event_type listesi buradan
 * çıkarılır. */
const recentEvents = [];
let commonConditions = [];

function formatCost(cost) {
  if (!cost || cost.alt_sinir_tl == null || cost.ust_sinir_tl == null) return 'Bilinmiyor';
  return `${cost.alt_sinir_tl.toLocaleString('tr-TR')} - ${cost.ust_sinir_tl.toLocaleString('tr-TR')} ${cost.para_birimi || 'TL'}`;
}

function selectedCondition() {
  const text = el.queryInput.value.trim();
  return commonConditions.find((condition) => condition.baslik === text) || null;
}

function renderConditionCost() {
  const text = el.queryInput.value.trim();
  const condition = selectedCondition();
  el.conditionCost.classList.toggle('known', Boolean(condition));
  if (condition) {
    el.conditionCost.textContent = `Tahmini maliyet: ${formatCost(condition.maliyet_tahmini)}`;
  } else if (text) {
    el.conditionCost.textContent = 'Özel durum kabul edildi; tahmini maliyet: Bilinmiyor.';
  } else {
    el.conditionCost.textContent = 'Bir durum seçin veya kendi durumunuzu yazın; özel durumların maliyeti bilinmiyor olarak değerlendirilir.';
  }
}

async function loadCommonConditions() {
  try {
    const data = await api('/suggestions/common-conditions');
    commonConditions = Array.isArray(data) ? data : [];
    el.commonConditions.innerHTML = commonConditions.map((condition) =>
      `<option value="${escapeHtml(condition.baslik)}">${escapeHtml(formatCost(condition.maliyet_tahmini))}</option>`
    ).join('');
  } catch (err) {
    console.error('Yaygın durumlar yüklenemedi:', err);
  }
  renderConditionCost();
}

const FEEDBACK_TYPE_BADGE = {
  correct: '<span class="status-badge status-tamamlandi">✔ Doğrulandı</span>',
  false_positive: '<span class="status-badge status-atandi">✖ Yanlış Alarm</span>',
  wrong_risk: '<span class="status-badge status-goruldu">⚠ Hatalı Risk</span>',
  wrong_event: '<span class="status-badge status-goruldu">⚠ Hatalı Olay</span>',
  wrong_action: '<span class="status-badge status-goruldu">⚠ Hatalı Aksiyon</span>',
  other: '<span class="status-badge">Diğer Düzeltme</span>',
};

function formatTime(date = new Date()) {
  return date.toLocaleTimeString('tr-TR');
}

// ---------------------------------------------------------------------------
// KPI + risk grafiği
// ---------------------------------------------------------------------------

async function loadMetrics() {
  try {
    const data = await api('/metrics');
    el.kpiEvents.textContent = data.events_detected ?? '—';
    el.kpiDecisions.textContent = data.decisions_made ?? '—';
    el.kpiTools.textContent = data.tools_executed ?? '—';
    el.kpiNotifications.textContent = data.notifications_sent ?? '—';
    drawRiskChart(data.risk_distribution || {});
  } catch (err) {
    console.error('Metrikler yüklenemedi:', err);
  }
}

function drawRiskChart(distribution) {
  const canvas = el.riskChart;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.scale(dpr, dpr);

  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  // Risk seviyeleri kesin olarak üç değerdir (bkz. app.js RISK_CLASS).
  const labels = ['Düşük', 'Orta', 'Yüksek'];
  const colors = ['#22c55e', '#f59e0b', '#ef4444'];
  const values = labels.map((l) => Number(distribution[l]) || 0);
  const max = Math.max(...values, 1);

  const padding = 24;
  const chartW = width - padding * 2;
  const chartH = height - padding * 2;
  const step = chartW / labels.length;
  const barW = step * 0.5;

  ctx.font = '12px system-ui';
  ctx.textAlign = 'center';

  labels.forEach((label, i) => {
    const value = values[i];
    const barH = (value / max) * chartH;
    const x = padding + step * i + step / 2 - barW / 2;
    const y = padding + chartH - barH;

    ctx.fillStyle = colors[i];
    ctx.fillRect(x, y, barW, barH);

    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(String(value), x + barW / 2, y - 6);

    ctx.fillStyle = '#94a3b8';
    ctx.fillText(label, x + barW / 2, height - 6);
  });
}

// ---------------------------------------------------------------------------
// RLHF / DPO geri bildirim tablosu
// ---------------------------------------------------------------------------

async function loadFeedbackData() {
  try {
    const [stats, items] = await Promise.all([
      api('/feedback/stats'),
      api('/feedback?limit=25'),
    ]);
    el.kpiFeedbacks.textContent = stats.total ?? 0;
    el.kpiAccuracy.textContent = stats.total > 0 ? `%${stats.accuracy_rate}` : '%100';
    renderFeedbacksTable(items);
  } catch (err) {
    console.error('Geri bildirim verileri yüklenemedi:', err);
  }
}

function renderFeedbacksTable(items) {
  if (!items || !items.length) {
    el.feedbacksBody.innerHTML = '<tr><td colspan="6" class="empty-state">Henüz geri bildirim kaydı yok.</td></tr>';
    return;
  }
  el.feedbacksBody.innerHTML = items.map((f) => `
    <tr>
      <td>${escapeHtml(f.created_at ? new Date(f.created_at).toLocaleTimeString('tr-TR') : '-')}</td>
      <td><strong>${escapeHtml(f.analysis_slug || '-')}</strong> <span class="meta">(${escapeHtml(f.camera_id || '-')})</span></td>
      <td>${FEEDBACK_TYPE_BADGE[f.feedback_type] || escapeHtml(f.feedback_type)}</td>
      <td><span class="risk-badge ${riskClass(f.original_risk)}">${escapeHtml(f.original_risk || '-')}</span></td>
      <td><span class="risk-badge ${riskClass(f.corrected_risk)}">${escapeHtml(f.corrected_risk || '-')}</span></td>
      <td>${escapeHtml(f.supervisor_notes || f.corrected_summary || '-')}</td>
    </tr>
  `).join('');
}

// ---------------------------------------------------------------------------
// Son olaylar tablosu
// ---------------------------------------------------------------------------

function addEvent(stream, data) {
  recentEvents.unshift({ stream, time: new Date(), data });
  if (recentEvents.length > 50) recentEvents.pop();
  renderEventsTable();
}

function renderEventsTable() {
  if (!recentEvents.length) {
    el.eventsBody.innerHTML = '<tr><td colspan="5" class="empty-state">Henüz olay yok.</td></tr>';
    return;
  }
  el.eventsBody.innerHTML = recentEvents.map((e) => `
    <tr>
      <td>${escapeHtml(formatTime(e.time))}</td>
      <td>${escapeHtml(e.data.camera_id || '-')}</td>
      <td>${escapeHtml(e.stream)}</td>
      <td>${escapeHtml(e.data.event_type || '-')}</td>
      <td>${escapeHtml(e.data.risk || '-')}</td>
    </tr>
  `).join('');
}

function collectEventTypes() {
  const types = [];
  for (const e of recentEvents) {
    const t = e.data.event_type;
    if (t && !types.includes(t)) types.push(t);
  }
  return types;
}

// ---------------------------------------------------------------------------
// Çalışma alanı önerileri — seçilen veya serbest yazılan durumdan öneri üretir
// ---------------------------------------------------------------------------

async function fetchSuggestions() {
  const eventTypes = collectEventTypes();
  const queryText = el.queryInput.value.trim();
  const condition = selectedCondition();
  if (!eventTypes.length && !queryText) {
    showToast('Bir durum seçin/yazın veya en az bir olay bekleyin.');
    return;
  }
  const original = el.suggestBtn.textContent;
  el.suggestBtn.disabled = true;
  el.suggestBtn.textContent = 'Yükleniyor...';
  try {
    const data = await api('/suggestions/query', {
      method: 'POST',
      body: JSON.stringify({
        event_types: eventTypes,
        query_text: queryText,
        condition_id: condition?.condition_id || null,
      }),
    });
    renderConditionCostFromResponse(data?.selected_condition);
    renderSuggestions(Array.isArray(data?.suggestions) ? data.suggestions : []);
  } catch (err) {
    showToast(`Öneriler alınamadı: ${err.message}`, true);
  } finally {
    el.suggestBtn.disabled = false;
    el.suggestBtn.textContent = original;
  }
}

function renderConditionCostFromResponse(condition) {
  if (!condition) {
    renderConditionCost();
    return;
  }
  el.conditionCost.classList.toggle('known', Boolean(condition.cost_known));
  el.conditionCost.textContent = condition.cost_known
    ? `Tahmini maliyet: ${formatCost(condition.maliyet_tahmini)}`
    : 'Özel durum kabul edildi; tahmini maliyet: Bilinmiyor.';
}

/**
 * Öneriler düz metin satırları olarak render edilir: başlık, kategori,
 * öncelik, skor, maliyet. Hiçbiri tıklanamaz — post-it kartı ve LLM sohbet
 * çekmecesi (eski özellik) buraya KASITLI olarak eklenmedi.
 *
 * "Öncelik" burada önerinin öncelik seviyesidir (data/isg_onerileri.yaml
 * oncelik alanı) — kamera duvarındaki RİSK seviyesiyle aynı vokabüler
 * DEĞİLDİR, karışmaması için "Öncelik" olarak açıkça etiketlenir.
 */
function renderSuggestions(suggestions) {
  if (!suggestions.length) {
    el.suggestionsList.innerHTML = '<div class="empty-state">Öneri bulunamadı.</div>';
    return;
  }
  el.suggestionsList.innerHTML = suggestions.map((s) => `
    <div class="suggestion-row">
      <div class="suggestion-title">${escapeHtml(s.baslik)}</div>
      <div class="meta">
        <strong>Kategori:</strong> ${escapeHtml(s.kategori || '-')}
        · <strong>Öncelik:</strong> ${escapeHtml(s.oncelik || '-')}
        · <strong>Skor:</strong> ${escapeHtml(String(s.skor ?? '-'))}
      </div>
      <div class="meta"><strong>Maliyet:</strong> ${escapeHtml(formatCost(s.maliyet_tahmini))}</div>
    </div>
  `).join('');
}

// ---------------------------------------------------------------------------
// WebSocket + başlatma
// ---------------------------------------------------------------------------

function handleWsMessage(msg) {
  const { stream, data } = msg || {};
  if (stream === 'event.detected' || stream === 'decision.final') {
    addEvent(stream, data || {});
  } else if (stream === 'feedback.created') {
    loadFeedbackData();
    showToast('🎯 Yeni RLHF geri bildirimi kaydedildi.');
  }
}

function init() {
  loadMetrics();
  loadFeedbackData();
  loadCommonConditions();
  setInterval(() => {
    loadMetrics();
    loadFeedbackData();
  }, 5000);

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WsClient(`${proto}://${location.host}/ws`);
  ws.connect();
  ws.onMessage(handleWsMessage);

  el.suggestBtn.addEventListener('click', fetchSuggestions);
  el.queryInput.addEventListener('input', renderConditionCost);
  el.queryInput.addEventListener('focus', () => el.queryInput.showPicker?.());
  window.addEventListener('resize', loadMetrics);
}

init();
