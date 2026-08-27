import { WsClient } from './ws.js';

const kpiEvents = document.getElementById('kpi-events');
const kpiDecisions = document.getElementById('kpi-decisions');
const kpiTools = document.getElementById('kpi-tools');
const kpiNotifications = document.getElementById('kpi-notifications');
const kpiFeedbacks = document.getElementById('kpi-feedbacks');
const kpiAccuracy = document.getElementById('kpi-accuracy');
const feedbacksBody = document.getElementById('feedbacks-body');
const eventsBody = document.getElementById('events-body');
const riskChart = document.getElementById('risk-chart');
const suggestBtn = document.getElementById('suggest-btn');
const queryInput = document.getElementById('query-text');
const suggestionsList = document.getElementById('suggestions-list');
const toastEl = document.getElementById('toast');

const recentEvents = [];
let toastTimer = null;

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 3000);
}

function formatTime(date = new Date()) {
  return date.toLocaleTimeString('tr-TR');
}

async function loadFeedbackData() {
  try {
    const [statsRes, listRes] = await Promise.all([
      fetch('/api/v1/feedback/stats'),
      fetch('/api/v1/feedback?limit=25'),
    ]);

    if (statsRes.ok && kpiFeedbacks && kpiAccuracy) {
      const stats = await statsRes.json();
      kpiFeedbacks.textContent = stats.total ?? 0;
      kpiAccuracy.textContent = stats.total > 0 ? `%${stats.accuracy_rate}` : '%100';
    }

    if (listRes.ok && feedbacksBody) {
      const items = await listRes.json();
      renderFeedbacksTable(items);
    }
  } catch (err) {
    console.error('Geri bildirim verileri yüklenemedi:', err);
  }
}

function renderFeedbacksTable(items) {
  if (!feedbacksBody) return;
  if (!items || !items.length) {
    feedbacksBody.innerHTML = '<tr><td colspan="6" class="empty-state">Henüz geri bildirim kaydı yok.</td></tr>';
    return;
  }

  const typeLabels = {
    correct: '<span class="status-badge status-tamamlandi">✔ Doğrulandı</span>',
    false_positive: '<span class="status-badge status-atandi">✖ Yanlış Alarm</span>',
    wrong_risk: '<span class="status-badge status-goruldu">⚠ Hatalı Risk</span>',
    wrong_event: '<span class="status-badge status-goruldu">⚠ Hatalı Olay</span>',
    wrong_action: '<span class="status-badge status-goruldu">⚠ Hatalı Aksiyon</span>',
    other: '<span class="status-badge">Diğer Düzeltme</span>',
  };

  feedbacksBody.innerHTML = items.map((f) => `
    <tr>
      <td>${escapeHtml(f.created_at ? new Date(f.created_at).toLocaleTimeString('tr-TR') : '-')}</td>
      <td><strong>${escapeHtml(f.analysis_slug || '-')}</strong> <span class="meta">(${escapeHtml(f.camera_id || '-')})</span></td>
      <td>${typeLabels[f.feedback_type] || escapeHtml(f.feedback_type)}</td>
      <td><span class="risk-badge ${(f.original_risk || '').toLowerCase() === 'yüksek' ? 'high' : (f.original_risk || '').toLowerCase() === 'orta' ? 'medium' : 'low'}">${escapeHtml(f.original_risk || '-')}</span></td>
      <td><span class="risk-badge ${(f.corrected_risk || '').toLowerCase() === 'yüksek' ? 'high' : (f.corrected_risk || '').toLowerCase() === 'orta' ? 'medium' : 'low'}">${escapeHtml(f.corrected_risk || '-')}</span></td>
      <td>${escapeHtml(f.supervisor_notes || f.corrected_summary || '-')}</td>
    </tr>
  `).join('');
}

async function loadMetrics() {
  try {
    const res = await fetch('/api/v1/metrics');
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    kpiEvents.textContent = data.events_detected ?? '—';
    kpiDecisions.textContent = data.decisions_made ?? '—';
    kpiTools.textContent = data.tools_executed ?? '—';
    kpiNotifications.textContent = data.notifications_sent ?? '—';
    drawRiskChart(data.risk_distribution || {});
  } catch (err) {
    console.error('Metrikler yüklenemedi:', err);
  }
}


function drawRiskChart(distribution) {
  const canvas = riskChart;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.scale(dpr, dpr);

  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  const labels = ['Düşük', 'Orta', 'Yüksek'];
  const colors = ['#22c55e', '#f59e0b', '#ef4444'];
  const values = labels.map((l) => Number(distribution[l]) || 0);
  const max = Math.max(...values, 1);

  const padding = 24;
  const chartW = width - padding * 2;
  const chartH = height - padding * 2;
  const barW = chartW / labels.length * 0.5;
  const step = chartW / labels.length;

  ctx.fillStyle = '#94a3b8';
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

function addEvent(stream, data) {
  recentEvents.unshift({ stream, time: new Date(), data });
  if (recentEvents.length > 50) recentEvents.pop();
  renderEventsTable();
}

function renderEventsTable() {
  if (!recentEvents.length) {
    eventsBody.innerHTML = '<tr><td colspan="5" class="empty-state">Henüz olay yok.</td></tr>';
    return;
  }
  eventsBody.innerHTML = recentEvents.map((e) => `
    <tr>
      <td>${escapeHtml(formatTime(e.time))}</td>
      <td>${escapeHtml(e.data.camera_id || '-')}</td>
      <td>${escapeHtml(e.stream)}</td>
      <td>${escapeHtml(e.data.event_type || e.data.current_event || '-')}</td>
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

async function fetchSuggestions() {
  const eventTypes = collectEventTypes();
  if (!eventTypes.length) {
    showToast('Henüz olay tipi yok.');
    return;
  }
  suggestBtn.disabled = true;
  suggestBtn.textContent = 'Yükleniyor...';
  try {
    const res = await fetch('/api/v1/suggestions/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event_types: eventTypes, query_text: queryInput.value || '' }),
    });
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    renderSuggestions(Array.isArray(data) ? data : []);
  } catch (err) {
    showToast('Öneriler alınamadı: ' + err.message);
  } finally {
    suggestBtn.disabled = false;
    suggestBtn.textContent = 'Seçili olaylar için öneri al';
  }
}

function formatCost(cost) {
  if (!cost) return '-';
  return `${cost.alt_sinir_tl || 0} - ${cost.ust_sinir_tl || 0} ${cost.para_birimi || 'TL'}`;
}

/**
 * Öneriler artık düz metin satırları olarak gösterilir; tıklama, post-it
 * kartı ve LLM sohbet çekmecesi kaldırıldı (özellik gereksiz yere
 * ağırlaştırılmıştı ve burada bir yorum/sohbet bölümüne ihtiyaç yok).
 * "Öncelik" burada önerinin öncelik seviyesidir (data/isg_onerileri.yaml
 * oncelik alanı) — kamera duvarındaki RİSK seviyesiyle aynı vokabüler
 * DEĞİLDİR; karışmaması için ayrıca etiketlenir.
 */
function renderSuggestions(suggestions) {
  suggestionsList.innerHTML = '';
  if (!suggestions.length) {
    suggestionsList.innerHTML = '<div class="empty-state">Öneri bulunamadı.</div>';
    return;
  }
  suggestionsList.innerHTML = suggestions.map((s) => `
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
  setInterval(() => {
    loadMetrics();
    loadFeedbackData();
  }, 5000);

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WsClient(`${proto}://${location.host}/ws`);
  ws.connect();
  ws.onMessage(handleWsMessage);

  suggestBtn.addEventListener('click', fetchSuggestions);
  window.addEventListener('resize', () => loadMetrics().then(() => {}));
}

init();
