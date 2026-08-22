import { WsClient } from './ws.js';

const kpiEvents = document.getElementById('kpi-events');
const kpiDecisions = document.getElementById('kpi-decisions');
const kpiTools = document.getElementById('kpi-tools');
const kpiNotifications = document.getElementById('kpi-notifications');
const eventsBody = document.getElementById('events-body');
const riskChart = document.getElementById('risk-chart');
const suggestBtn = document.getElementById('suggest-btn');
const queryInput = document.getElementById('query-text');
const suggestionsList = document.getElementById('suggestions-list');
const chatDrawer = document.getElementById('chat-drawer');
const chatTitle = document.getElementById('chat-title');
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatClose = document.getElementById('chat-close');
const toastEl = document.getElementById('toast');

const recentEvents = [];
let toastTimer = null;
let currentSuggestion = null;

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

function formatTime(date = new Date()) {
  return date.toLocaleTimeString('tr-TR');
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

function renderSuggestions(suggestions) {
  suggestionsList.innerHTML = '';
  if (!suggestions.length) {
    suggestionsList.innerHTML = '<div class="empty-state">Öneri bulunamadı.</div>';
    return;
  }
  for (const s of suggestions) {
    const card = document.createElement('div');
    card.className = 'post-it';
    card.innerHTML = `
      <h4>${escapeHtml(s.baslik)}</h4>
      <p><strong>Kategori:</strong> ${escapeHtml(s.kategori || '-')}</p>
      <p><strong>Öncelik:</strong> ${escapeHtml(s.oncelik || '-')} | <strong>Skor:</strong> ${escapeHtml(String(s.skor ?? '-'))}</p>
      <p><strong>Maliyet:</strong> ${escapeHtml(formatCost(s.maliyet_tahmini))}</p>
    `;
    card.addEventListener('click', () => openChat(s));
    suggestionsList.appendChild(card);
  }
}

function openChat(suggestion) {
  currentSuggestion = suggestion;
  currentSuggestion.messages = currentSuggestion.messages || [];
  chatTitle.textContent = suggestion.baslik;
  chatMessages.innerHTML = '';
  chatDrawer.classList.add('open');
  renderChatMessages();
}

function closeChat() {
  chatDrawer.classList.remove('open');
  currentSuggestion = null;
}

function renderChatMessages() {
  chatMessages.innerHTML = currentSuggestion.messages.map((m) => `
    <div class="chat-bubble ${m.role === 'user' ? 'user' : ''}">
      <strong>${m.role === 'user' ? 'Siz' : 'Asistan'}:</strong> ${escapeHtml(m.content)}
    </div>
  `).join('');
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendChatMessage(content) {
  if (!currentSuggestion) return;
  currentSuggestion.messages.push({ role: 'user', content });
  renderChatMessages();
  chatInput.value = '';

  try {
    const res = await fetch('/api/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ suggestion_id: currentSuggestion.oneri_id, messages: currentSuggestion.messages }),
    });
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    currentSuggestion.messages.push({ role: 'assistant', content: data.response || 'Yanıt yok.' });
    renderChatMessages();
  } catch (err) {
    currentSuggestion.messages.push({ role: 'assistant', content: 'Hata: ' + err.message });
    renderChatMessages();
  }
}

function handleWsMessage(msg) {
  const { stream, data } = msg || {};
  if (stream === 'event.detected' || stream === 'decision.final') {
    addEvent(stream, data || {});
  }
}

function init() {
  loadMetrics();
  setInterval(loadMetrics, 5000);

  const ws = new WsClient(`ws://${location.host}/ws`);
  ws.connect();
  ws.onMessage(handleWsMessage);

  suggestBtn.addEventListener('click', fetchSuggestions);
  window.addEventListener('resize', () => loadMetrics().then(() => {}));

  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;
    sendChatMessage(text);
  });

  chatClose.addEventListener('click', closeChat);
  document.addEventListener('click', (e) => {
    if (chatDrawer.classList.contains('open') && !chatDrawer.contains(e.target) && e.target !== suggestBtn) {
      // keep drawer open on outside click for usability
    }
  });
}

init();
