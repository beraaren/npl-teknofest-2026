import { WsClient } from './ws.js';
import {
  API, api, escapeHtml, mmss, riskClass, showToast, initToast, runOnce, SEVERITY_CLASS, SEVERITY_LABEL
} from './app.js';

// OpenCV BGR tuple'ları → CSS rgb(...). BGR'de ilk değer Mavi, son Kırmızı.
const CLASS_COLORS = {
  arac: 'rgb(0,140,255)',
  insan: 'rgb(127,255,0)',
  palet: 'rgb(255,215,0)',
  baret: 'rgb(0,191,255)',
  yelek: 'rgb(50,205,50)',
  yangin: 'rgb(255,0,0)',
  duman: 'rgb(180,180,180)',
  unknown: 'rgb(200,200,200)',
};

const RELATION_COLORS = {
  near: 'rgb(255,70,0)',
  wearing: 'rgb(0,255,0)',
  carrying: 'rgb(0,255,255)',
};

const STEP_ORDER = ['ingest', 'perception', 'vlm', 'decision'];

const el = {
  dropZone: document.getElementById('drop-zone'),
  uploadSection: document.getElementById('upload-section'),
  videoInput: document.getElementById('video-input'),
  progressSection: document.getElementById('progress-section'),
  progressFill: document.getElementById('progress-fill'),
  progressSteps: document.getElementById('progress-steps'),
  videoSection: document.getElementById('video-section'),
  video: document.getElementById('demo-video'),
  canvas: document.getElementById('overlay-canvas'),
  videoMeta: document.getElementById('video-meta'),
  decisionSummary: document.getElementById('decision-summary'),
  modalRisk: document.getElementById('modal-risk'),
  modalEvidence: document.getElementById('modal-evidence'),
  modalProgress: document.getElementById('modal-progress'),
  modalSummary: document.getElementById('modal-summary'),
  modalReasoning: document.getElementById('modal-reasoning'),
  modalEvents: document.getElementById('modal-events'),
  modalFindings: document.getElementById('modal-findings'),
  modalUncertain: document.getElementById('modal-uncertain'),
  suggestedActions: document.getElementById('suggested-actions'),

  // Kanal tab'ları
  demoTabs: document.getElementById('demo-tabs'),
  vlmMeta: document.getElementById('vlm-meta'),
  vlmSummary: document.getElementById('vlm-summary'),
  vlmRisks: document.getElementById('vlm-risks'),
  vlmEntities: document.getElementById('vlm-entities'),
  vlmActions: document.getElementById('vlm-actions'),
  ragVerified: document.getElementById('rag-verified'),
  ragUnverified: document.getElementById('rag-unverified'),
  
  // Manuel aksiyon
  manualTool: document.getElementById('manual-tool'),
  manualNote: document.getElementById('manual-note'),
  manualRunBtn: document.getElementById('manual-run-btn'),
  manualResult: document.getElementById('manual-result'),
  
  // Ekip atama
  assignRole: document.getElementById('assign-role'),
  assignEvent: document.getElementById('assign-event'),
  assignBtn: document.getElementById('assign-btn'),
  assignExisting: document.getElementById('assign-existing'),

  // RLHF Feedback
  fbBadge: document.getElementById('feedback-badge'),
  fbQuickActions: document.getElementById('fb-quick-actions'),
  fbBtnCorrect: document.getElementById('fb-btn-correct'),
  fbBtnToggleEdit: document.getElementById('fb-btn-toggle-edit'),
  fbEditPanel: document.getElementById('fb-edit-panel'),
  fbType: document.getElementById('fb-type'),
  fbCorrectRisk: document.getElementById('fb-correct-risk'),
  fbCorrectSummary: document.getElementById('fb-correct-summary'),
  fbNotes: document.getElementById('fb-notes'),
  fbBtnSubmitCorrection: document.getElementById('fb-btn-submit-correction'),
  fbBtnCancelEdit: document.getElementById('fb-btn-cancel-edit'),

  demoStatus: document.getElementById('demo-status'),
  toast: document.getElementById('toast'),
};
initToast(el.toast);

let currentJobId = null;
let uploadedFile = null;
let currentEvents = [];
let vlmInterpretation = null;
let decisionPayload = null;
let wsClient = null;
let pollInterval = null;
let completedSteps = new Set();
let toolCatalog = [];
const executedTools = new Set();
let fakeProgressValue = 0;
let fakeProgressTimer = null;

function startFakeProgress() {
  fakeProgressValue = 0;
  if (fakeProgressTimer) clearInterval(fakeProgressTimer);
  el.progressFill.style.width = '0%';
  
  fakeProgressTimer = setInterval(() => {
    if (fakeProgressValue < 85) {
      fakeProgressValue += (85 - fakeProgressValue) * 0.05 + 0.5;
    } else if (fakeProgressValue < 99) {
      fakeProgressValue += 0.1;
    }
    el.progressFill.style.width = `${Math.min(99.5, fakeProgressValue)}%`;
  }, 100);
}

function stopFakeProgress() {
  if (fakeProgressTimer) {
    clearInterval(fakeProgressTimer);
    fakeProgressTimer = null;
  }
}

const toolLabel = (name) => {
  const tool = toolCatalog.find((t) => t.tool_name === name);
  if (!tool) return name;
  const label = tool.description || tool.tool_name || name;
  return label.charAt(0).toUpperCase() + label.slice(1);
};

// ---------------------------------------------------------------------------
// MM:SS ↔ saniye dönüşümleri
// ---------------------------------------------------------------------------

function parseMmss(ts) {
  if (typeof ts !== 'string' || !ts.includes(':')) return 0;
  const [m, s] = ts.split(':');
  return (parseInt(m, 10) || 0) * 60 + (parseInt(s, 10) || 0);
}

function formatConfidence(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return `%${(n * 100).toFixed(0)}`;
}

// ---------------------------------------------------------------------------
// Upload + sürükle-bırak
// ---------------------------------------------------------------------------

function initUpload() {
  el.dropZone.addEventListener('click', () => el.videoInput.click());

  el.videoInput.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) startUpload(file);
  });

  ['dragenter', 'dragover'].forEach((name) => {
    el.uploadSection.addEventListener(name, (e) => {
      e.preventDefault();
      el.uploadSection.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach((name) => {
    el.uploadSection.addEventListener(name, (e) => {
      e.preventDefault();
      el.uploadSection.classList.remove('drag-over');
    });
  });

  el.uploadSection.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) startUpload(file);
    else if (file) showToast('Lütfen bir video dosyası yükleyin.', true);
  });
}

async function startUpload(file) {
  uploadedFile = file;
  currentJobId = null;
  currentEvents = [];
  vlmInterpretation = null;
  decisionPayload = null;
  completedSteps = new Set();
  resetUI();

  el.demoStatus.textContent = 'Yükleniyor…';
  const formData = new FormData();
  formData.append('video', file);
  formData.append('camera_id', 'demo_upload');

  try {
    const res = await fetch(`${API}/analyses/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch { /* ignore */ }
      throw new Error(detail);
    }
    const data = await res.json();
    currentJobId = data.job_id;
    el.demoStatus.textContent = `Analiz başlatıldı: ${currentJobId.slice(0, 8)}`;
    showToast('Video yüklendi; analiz başlatıldı.');
    el.progressSection.hidden = false;
    startFakeProgress();
    startWatching(currentJobId);
  } catch (err) {
    el.demoStatus.textContent = 'Yükleme başarısız';
    showToast(`Yükleme hatası: ${err.message}`, true);
  }
}

function resetUI() {
  stopFakeProgress();
  el.progressFill.style.width = '0%';
  el.progressSection.hidden = true;
  el.videoSection.hidden = true;
  el.canvas.getContext('2d')?.clearRect(0, 0, el.canvas.width, el.canvas.height);
  el.modalRisk.textContent = '—';
  el.modalRisk.className = 'risk-badge';
  el.modalEvidence.textContent = 'Kanıt: —';
  el.modalSummary.innerHTML = 'Özet bekleniyor…';
  el.modalReasoning.innerHTML = '';
  el.modalEvents.innerHTML = '';
  el.modalFindings.innerHTML = '';
  el.modalUncertain.innerHTML = '';
  el.suggestedActions.innerHTML = '';
  el.manualResult.textContent = '';
  el.assignExisting.textContent = '';
  el.vlmMeta.textContent = '';
  el.vlmSummary.textContent = '—';
  el.vlmRisks.innerHTML = '';
  el.vlmEntities.innerHTML = '';
  el.vlmActions.innerHTML = '';
  el.ragVerified.innerHTML = '';
  el.ragUnverified.innerHTML = '';
  // Tab'ı başa döndür
  el.demoTabs.querySelectorAll('.demo-tab').forEach((b, i) => b.classList.toggle('active', i === 0));
  document.querySelectorAll('#decision-summary .tab-pane').forEach((p, i) => p.classList.toggle('active', i === 0));
  
  el.fbBadge.className = 'fb-badge pending';
  el.fbBadge.textContent = 'Değerlendirilmedi';
  el.fbEditPanel.hidden = true;
  el.fbNotes.value = '';
  executedTools.clear();
  updateStepUI();
}

// ---------------------------------------------------------------------------
// Pipeline progress + WebSocket
// ---------------------------------------------------------------------------

function updateStepUI() {
  el.progressSteps.querySelectorAll('span').forEach((span) => {
    const step = span.dataset.step;
    span.classList.remove('done', 'active');
    if (completedSteps.has(step)) {
      span.classList.add('done');
      span.textContent = span.textContent.replace(/^⏳\s*/, '✓ ');
    } else if (step === nextPendingStep()) {
      span.classList.add('active');
      if (!span.textContent.startsWith('⏳') && !span.textContent.startsWith('✓')) {
        span.textContent = '⏳ ' + span.textContent.replace(/^[✓⏳]\s*/, '');
      }
    }
  });
}

function nextPendingStep() {
  return STEP_ORDER.find((s) => !completedSteps.has(s));
}

function stopPolling() {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}

function startWatching(jobId) {
  if (wsClient) {
    wsClient.close();
    wsClient = null;
  }
  stopPolling();

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  wsClient = new WsClient(`${proto}://${location.host}/ws`);
  wsClient.connect();
  wsClient.onMessage(handleWsMessage);

  // Güvenlik ağı: decision.final kaçırılırsa polling ile sonucu çek.
  // jobId kapanışta tutulur; önceki bir upload'ın zamanlayıcısı yeni işi
  // yanlışlıkla tamamlayamaz.
  pollInterval = setInterval(async () => {
    if (decisionPayload || currentJobId !== jobId) {
      stopPolling();
      return;
    }
    try {
      const analysis = await api(`/analyses/${jobId}`);
      if (analysis) {
        const response = await api(`/analyses/${jobId}/events`);
        finalize(analysis, response.events || []);
        stopPolling();
      }
    } catch { /* henüz hazır değil */ }
  }, 2500);
}

function handleWsMessage(msg) {
  const { stream, data } = msg || {};
  if (!data || data.job_id !== currentJobId) return;

  if (stream === 'frame.chunk') {
    completedSteps.add('ingest');
    updateStepUI();
  } else if (stream === 'event.detected') {
    completedSteps.add('ingest');
    completedSteps.add('perception');
    currentEvents.push({ stream, data });
    updateStepUI();
  } else if (stream === 'vlm.interpreted') {
    completedSteps.add('ingest');
    completedSteps.add('perception');
    completedSteps.add('vlm');
    vlmInterpretation = data.interpretation || {};
    updateStepUI();
  } else if (stream === 'decision.final') {
    completedSteps.add('ingest');
    completedSteps.add('perception');
    completedSteps.add('vlm');
    completedSteps.add('decision');
    decisionPayload = data;
    updateStepUI();
    // Events tablosundan tüm kayıtları çek ve ekranı doldur
    fetchAndFinalize();
  }
}

async function fetchAndFinalize() {
  if (!currentJobId) return;
  try {
    const [analysis, events] = await Promise.all([
      api(`/analyses/${currentJobId}`),
      api(`/analyses/${currentJobId}/events`).catch(() => ({ events: [] })),
    ]);
    finalize(analysis || decisionPayload, events.events || []);
  } catch (err) {
    showToast(`Sonuçlar alınamadı: ${err.message}`, true);
  }
}

// ---------------------------------------------------------------------------
// Nihai ekranı doldur
// ---------------------------------------------------------------------------

function finalize(analysis, events) {
  if (!analysis) return;

  stopFakeProgress();
  el.progressFill.style.width = '100%';
  setTimeout(() => {
    el.progressSection.hidden = true;
  }, 800);

  // WebSocket kaçırılırsa REST yedeğinden gelen olayları da overlay ve
  // zaman çizelgesinin kullandığı kanonik koleksiyona al.
  currentEvents = Array.isArray(events) ? events : [];

  // Yerel video kaynağı (upload edilen dosya)
  if (uploadedFile) {
    el.video.src = URL.createObjectURL(uploadedFile);
    el.video.load();
    el.video.play().catch(e => console.log('Oynatma hatası:', e));
  }
  el.videoSection.hidden = false;
  el.decisionSummary.hidden = false;
  el.demoStatus.textContent = 'Analiz tamamlandı';

  renderDecision(analysis);
  refreshFeedbackBadge();
  refreshAssignments();
}

function parseSeconds(ts) {
  if (typeof ts === 'number') return ts;
  return parseMmss(ts);
}

function renderDecision(analysis) {
  el.modalRisk.textContent = analysis.risk || '—';
  el.modalRisk.className = `risk-badge ${riskClass(analysis.risk)}`;
  const results = analysis.results || [];
  const agreements = [...new Set(results.map((r) => r?.evidence?.agreement).filter(Boolean))];
  el.modalEvidence.textContent = analysis.uncertain
    ? `İnceleme gerekli: ${analysis.uncertainty_reason || 'Kanıt yetersiz.'}`
    : `Kanıt: ${agreements.length ? agreements.join(', ') : 'Yerel Demo'}`;
  
  el.modalSummary.innerHTML = analysis.summary || 'Özet üretilmedi.';
  el.modalReasoning.innerHTML = analysis.reasoning || 'Gerekçe kaydı yok.';
  
  renderTimeline(analysis.events || []);
  renderContextualResults(results);
  renderChannelB(analysis.vlm_interpretation || vlmInterpretation);
  renderRag(analysis.rag_context || {});
  renderSuggestedActions(analysis);
  fillAssignEventSelect(analysis.events || []);
  
  el.fbCorrectRisk.value = analysis.risk || 'Düşük';
  el.fbCorrectSummary.value = analysis.summary || '';
}

function renderTimeline(stamps) {
  el.modalEvents.innerHTML = stamps.length
    ? stamps.map((s) => `
        <li class="timeline-item severity-${SEVERITY_CLASS[s.severity] || 'low'}">
          <span class="timeline-time">${escapeHtml(s.time || s.timestamp || '00:00')}</span>
          <span class="timeline-type">${escapeHtml(s.event_type)}</span>
          <span class="timeline-sev ${SEVERITY_CLASS[s.severity] || 'low'}">${SEVERITY_LABEL[s.severity] || 'Belirsiz'}</span>
          <span class="meta">${escapeHtml(s.event || s.description || '')}</span>
          <span class="meta">Güven: ${formatConfidence(s.confidence)}</span>
        </li>`).join('')
    : '<li class="empty-state">Tespit edilen olay yok.</li>';
}

function renderContextualResults(results) {
  const findings = results.filter((r) => r?.result_type === 'contextual_finding');
  const uncertain = results.filter((r) => r?.result_type === 'uncertain_observation');
  
  const render = (target, rows, emptyText) => {
    target.innerHTML = rows.length
      ? rows.map((r) => {
        const agreement = r?.evidence?.agreement || 'bilinmiyor';
        const detail = r.uncertain ? (r.uncertainty_reason || 'Kanıt yetersiz.') : (r.hazard_mechanism || r.event || 'Açıklama yok.');
        const timeLabel = r.time || r.timestamp || '—';
        return `<li class="timeline-item severity-${SEVERITY_CLASS[r.severity] || 'low'}">
          <span class="timeline-time">${escapeHtml(timeLabel)}</span>
          <span class="timeline-type">${escapeHtml(r.event_type || r.result_type)}</span>
          <span class="timeline-sev ${SEVERITY_CLASS[r.severity] || 'low'}">${SEVERITY_LABEL[r.severity] || 'Belirsiz'}</span>
          <span class="meta">${escapeHtml(detail)}</span>
          <span class="meta">Kanıt: ${escapeHtml(agreement)}</span>
        </li>`;
      }).join('')
      : `<li class="empty-state">${escapeHtml(emptyText)}</li>`;
  };
  render(el.modalFindings, findings, 'Bağlama uygun sürekli bulgu yok.');
  render(el.modalUncertain, uncertain, 'İnsan incelemesi gereken belirsiz gözlem yok.');
}

// ---------------------------------------------------------------------------
// Kanal tab'ları: Kanal B (VLM) ve RAG çıktıları
// ---------------------------------------------------------------------------

const VLM_SEVERITY_CLASS = { low: 'low', medium: 'medium', high: 'high', critical: 'high' };
const VLM_SEVERITY_LABEL = { low: 'Düşük', medium: 'Orta', high: 'Yüksek', critical: 'Kritik' };

function renderChannelB(vlm) {
  if (!vlm || typeof vlm !== 'object') {
    el.vlmMeta.textContent = '';
    el.vlmSummary.textContent = 'Kanal B yorumu alınamadı.';
    el.vlmRisks.innerHTML = '<li class="empty-state">VLM çıktısı yok.</li>';
    el.vlmEntities.innerHTML = '';
    el.vlmActions.innerHTML = '';
    return;
  }

  const metaParts = [];
  if (vlm.model_name) metaParts.push(`Model: ${vlm.model_name} (${vlm.model_backend || 'server'})`);
  const segCount = Number(vlm.segment_count);
  if (Number.isFinite(segCount) && segCount > 1) {
    const failed = (vlm.failed_segments || []).length;
    metaParts.push(`${segCount} segment iteratif analiz edildi${failed ? ` (${failed} başarısız)` : ''}`);
  } else {
    metaParts.push('Tek parça analiz');
  }
  if (Number.isFinite(Number(vlm.confidence_overall))) {
    metaParts.push(`Güven: ${formatConfidence(vlm.confidence_overall)}`);
  }
  const latency = Number(vlm.inference?.latency_ms);
  if (Number.isFinite(latency) && latency > 0) {
    metaParts.push(`Süre: ${(latency / 1000).toFixed(1)} sn`);
  }
  el.vlmMeta.textContent = metaParts.join(' · ');

  el.vlmSummary.textContent = vlm.scene_summary_tr || 'Sahne özeti üretilmedi.';

  const risks = Array.isArray(vlm.risk_events) && vlm.risk_events.length
    ? vlm.risk_events
    : (vlm.risk_flags_tr || []).map((flag) => ({ description_tr: flag, severity: 'medium' }));
  el.vlmRisks.innerHTML = risks.length
    ? risks.map((r) => {
      const sev = VLM_SEVERITY_CLASS[r.severity] || 'medium';
      const ts = Number.isFinite(Number(r.timestamp_sec)) ? mmss(r.timestamp_sec) : '—';
      return `<li class="timeline-item severity-${sev}">
        <span class="timeline-time">${escapeHtml(ts)}</span>
        <span class="timeline-sev ${sev}">${VLM_SEVERITY_LABEL[r.severity] || 'Orta'}</span>
        <span class="meta">${escapeHtml(r.description_tr || String(r))}</span>
        <span class="meta">Güven: ${formatConfidence(r.confidence)}</span>
      </li>`;
    }).join('')
    : '<li class="empty-state">VLM sahada risk işareti bulamadı.</li>';

  const entities = vlm.detected_entities || [];
  el.vlmEntities.innerHTML = entities.length
    ? entities.map((e) =>
        `<li class="chip">${escapeHtml(e.label || '?')} <span class="meta">${escapeHtml(e.confidence_hint || '')}${e.notes_tr ? ` — ${escapeHtml(e.notes_tr)}` : ''}</span></li>`
      ).join('')
    : '<li class="empty-state">Varlık listesi yok.</li>';

  const actions = vlm.detected_actions_tr || [];
  el.vlmActions.innerHTML = actions.length
    ? actions.map((a) => `<li class="chip">${escapeHtml(a)}</li>`).join('')
    : '<li class="empty-state">Eylem listesi yok.</li>';
}

function renderRag(rag) {
  const renderHyp = (target, rows, emptyText) => {
    target.innerHTML = rows.length
      ? rows.map((h) => {
        const sim = Number.isFinite(Number(h.similarity)) ? `Benzerlik: %${(h.similarity * 100).toFixed(0)}` : '';
        const risk = h.risk_level ? `Risk: ${h.risk_level}` : '';
        return `<li class="timeline-item">
          <span class="timeline-type">${escapeHtml(h.pattern || 'hipotez')}</span>
          <span class="meta">${escapeHtml(h.hazard_mechanism || '')}</span>
          <span class="meta">${[sim, risk].filter(Boolean).join(' · ')}</span>
        </li>`;
      }).join('')
      : `<li class="empty-state">${escapeHtml(emptyText)}</li>`;
  };
  renderHyp(el.ragVerified, rag.hypotheses || [], 'Doğrulanmış hipotez yok.');
  renderHyp(el.ragUnverified, rag.unverified_hypotheses || [], 'Doğrulanmamış hipotez yok.');
}

function initTabs() {
  el.demoTabs.querySelectorAll('.demo-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      el.demoTabs.querySelectorAll('.demo-tab').forEach((b) => b.classList.toggle('active', b === btn));
      document.querySelectorAll('#decision-summary .tab-pane').forEach((pane) => {
        pane.classList.toggle('active', pane.id === btn.dataset.tab);
      });
    });
  });
}

function fillAssignEventSelect(stamps) {
  el.assignEvent.innerHTML = stamps.length
    ? stamps.map((s, i) =>
        `<option value="${i}">${escapeHtml(s.time || s.timestamp)} · ${escapeHtml(s.event_type)} (${SEVERITY_LABEL[s.severity] || ''})</option>`
      ).join('')
    : '<option value="">Olay yok</option>';
}

function renderSuggestedActions(analysis) {
  const catalogByName = Object.fromEntries(toolCatalog.map((t) => [t.tool_name, t]));
  const tools = (analysis.triggered_mock_tools || []).filter((call) => catalogByName[call.tool_name]);

  el.suggestedActions.innerHTML = '';

  for (const call of tools) {
    const def = catalogByName[call.tool_name];
    const isEnabled = def.enabled !== false;
    const already = executedTools.has(call.tool_name);
    const btn = document.createElement('button');
    btn.className = 'btn btn-secondary';
    btn.title = JSON.stringify(call.params || {});
    if (already || !isEnabled) {
      btn.disabled = true;
      btn.dataset.locked = '1';
      if (already) btn.classList.add('btn-done');
      if (!isEnabled) btn.classList.add('btn-disabled');
      btn.textContent = already ? `✓ ${toolLabel(call.tool_name)}` : `🔒 ${toolLabel(call.tool_name)}`;
    } else {
      btn.textContent = toolLabel(call.tool_name);
      btn.addEventListener('click', () => executeTool(btn, call.tool_name));
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

async function callToolExecute(toolName, params = {}) {
  return api('/tools/execute', {
    method: 'POST',
    body: JSON.stringify({
      tool_name: toolName,
      params,
      camera_id: 'demo_upload',
      job_id: currentJobId || 'manuel',
      analysis_slug: currentJobId || '',
      triggered_by: 'supervisor',
    }),
  });
}

async function executeTool(btn, toolName) {
  try {
    await runOnce(btn, async () => {
      const res = await callToolExecute(toolName, { reason: `Önerilen aksiyon: ${toolName}` });
      executedTools.add(toolName);
      showToast(`${toolLabel(toolName)} çalıştırıldı`);
      return { label: `✓ ${toolLabel(toolName)}` };
    });
  } catch (err) {
    showToast(`Araç çalıştırılamadı: ${err.message}`, true);
  }
}

async function runManualAction() {
  const toolName = el.manualTool.value;
  if (!toolName) return;
  const original = el.manualRunBtn.textContent;
  el.manualRunBtn.disabled = true;
  el.manualRunBtn.textContent = `${original} …`;
  try {
    const params = {
      location: 'demo_upload',
      urgency: decisionPayload?.risk || 'Orta',
      reason: el.manualNote.value.trim() || (decisionPayload?.summary || 'Demo manuel aksiyon'),
    };
    const res = await callToolExecute(toolName, params);
    el.manualResult.textContent = res.already_executed
      ? `${toolLabel(toolName)} zaten çalıştırılmıştı.`
      : `${toolLabel(toolName)}: ${res.mock_result || res.status}`;
    if (res.assignment) {
      showToast(`Görev oluşturuldu: ${res.assignment.role}`);
      refreshAssignments();
    }
    el.manualRunBtn.textContent = 'Çalıştırıldı ✓';
    setTimeout(() => { el.manualRunBtn.textContent = original; }, 1200);
  } catch (err) {
    showToast(`Araç çalıştırılamadı: ${err.message}`, true);
    el.manualRunBtn.textContent = original;
  } finally {
    el.manualRunBtn.disabled = false;
  }
}

async function assignRole() {
  const role = el.assignRole.value;
  if (!role || !currentJobId) return;
  const eventIdx = el.assignEvent.value;
  const original = el.assignBtn.textContent;
  el.assignBtn.disabled = true;
  el.assignBtn.textContent = `${original} …`;
  try {
    const payload = {
      analysis_slug: currentJobId,
      camera_id: 'demo_upload',
      role,
      event_index: eventIdx === '' ? null : parseInt(eventIdx, 10),
      note: 'Canlı demo ekip ataması',
      assigned_by: 'supervisor',
    };
    const res = await api('/assignments', { method: 'POST', body: JSON.stringify(payload) });
    showToast(`Görev ${res.role} ekibine atandı`);
    await refreshAssignments();
  } catch (err) {
    showToast(`Atama yapılamadı: ${err.message}`, true);
  } finally {
    el.assignBtn.disabled = false;
    el.assignBtn.textContent = original;
  }
}

async function refreshAssignments() {
  if (!currentJobId) return;
  try {
    const rows = await api(`/assignments?analysis_slug=${encodeURIComponent(currentJobId)}`);
    if (rows.length) {
      const latest = rows[rows.length - 1];
      el.assignExisting.textContent = `✓ ${latest.role.toUpperCase()} ekibine atandı — ${latest.status}`;
    } else {
      el.assignExisting.textContent = '';
    }
  } catch (err) {
    console.error('Atamalar yenilenemedi:', err);
  }
}

function renderFeedbackBadge(feedback) {
  if (!feedback) {
    el.fbBadge.className = 'fb-badge pending';
    el.fbBadge.textContent = 'Değerlendirilmedi';
    return;
  }
  if (feedback.feedback_type === 'correct') {
    el.fbBadge.className = 'fb-badge correct';
    el.fbBadge.textContent = '✔ Karar Doğrulandı';
    return;
  }
  const typeMap = {
    false_positive: 'Yanlış Alarm',
    wrong_risk: 'Hatalı Risk',
    wrong_event: 'Hatalı Olay',
    wrong_action: 'Hatalı Aksiyon',
    other: 'Düzeltildi',
  };
  el.fbBadge.className = 'fb-badge corrected';
  el.fbBadge.textContent = `⚠️ ${typeMap[feedback.feedback_type] || 'Düzeltildi'}`;
}

async function refreshFeedbackBadge() {
  if (!currentJobId) return renderFeedbackBadge(null);
  try {
    const rows = await api(`/feedback?analysis_slug=${encodeURIComponent(currentJobId)}&limit=1`);
    renderFeedbackBadge(rows.length ? rows[0] : null);
  } catch {
    renderFeedbackBadge(null);
  }
}

async function submitFeedback(feedbackType) {
  if (!currentJobId || !decisionPayload) {
    showToast('Henüz tamamlanmış analiz yok', true);
    return;
  }
  const isCorrection = el.fbType.value !== 'correct';
  if (isCorrection) feedbackType = el.fbType.value;

  runOnce(el.fbBtnSubmitCorrection, async () => {
    const payload = {
      analysis_slug: currentJobId,
      camera_id: 'demo_upload',
      feedback_type: feedbackType,
      original_risk: decisionPayload.risk || '',
      original_summary: decisionPayload.summary || '',
      original_output: decisionPayload,
      corrected_risk: isCorrection ? el.fbCorrectRisk.value : (decisionPayload.risk || ''),
      corrected_summary: isCorrection ? el.fbCorrectSummary.value.trim() : (decisionPayload.summary || ''),
      corrected_actions: [],
      supervisor_notes: isCorrection ? el.fbNotes.value.trim() : 'Süpervizör tarafından doğrulandı.',
      prompt_context: { camera_label: 'demo_upload', source: 'demo_page' },
    };
    try {
      const res = await api('/feedback', { method: 'POST', body: JSON.stringify(payload) });
      renderFeedbackBadge(res);
      el.fbEditPanel.hidden = true;
      showToast(isCorrection ? '🎯 Düzeltme DPO havuzuna kaydedildi' : '✔ Karar doğrulandı');
    } catch (err) {
      showToast(`Geri bildirim kaydedilemedi: ${err.message}`, true);
    }
  });
}

// ---------------------------------------------------------------------------
// Canvas overlay
// ---------------------------------------------------------------------------

function resizeCanvas() {
  if (!el.video.videoWidth) return;
  const { clientWidth, clientHeight } = el.video;
  if (el.canvas.width !== clientWidth || el.canvas.height !== clientHeight) {
    el.canvas.width = clientWidth;
    el.canvas.height = clientHeight;
  }
}

function findNearestSnapshot(timeSec) {
  const events = currentEvents
    .filter((e) => e.stream === 'event.detected')
    .map((e) => e.data)
    .filter((ev) => ev.snapshot?.detections?.length);

  let best = null;
  let bestDelta = Infinity;
  events.forEach((ev) => {
    const evSec = parseMmss(ev.timestamp);
    const delta = Math.abs(evSec - timeSec);
    if (delta < bestDelta) {
      bestDelta = delta;
      best = ev.snapshot;
    }
  });
  return bestDelta <= 1.5 ? best : null;
}

function drawOverlay() {
  resizeCanvas();
  const ctx = el.canvas.getContext('2d');
  ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);

  if (!el.video.videoWidth || !el.video.videoHeight) return;

  const snapshot = findNearestSnapshot(el.video.currentTime);
  if (!snapshot) return;

  const scaleX = el.canvas.width / el.video.videoWidth;
  const scaleY = el.canvas.height / el.video.videoHeight;

  // Sahne grafiği çizgileri
  const centers = {};
  (snapshot.detections || []).forEach((det) => {
    const [x1, y1, x2, y2] = det.bbox || [0, 0, 0, 0];
    centers[`${det.class}_${det.track_id}`] = {
      x: ((x1 + x2) / 2) * scaleX,
      y: ((y1 + y2) / 2) * scaleY,
    };
  });

  (snapshot.scene_graph?.edges || []).forEach((edge) => {
    const src = centers[edge.source];
    const tgt = centers[edge.target];
    if (!src || !tgt) return;
    ctx.beginPath();
    ctx.moveTo(src.x, src.y);
    ctx.lineTo(tgt.x, tgt.y);
    ctx.strokeStyle = RELATION_COLORS[edge.relation] || RELATION_COLORS.near;
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  // Bounding box'lar
  (snapshot.detections || []).forEach((det) => {
    const [x1, y1, x2, y2] = det.bbox || [0, 0, 0, 0];
    const sx1 = x1 * scaleX;
    const sy1 = y1 * scaleY;
    const sx2 = x2 * scaleX;
    const sy2 = y2 * scaleY;
    const color = CLASS_COLORS[det.class] || CLASS_COLORS.unknown;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

    const label = `${(det.class || 'unknown').toUpperCase()} ${formatConfidence(det.confidence)}`;
    ctx.font = '12px system-ui, sans-serif';
    const tm = ctx.measureText(label);
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(sx1, Math.max(0, sy1 - 16), tm.width + 8, 16);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, sx1 + 4, Math.max(12, sy1 - 4));
  });
}



// ---------------------------------------------------------------------------
// Başlatma
// ---------------------------------------------------------------------------

async function init() {
  initUpload();
  initTabs();

  try {
    toolCatalog = (await api('/tools')) || [];
    el.manualTool.innerHTML = [
      '<option value="">Araç seçin</option>',
      ...toolCatalog.map((t) => {
        const disabled = t.enabled === false ? ' disabled' : '';
        return `<option value="${escapeHtml(t.tool_name)}"${disabled}>${escapeHtml(toolLabel(t.tool_name))}</option>`;
      }),
    ].join('');
  } catch (err) {
    console.error('Katalog yüklenemedi:', err);
    showToast('Araç kataloğu yüklenemedi', true);
  }

  try {
    const roles = (await api('/roles')) || [];
    el.assignRole.innerHTML = [
      '<option value="">Ekip seçin</option>',
      ...roles.map((r) => `<option value="${escapeHtml(r.role)}">${escapeHtml(r.label)}</option>`),
    ].join('');
  } catch (err) {
    console.error('Roller yüklenemedi:', err);
  }

  el.manualRunBtn.addEventListener('click', runManualAction);
  el.assignBtn.addEventListener('click', assignRole);

  el.fbBtnCorrect.addEventListener('click', () => submitFeedback('correct'));
  el.fbBtnToggleEdit.addEventListener('click', () => { el.fbEditPanel.hidden = !el.fbEditPanel.hidden; });
  el.fbBtnCancelEdit.addEventListener('click', () => { el.fbEditPanel.hidden = true; });
  el.fbBtnSubmitCorrection.addEventListener('click', () => submitFeedback());

  el.video.addEventListener('timeupdate', drawOverlay);
  el.video.addEventListener('resize', drawOverlay);
  window.addEventListener('resize', drawOverlay);
}

init();
