/**
 * Ortak yardımcı katman — süpervizör, saha ve admin sayfalarının üçü de
 * bunu import eder. Tek kaynak: `api()`, `escapeHtml()`, `mmss()`, toast
 * mekanizması ve risk sabitleri üç dosyada ayrı ayrı taşınmaz.
 */

export const API = '/api/v1';

/**
 * Analiz/atama seviyesindeki RİSK vokabüleri. Backend tam olarak üç değer
 * üretir (bkz. contracts/messages.py: Literal["Düşük","Orta","Yüksek"]).
 * "Kritik" burada KASITLI OLARAK yoktur; backend'de risk seviyesi değil.
 */
export const RISK_CLASS = { 'Yüksek': 'high', 'Orta': 'medium', 'Düşük': 'low' };

/**
 * Olay ŞİDDET vokabüleri (event_timestamps[].severity) — RİSK ile farklı bir
 * kavramdır. Dört değer alır: low/medium/high/critical. "critical", görsel
 * olarak "high" ile aynı sınıfa eşlenir (backend'in SEVERITY_TO_RISK'iyle
 * tutarlı: critical -> Yüksek). Türkçe etiketler yalnızca gösterim içindir.
 */
export const SEVERITY_LABEL = { critical: 'KRİTİK', high: 'YÜKSEK', medium: 'ORTA', low: 'DÜŞÜK' };
export const SEVERITY_CLASS = { critical: 'high', high: 'high', medium: 'medium', low: 'low' };

export function riskClass(risk) {
  return RISK_CLASS[risk] || 'low';
}

export function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

export function mmss(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

let _toastEl = null;
let _toastTimer = null;

export function initToast(toastEl) {
  _toastEl = toastEl;
}

export function showToast(message, isError = false) {
  if (!_toastEl) return;
  _toastEl.textContent = message;
  _toastEl.classList.toggle('toast-error', isError);
  _toastEl.classList.add('show');
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => _toastEl.classList.remove('show'), 3200);
}

/** `${API}/...` uç noktasına JSON istek atar; hata durumunda sunucunun
 * `detail` alanını Error mesajı olarak fırlatır. */
export async function api(path, options = {}) {
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

/**
 * Bir aksiyon butonunu "tek kullanımlık" hale getirir: tıklanınca disable
 * eder, sunucu isteği başarıyla dönerse KALICI olarak kilitlenmiş ("yapıldı")
 * durumuna geçer ve bir daha tıklanamaz. Hata olursa eski haline döner.
 *
 * @param {HTMLButtonElement} btn
 * @param {() => Promise<{label?: string}>} action
 */
export async function runOnce(btn, action) {
  if (btn.disabled || btn.dataset.locked === '1') return;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = `${original} …`;
  try {
    const result = await action();
    btn.dataset.locked = '1';
    btn.classList.add('btn-done');
    btn.textContent = (result && result.label) || `✓ ${original}`;
  } catch (err) {
    btn.disabled = false;
    btn.textContent = original;
    throw err;
  }
}
