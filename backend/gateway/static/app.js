/**
 * Ortak yardımcılar — süpervizör, saha ve admin ekranlarının üçü de bunu kullanır.
 *
 * Neden ayrı bir dosya: üç sayfa da aynı `api()`, `escapeHtml()`, `mmss()`,
 * `showToast()` ve risk sabitlerini birebir kopyalıyordu. Bir yerde düzeltilen
 * hata diğer ikisinde kalıyordu (örn. risk sınıf haritası). Tek kaynak.
 */

export const API = '/api/v1';

/** Analiz/atama düzeyindeki risk seviyesi. "Kritik" burada YOKTUR — backend
 * yalnızca üç seviye üretir (bkz. contracts/messages.py: Literal["Düşük","Orta","Yüksek"]). */
export const RISK_CLASS = { 'Yüksek': 'high', 'Orta': 'medium', 'Düşük': 'low' };

/** Olay şiddeti (event_timestamps[].severity) dört değer alır; risk ile aynı
 * vokabüler DEĞİLDİR. "critical" risk karşılığı olarak "Yüksek" ile aynı
 * renkte gösterilir (backend SEVERITY_TO_RISK ile aynı eşleme). */
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
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

export function clockTime(iso) {
  const d = iso ? new Date(iso) : new Date();
  return Number.isNaN(d.getTime())
    ? new Date().toLocaleTimeString('tr-TR')
    : d.toLocaleTimeString('tr-TR');
}

let _toastEl = null;
let _toastTimer = null;

/** Toast elemanını kaydeder (her sayfa kendi `#toast`'ını geçirir). */
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
 * Bir butonu "tek kullanımlık" yapar: tıklanınca disable eder, işlem bitince
 * kalıcı olarak "yapıldı" durumuna kilitler (tekrar tıklanamaz hale getirir).
 *
 * Bu, eski davranıştaki asıl kusuru giderir: önceki kod butonu `finally`
 * içinde yeniden etkinleştiriyordu ("kozmetik" kilit), bu da çift tıklamayı
 * veya modalin yeniden render edilmesini (WS mesajı geldiğinde) aynı aksiyonun
 * tekrar tetiklenmesine açık bırakıyordu. Burada kilit `data-locked` niteliği
 * ile DOM'a yazılır; öğeyi yeniden oluşturan kod bu niteliği kontrol edip
 * kilitli görünümü koruyabilir (bkz. restoreLock).
 *
 * @param {HTMLButtonElement} btn
 * @param {() => Promise<{label: string}>} action - Sunucu isteğini yapar,
 *   başarıyla biten etiketi döner (örn. "✓ Çağrıldı: ...").
 */
export async function runOnce(btn, action) {
  if (btn.dataset.locked === '1' || btn.disabled) return;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = `${original} …`;
  try {
    const { label } = await action();
    btn.dataset.locked = '1';
    btn.classList.add('btn-done');
    btn.textContent = label || `✓ ${original}`;
  } catch (err) {
    btn.disabled = false;
    btn.textContent = original;
    throw err;
  }
}

/** `runOnce` ile kilitlenmiş bir butonun görünümünü, DOM yeniden
 * oluşturulduktan sonra (örn. liste tazelenince) eski hâline getirir. */
export function restoreLock(btn, label) {
  btn.dataset.locked = '1';
  btn.disabled = true;
  btn.classList.add('btn-done');
  btn.textContent = label;
}
