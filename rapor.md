# Kalite Sorunları Raporu — Arson032_x264A Koşusu

Tarih: 2026-08-09 | Test: `test_akis.py --category Arson` | Backend: LLaVA-1.6-Mistral-7B Q8_0 (llama.cpp Vulkan sunucusu, n_ctx 32768)

**Sonuç özeti:** Pipeline uçtan uca çökmesiz çalışıyor; ancak nihai çıktı "Bilmiyorum / Düşük" fallback'ine düştü. Sorunlar altyapıda değil, **model davranışı + algı-uyumsuzluğu + parse kırılganlığı** üçgeninde.

---

## 1. Karar ajanı recursive tekrar döngüsüne giriyor (KRİTİK)

**Kanıt:** Aşama 8 ham çıktısı — model önce makul bir JSON başlatıyor ama `reasoning` alanına prompt'taki talimat cümlesini iç-içe kopyalayarak sonsuz döngüye giriyor; 4096 token dolana kadar aynı bloğu tekrarlıyor. Guardrail denemeleri:

- Deneme 1: `Invalid control character at line 9` → model JSON string'i içine **kaçışsız ham newline** bastı, `json.loads` patladı.
- Deneme 2-3 (temp 0.10, 0.05): aynı prompt aynen tekrar gönderildiği için ("full prompt already cached") model aynı döngüyü üretti → `JSON bloğu bulunamadı`.

**Kök nedenler:**

- **Model zayıf + prompt uzun:** 14.4k karakterlik prompt; 7B Q8 model uzun Türkçe yapılandırılmış talimatı takip edemiyor, bunun yerine prompt'un sonundaki şema metnini papağan gibi tekrarlıyor.
- **Şema placeholder'ları aynen kopyalanıyor:** `"summary": "Videonun genel özeti"` gibi yer tutucu metinler model tarafından literal değer olarak basılıyor. Placeholder'a değil **örnek değer** koymak bu davranışı kırar.
- **Penaltı yok:** `OpenAIServerBackend` isteği sadece `temperature` + `max_tokens` gönderiyor; sunucu tarafında `repeat_penalty` handler varsayılanında (1.1) kalıyor ve düşük sıcaklıkta (0.15→0.05) tekrar döngüsünü kıramıyor.
- **Retry kör:** Guardrail aynı prompt'u değiştirmeden tekrarlıyor; hata geri bildirimi ("önceki çıktın JSON değildi, tekrar etme") eklenmiyor. Aynı girdi + yakın sıcaklık = aynı hata.

**Öneriler (öncelik sırasıyla):**
1. Şema talimatındaki placeholder'ları gerçekçi örnek değerlerle değiştir (`decision_agent.py::_build_prompt`).
2. Server isteğine `repeat_penalty: ~1.15-1.2` ekle (`vlm_backend.py::OpenAIServerBackend`).
3. Guardrail `_extract_json`'da `json.loads(..., strict=False)` veya kontrol karakteri temizliği (newline kaçışsız string kurtarılabilir).
4. Retry'da prompt'a kısa bir düzeltme notu ekleme mekanizması (guardrail.py arkadaşının dosyası — birlikte karar).

## 2. Kanal B (S8) şema ekosu üretiyor (KRİTİK)

**Kanıt:** Ham çıktı, modelin şemayı *anlattığı* bir metin; "ayrıştırma başarılı" görünen JSON aslında placeholder'ların birebir kopyası: `"scene_summary_tr": "1-3 cümlelik Türkçe sahne özeti"`, `"label": "kısa etiket"`. 512 token'da kesilmiş (`tokens_generated: 512`, cümle yarım: "Bu örnekte, `").

**Kök neden:** 1. maddedeki aynı hastalık (placeholder echo) + Kanal_B'nin kendi `max_tokens=512` sınırı. Ayrıca Kanal_B'nin parser'ı placeholder dolu JSON'u "geçerli" sayıyor — içerik doğrulaması yok.

**Sonuç:** Videodaki **gerçek yangın hiçbir kanalda görülmedi** (aşağıda 4. madde). S8'in tek işi buydu ve placeholder üretti.

**Öneri:** Kanal_B prompt'undaki şema placeholder'ları örnek değere çevrilmeli; max_tokens 512→1024; parser "placeholder mı?" kontrolü ekleyebilir (Kanal_B arkadaşının paketi — birlikte karar).

## 3. Algı katmanı senaryo dışı videolarda gürültü üretiyor (YÜKSEK)

**Kanıt:** Bir kundaklama videosunda sınıf dağılımı: `forklift: 561`, `bird: 34`, `tv: 16`, `microwave: 11`, `bed: 18`, `refrigerator: 5`... Sonuç: 8 adet `forklift_tip_over` sinyali (araba/motosiklet kutuları "forklift"e eşleniyor), `dangerous_proximity (~0 piksel)` (kutular çakışık — muhtemel aynı-nesne çift tespiti), tekrarlı `ppe_missing` (her insan "baretsiz/yeleksiz personel" sayılıyor).

**Kök nedenler:**

- **COCO eşlemesi geçici ve kaba:** `truck/car/motorcycle → forklift` eşlemesi İSG sahası dışındaki her videoda yanlış pozitif üretir. Bu bilinen geçici durum; asıl çözüm İSG alanında fine-tune / sınıf seti.
- **Yangın/duman sınıfı yok:** COCO'da fire/smoke yok → kural motoru asla `fire_smoke` üretemez. Yangını görebilecek tek kanal VLM (S8) idi, o da 2. maddedeki sorunla boş döndü.
- **PPE kuralı bağlamsız:** Her tespit edilen insanı "saha personeli" varsayıyor; sokak/gözetleme videosunda doğal olarak spam.
- **~0 piksel yakınlık:** Aynı nesnenin iki kutusu (veya iç-içe kutu) mesafe=0 üretiyor; kuralda minimum kutu ayrımı/IoU filtresi yok.

**Öneri:** Benchmark için beklenti ayarı: `Normal` videolarda sinyal azlığı, `Arson/Explosion`'da ise yangın sinyalinin ancak S8 üzerinden gelmesi hedeflenmeli. `dangerous_proximity`'ye IoU/ayrım filtresi, PPE'ye süreklilik şartı (zaten streak var, sıkılaştırılabilir).

## 4. Nihai fallback anlamsız çıktı üretiyor (ORTA)

**Kanıt:** 3 deneme başarısız → `summary: "Bilmiyorum", risk: "Düşük"` + `record_incident(reason="Bilmiyorum")`. Halbuki elde 19 sinyal ve RAG "Yüksek/95" vardı; fallback bunların hiçbirini yansıtmıyor.

**Kök neden:** Fallback tasarımı "model sustu" senaryosu için; ama burada model konuştu, sadece JSON'u bozdu. Zengin kanıt varken sıfır çıktı vermek yanıltıcı.

**Öneri:** Fallback'te RAG risk seviyesini ve sinyal özetini pasifle de olsa çıktıya yansıt (örn. `risk: rag_risk_level`, `summary: "Otomatik analiz doğrulanamadı; N sinyal mevcut"`).

## 5. Kozmetik / yanıltıcı metadata (DÜŞÜK)

- S8 paketinde `"model_name": "Qwen/Qwen2.5-VL-7B-Instruct", "model_backend": "vllm"` yazıyor — gerçekte LLaVA GGUF + llama.cpp sunucusu. Kanal_B'nin metadata'sı sabit kodlu; raporlama/deney arşivi yanılır.
- Hafıza bağlamı son 10 sinyalle sınırlı; 19 sinyalin ilk 9'u karar prompt'unda var ama hafıza bölümünde yok (tasarım gereği, ama tutarsız görünüyor).

---

## Öncelik listesi

| # | Sorun | Etki | Zorluk |
|---|-------|------|--------|
| 1 | Karar ajanı tekrar döngüsü + placeholder echo | Nihai çıktı çöküyor | Düşük (prompt + penaltı) |
| 2 | S8 placeholder echo + 512 token sınırı | Bağımsız kanal boş | Düşük (Kanal_B prompt'u) |
| 4 | Fallback kanıtı yansıtmıyor | Yanıltıcı "Düşük" | Düşük |
| 3 | COCO eşlemesi + yangın sınıfı yok + PPE spam | Yanlış pozitif sinyaller | Orta-Yüksek (model/eğitim) |
| 5 | Metadata sabit kodlu | Rapor kirliliği | Çok düşük |

**Not:** 1-2-4 numaralı düzeltmeler prompt/config seviyesinde; 3 numara algı modeli seviyesinde (geçici eşleme kabulüyle yaşanabilir, benchmark yorumu buna göre yapılmalı).
