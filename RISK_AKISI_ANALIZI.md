# Risk ve Güven Akışı Analizi — Karar Ajanı Neden Her Şeye "Yüksek" Diyor

Tarih: 2026-08-27
Kapsam: Kanal A (YOLO + kural motoru) → RAG → Kanal B (VLM) → Karar Ajanı → Guardrail → Gateway → UI
Yöntem: salt-okunur kod analizi + `data/library/analyses/*.json` üzerinde ampirik ölçüm

---

## 0. Kısa cevap

Üç ayrı sorun var ve biri diğerini besliyor:

1. **RAG risk seviyesi fiilen sabit.** Ölçüm: 14 analizin **14'ünde** `rag_context.risk_level = "Yüksek"`, `risk_score = 100`, `matched_patterns` sayısı **20/20** (yani katalogdaki her pattern eşleşiyor). Bu bir model sorunu değil, `build_context()` içindeki eşik + agregasyon seçiminin matematiksel sonucu.
2. **Karar ajanının prompt'u yalnızca yukarı sapmaya izin veriyor.** RAG "Yüksek" der, prompt "daha yükseğe sapmaktan çekinme" der, aşağı sapma için simetrik izin yoktur. Sonuç: 14/14 `risk = "Yüksek"`.
3. **Miras gerçekten var ama tahmin edilen yerde değil.** Karar ajanı event'lere risk yazmıyor (şemada event-seviyesi risk alanı yok). Miras iki yerde oluyor: `build_event_timestamps()` genel riski her olayın `severity`'sine kopyalıyor, ve gateway analiz riskini her olay/atama kartına kopyalıyor.

Ek olarak: güven skoru UI'da **yüzde değil**, `0.90` biçiminde ham ondalık gösteriliyor (`index.js:281`) — ve bu skorun ayırt edici gücü yok (14 analizde 0.85–0.95, ortalama 0.899). "Bilmiyorum" / belirsizlik bayrağı üretiliyor ama UI'a hiç ulaşmıyor.

---

## 1. Ölçülen mevcut durum (14 kütüphane analizi)

| Metrik | Değer |
|---|---|
| Analiz sayısı | 14 |
| `risk = "Yüksek"` | **14 / 14 (%100)** |
| `rag_context.risk_level = "Yüksek"` | **14 / 14** |
| `rag_context.risk_score` | **her analizde 100** |
| `matched_patterns` sayısı | **her analizde 20/20** |
| Analiz `confidence` | ort **0.899**, min 0.85, maks 0.95 |
| Toplam event | 61 |
| Event `confidence` | ort 0.753, min 0.111, maks 1.0 |
| Event `severity` dağılımı | ezici çoğunluk `high`, kalan `critical`, yalnız 3 tane `medium`, **hiç `low` yok** |

Slug bazında (kısaltılmış):

```
18-cctv-footage-of-man...        risk=Yüksek ragRisk=Yüksek ragScore=100 mp=20 geoSig=90  ev=5 sev=[high×5]
always-wear-safety-harness...    risk=Yüksek ragRisk=Yüksek ragScore=100 mp=20 geoSig=24  ev=2 sev=[high,critical]
fire-in-bale-plucker...          risk=Yüksek ragRisk=Yüksek ragScore=100 mp=20 geoSig=0   ev=2 sev=[critical,high]
kaos-and-the-hands-of-fate...    risk=Yüksek ragRisk=Yüksek ragScore=100 mp=20 geoSig=6   ev=5 sev=[critical×5]
plastic-molding-incident...      risk=Yüksek ragRisk=Yüksek ragScore=100 mp=20 geoSig=79  ev=4 sev=[medium,medium,high,high]
...
```

Dikkat: `fire-in-bale-plucker` analizinde **geometrik sinyal sayısı 0**, sinyal kaynaklı pattern eşleşmesi 0, buna rağmen RAG riski "Yüksek"/100. Yani risk seviyesi kanıttan bağımsız üretiliyor.

Kanal A'nın (YOLO + kural motoru) tüm kütüphanede ürettiği sinyal tipleri:

```
ppe_missing : 326      gathering : 185
person_fall : 20       dangerous_proximity : 6
```

`ppe_missing` tek başına sinyallerin %58'i. Bu kural sabit `confidence=0.7` ile üretiliyor (`src/events/rules.py:492`) ve piksel/geometri doğrulaması yok — baret/yelek sınıfı düğümü yoksa "eksik" sayılıyor. Yani gürültünün büyük kısmı buradan geliyor.

---

## 2. Bilgi akışı haritası

### 2.1 Kanal A — YOLO → event sinyali

```
VideoReader → frames
  ObjectDetector.detect()            src/perception/detector.py:208-244
    model(frame, conf=0.35)          detector.py:221     (config.yaml:59)
    → Detection(class_name, confidence, bbox, frame_idx)   detector.py:236-243
  ObjectTracker.track()              src/perception/tracker.py:215-276  (bytetrack)
    → TrackedObject(track_id, class_name, history)
  ObserverAgent.observe_frame()      src/perception/observer_agent.py:110-180
    → observation = {frame_idx, timestamp, detections, tracks, scene_graph}
  EventEngine.process_observation()  src/events/event_engine.py:79-110
    _observation_to_tracks()         event_engine.py:155-210  (confidence eksikse 1.0!)
    TrackStateMachine.update()       src/events/state_machine.py:46-145 (yalnız geometri)
    RuleSet.evaluate()               src/events/rules.py:98-133
    _is_recent() 10 sn de-dup        event_engine.py:212-229
  → EventSignal{event_type, timestamp, description, confidence, involved_track_ids, metadata}
```

**Önemli:** `EventSignal`'da **risk/severity alanı yok** (`rules.py:28-60`). Event confidence'ı her kuralda geometriden yeniden hesaplanır, YOLO confidence'ından türetilmez:

| Kural | confidence formülü | Dosya:satır |
|---|---|---|
| tip_over | `min(1, aspect_ratio/3.0)` | rules.py:173 |
| fall | `min(1, drop_ratio/(min_drop_ratio*2))` | rules.py:230 |
| gathering | `min(1, len(cluster)/5.0)` | rules.py:305 |
| **ppe_missing** | **sabit 0.7** | rules.py:492 |
| fire_smoke | `det.confidence` (tek istisna) | rules.py:430 |
| proximity | `1 - dist/effective_limit` | rules.py:627 |

YOLO confidence'ı **hiçbir yerde risk skoruna dönüştürülmüyor**. Bu kısım doğru tasarlanmış.

### 2.2 RAG — burada risk seviyesi üretiliyor

```
observations (TÜM video, zaman penceresi YOK)
  → _observations_to_natural_language()   rag_layer.py:65-215
      "Sahnede tespit edilenler: 120 insan, 8 arac. Nesnelerin büyük çoğunluğu hareketsiz..."
  → TF-IDF sorgu vektörü                  rag_layer.py:34-62
  → her pattern için _cosine(...)         rag_layer.py:371-388
  → sinyal eşleşmesi varsa sim = max(sim*1.5, 0.1)   rag_layer.py:396
  → sim < 0.1 ise at                      rag_layer.py:397
  → tekrar-boost (yalnız YUKARI)          rag_layer.py:415-423
  → top = max(matches, key=risk_score)    rag_layer.py:428
  → risk_level = top["risk_level"]        rag_layer.py:430
```

`data/risk_patterns.yaml` dağılımı: **Yüksek 10, Orta 9, Düşük 1**. `fire_smoke` risk_score **100** ve `required_nodes: []`.

Gerçek ölçülen benzerlikler (`fire-in-bale-plucker`, 20/20 eşleşme):

```
pallet_collapse   0.729 (structural_match)   overhead_hazard   0.399
ppe_missing       0.383                      person_fall       0.319
blocked_exit      0.340                      fire_smoke        0.257
vehicle_speeding  0.137  ← en düşük, hâlâ 0.1 eşiğinin üstünde
```

**Sonuç:** hiçbir pattern 0.1 eşiğine takılmıyor → 20/20 eşleşiyor → `max(risk_score)` her zaman `fire_smoke = 100` → `risk_level` her zaman "Yüksek". Bu deterministik. Video içeriği ne olursa olsun RAG "Yüksek" der.

### 2.3 Kanal B — VLM

```
prepare_video_segments()          Kanal_B/preprocessing.py:688-757
  video ≤60sn ve ≤720p → tek parça, yeniden kodlama yok
  aksi hâlde detect_scene_boundaries() (SSIM 0.30, min 15sn, maks 60sn)
ServerBackend.infer_video()       Kanal_B/vlm_backend.py:784-881
  prompt: build_video_system_prompt()   vlm_backend.py:579-666
  → S8: VLMInterpretation           Kanal_B/contracts.py:140-207
```

VLM'in prompt'u riski **metinsel severity + sayısal confidence** olarak istiyor:

```
"risk_events": [{"description_tr": ..., "severity": "low|medium|high|critical",
                 "confidence": 0.85, "timestamp_sec": ...}]
"confidence_overall": 0.0
```

Default'lar risk üretmeye eğilimli değil ama güven üretmeye eğilimli: `severity` yoksa `"low"`, `confidence` yoksa **0.5** (`vlm_backend.py:838-874`). Parse hatasında sessizce `{risk_events: [], confidence_overall: 0.0}` fallback'i kullanılıyor ve `parse_succeeded` bayrağı S8'e yazılmıyor (`vlm_backend.py:824-833`) — tüketici parse hatasını ayırt edemiyor.

`computed_confidence` formülü (`vlm_backend.py:177-217`):
`model_conf*0.5 + risk_score*0.3 + evidence_bonus(≤0.2)`
Video modunda `supporting_frame_count` prompt'ta hiç istenmiyor → hep default 1 → `evidence_bonus` sabit 0.025 → üst sınır ~0.825. **Bu skor hesaplanıyor ama karar ajanına veya UI'a hiç taşınmıyor.** Kullanılmayan, kalibre edilmiş tek sinyal bu.

Segment birleştirmede `confidence_overall` ve `computed_confidence` segmentlerin **düz ortalaması** (`contracts.py:387-388`) — tek kritik segment 3 segmentli bir videoda seyreltiliyor.

### 2.4 Karar ajanı — LLM

`DecisionAgent.decide()` (`src/reasoning/decision_agent.py:150-176`) hiçbir skor hesaplamıyor; ham metni döndürüyor. Prompt'a serialize edilenler (`decision_agent.py:190-243`):

```
system_prompt (config.yaml:249-293)
--- ALGI KATMANI OLAY SİNYALLERİ ---    event_signals JSON
--- RAG KONTEXTİ ---                    rag_context JSON (tamamı, 20 pattern dahil)
--- VLM KARE YORUMU ---                 vlm_interpretation JSON
--- SAHNE GRAFİ (SON KARE) ---
--- KISA SÜRELİ HAFIZA ---
--- KULLANILABİLİR ARAÇLAR ---
+ çıktı şeması
```

Çıktı şeması (`src/output/schema.py`):

```python
class EventEntry:      time, event, event_type, confidence(0-1), timestamp_sec, duration, end_time
class AnalysisOutput:  summary, events[], risk("Düşük|Orta|Yüksek"), actions[],
                       reasoning, confidence(0-1), triggered_mock_tools[]
```

Event'te risk/severity **yok**, analizde tek bir `risk` var. Belirsizlik/abstain alanı **yok**.

### 2.5 Guardrail

`src/output/guardrail.py:20-43`: parse + 3 retry (`temperatures [0.15, 0.10, 0.05]`).
Semantik kontrol (`guardrail.py:102-120`) **asimetrik**:

```python
if risk == "Yüksek" and not events:            raise   # retry
if risk == "Yüksek" and len(actions) < 2:      raise   # retry
if risk_order[rag_risk] > risk_order[risk]:    logger.warning(...)   # sadece log
```

Yani "Yüksek" biçimsel olarak zorlanıyor, "gereğinden yüksek" için hiç kontrol yok; RAG'den düşük risk vermek ise uyarı üretiyor. Fail-safe'ler ters yönde çalışıyor (doğru): `_null_response()` risk="Düşük", confidence=0.0 (`guardrail.py:122-131`).

### 2.6 Kütüphane çıktısı — miras noktası #1

`scripts/analyze_video_library.py:196-246` (`build_event_timestamps`):

```python
risk = str(final_output.get("risk") or "Düşük")
default_severity = {"Yüksek": "high", "Orta": "medium", "Düşük": "low"}.get(risk, "medium")
...
severity = default_severity
if vlm_events:
    nearest = min(vlm_events, key=lambda v: abs(v["seconds"] - seconds))
    if abs(nearest["seconds"] - seconds) <= 5.0:
        severity = nearest["severity"]      # VLM'den ödünç
        detail = nearest["description"]
```

**Bu, aradığın miras.** Genel risk "Yüksek" olduğu için ±5 sn içinde VLM karşılığı bulunmayan her olay `severity="high"` alıyor. Ve ölçümde risk her zaman "Yüksek" olduğundan, `default_severity` her zaman `high`. Ölçüm bunu doğruluyor: 61 event'in hiçbirinde `low` yok.

Eşleşme olduğunda da ters yönde bir karışma var: eşleştirme ölçütü **yalnızca ±5 sn zaman yakınlığı**, olay tipi kontrolü yok. Bir `ppe_missing` olayı 3 saniye ötedeki `critical` bir yangın riskinin severity'sini devralabiliyor.

İkincil: `contracts.py:372-377` — `timestamp_sec` üretilmemiş VLM risk olayına **segment başlangıç saniyesi** atanıyor. Sonra bu yapay zaman üzerinden ±5 sn eşleştirmesi yapılıyor.

### 2.7 Gateway + UI — miras noktası #2

| Nokta | Ne oluyor | Dosya:satır |
|---|---|---|
| Kamera rozeti | analiz riski, sürekli görünür | `replay.py:288` → `index.js:125-127` |
| Kamera çerçevesi | olayın `severity`'sinden risk (tek doğru yer) | `replay.py:96-115` → `index.js:135-137` |
| `event.detected` payload | `severity` → risk eşlemesi | `replay.py:404` |
| `decision.final` | hücre riskini analiz riskiyle ezer | `replay.py:379` → `index.js:672` |
| Modal rozeti | analiz riski | `library.py:237` → `index.js:279-280` |
| `notification.push` | `severity` yoksa **default "high"** | `replay.py:420-421` |
| **Saha görev kartı** | **analiz riski atamaya kopyalanıyor** | `ops.py:155` ve `ops.py:345` → `saha.js:83` |
| Araç `urgency` parametresi | analiz riski | `saha.js:262`, `index.js` manuel aksiyon |

`store.py:55-79` — `assignments` tablosunda `risk TEXT` var, **`confidence` kolonu yok**. Saha ekibine güven bilgisi şema düzeyinde hiç ulaşmıyor.

### 2.8 Güven skoru gösterimi — gerçek durum

Senin hatırladığın gibi yüzde değil:

```js
// backend/gateway/static/index.js:281
el.modalConfidence.textContent = `Güven: ${(Number(analysis.confidence) || 0).toFixed(2)}`;
```

Ekranda `Güven: 0.95` yazıyor. UI'ın hiçbir yerinde `confidence*100` veya `%` dönüşümü yok. `%` yalnızca `admin.js:112`'de RLHF doğruluk oranı için var (o zaten yüzde). Yani "yüzde gösteriliyor" tespiti yanlış — ama asıl sorun ölçek değil, **skorun kendisi**:

- 14 analizde 0.85–0.95 arası. Ayırt edici gücü yok.
- Kaynağı LLM'in kendi beyanı. Kalibre değil, dışsal doğrulaması yok.
- Event bazlı `confidence` verisi backend'den geliyor (`library.py:249` → `event_timestamps[].confidence`) ama UI'da **hiç render edilmiyor** (`renderTimeline`, `index.js:332-357`).
- Saha ekranında güven hiç yok.
- Kalibre edilmiş `computed_confidence` hesaplanıyor ama kimse okumuyor.

### 2.9 Belirsizlik / "emin değilim"

Mekanizma var, UI'a ulaşmıyor:

```python
# backend/decision/main.py:119-123
result["null_response"] = (str(result.get("summary","")).strip()
                           == str(cfg.output.guardrail.null_response).strip())   # "Bilmiyorum"
```

Bu bayrak `decision/main.py:174-181`'de **yalnızca metrik ve log** için kullanılıyor. `DecisionFinal` şemasında (`backend/contracts/messages.py:32-41`) `null_response` alanı **yok** → Redis'e çıkmıyor. Kütüphane yolunda ise (`replay.py:372-390`, `library.py:227-252`) böyle bir alan hiç üretilmiyor.

Sonuç: guardrail "Bilmiyorum" döndürdüğünde UI bunu normal bir özet gibi gösterir, yanında hâlâ risk rozeti ve `Güven: 0.00` yazar. Operatör "model bilmiyor" ile "model karar verdi"yi ayırt edemiyor.

Prompt'ta ise belirsizlik yalnızca serbest metin talimatı olarak var (`config.yaml`, system_prompt): görsel iddiaları `reasoning` alanında "single source" olarak not et ve `confidence` değerlerini buna göre ayarla. Yapısal bir alanı yok, dolayısıyla zorlanamıyor ve UI'da ayrı gösterilemiyor.

---

## 3. Kök nedenler — öncelik sırasına göre

### K1 — RAG risk seviyesi sabit "Yüksek" (birincil neden)

`src/reasoning/rag_layer.py:341-438`. Üç tasarım seçimi birleşiyor:

- `threshold = 0.1` (satır 346) — ölçülen en düşük benzerlik 0.137. Eşik hiçbir şeyi elemiyor.
- `sim = max(sim*1.5, threshold)` (satır 396) — sinyal eşleşmesi varsa eşik **zorla** geçiliyor, elenme imkânı yok.
- `top = max(matches, key=risk_score)` (satır 428) — 20 eşleşme arasından en yüksek skorlu tek pattern tüm konteksti belirliyor. `fire_smoke` (100) her zaman listede.

Etki: `rag_context.risk_level` bir sinyal değil, sabit. Ampirik: 14/14 "Yüksek", 14/14 skor 100.

### K2 — `structural_match` üyelik testi, sayı testi değil

`rag_layer.py:330-338`:

```python
required = pattern_data.get("required_nodes", [])
if required and all(req in detected_classes for req in required):
    matched[name] = {"signal": {"event_type": "structural_match", ...}}
```

`gathering`'in `required_nodes: ["insan","insan","insan"]` — `all(... in ...)` üyelik kontrolü olduğu için **tek bir "insan" tespiti** bunu eşleştiriyor. Tek insan tespiti şu 7 pattern'i birden tetikliyor: `person_fall`(90), `unauthorized_access`(65), `restricted_time_access`(65), `running_in_facility`(60), `gathering`(50), `fatigue_detection`(50), `ppe_missing`(45). Bir forklift eklenince +`forklift_tip_over`(95), +`vehicle_speeding`(75), +`dangerous_proximity`(80), +`blind_spot_movement`(75).

Ayrıca `required_nodes: []` olan 4 pattern (`fire_smoke`, `leakage`, `spill_slip_hazard`, `overhead_hazard`) structural yolla **asla** eşleşemez — yalnızca benzerlikle gelir, ki eşik 0.1 olduğu için zaten hep gelir.

### K3 — Prompt'ta tek yönlü yukarı bias

`config.yaml` system_prompt, birebir:

> The risk level and actions in the RAG context are the catalog's ground truth: take them as the basis, but **if your own evidence assessment requires a higher risk, do not hesitate to deviate** and explain the reason for the deviation in reasoning.

Aşağı sapma için simetrik bir cümle yok. RAG tabanı "Yüksek" olduğu için model yalnızca "Yüksek"te kalabiliyor. Ek olarak önceliklendirme paragrafı:

> Situations that directly threaten human health (falling, tip-over, fire/smoke, dangerous proximity) **always take precedence** over equipment-level warnings; order your risk level, actions, and tool selection according to this priority.

Tek bir düşme/duman iddiası genel riski yukarı çekiyor. Prompt'ta few-shot örneği yok, dolayısıyla "örneklerin hepsi yüksek" gibi bir bias kaynağı yok.

### K4 — Guardrail asimetrik

`src/output/guardrail.py:107-120`. "Yüksek" için biçim zorlaması var (event yoksa retry, aksiyon <2 ise retry), "gereğinden yüksek" için hiç kontrol yok. RAG'den düşük risk vermek uyarı üretiyor. Modele giden örtük mesaj: düşük söylemek şüpheli, yüksek söylemek serbest.

### K5 — Genel riskin event severity'sine mirası

`scripts/analyze_video_library.py:210-227`. Yukarıda 2.6'da ayrıntılı. `default_severity` genel riskten türetiliyor ve `risk` her zaman "Yüksek" olduğundan her olay `high` başlıyor.

### K6 — Gateway'de riskin olay/atama kartlarına kopyalanması

`backend/gateway/routers/ops.py:155` ve `:345`:

```python
risk=str(analysis.get("risk") or ""),
```

Saha ekibi tek bir olay için görev kartı görse bile üzerindeki risk rozeti **tüm analizin** genel riski. Olayın kendi severity'si atamaya hiç yazılmıyor.

### K7 — `replay.py:420` default severity "high"

```python
"severity": stamp.get("severity", "high"),
```

Damgada severity yoksa "high" varsayılıyor. Genel riskten bağımsız, ayrı bir yukarı sapma kaynağı.

### K8 — RAG'de zamansal boost yalnız yukarı

`rag_layer.py:415-423`: aynı pattern 5 dk penceresinde 3. kez görülürse Orta→Yüksek (+20), Düşük→Orta (+30). Aşağı indirme yok. Tekrar eden zararsız durumlar (örn. sürekli `ppe_missing`) zamanla riski şişiriyor.

### K9 — RAG video seviyesinde, zaman penceresi yok

`build_context()`'e tüm videonun gözlemleri tek seferde giriyor (`analyze_video_library.py:344`). Sorgu "120 insan, 8 arac tespit edildi" gibi tüm videoyu özetleyen tek bir metin. Zamansal lokalite tamamen kayboluyor; 5 dakikalık videonun 3. saniyesindeki bir tespit ile 290. saniyesindeki tespit aynı torbada.

### K10 — Aksiyon/araç geri beslemesi riski dramatikleştiriyor

`src/reasoning/mock_tools.py:47-53` — risk "Yüksek" olduğunda 7 araç adayı (`call_health_team`, `secure_area`, `lockdown_facility`, `trigger_fire_suppression`, `activate_cbrn_protocol`...). `analyze_video_library.py:455-470` `triggered_mock_tools` boşsa bunu otomatik dolduruyor. Çıktının "her şey acil" hissi buradan büyüyor.

### K11 — Sayısal güven skoru kalibre değil ve yanlış kaynaktan

LLM öz-beyanı (`AnalysisOutput.confidence`) 0.85–0.95 aralığında sıkışmış. Hesaplanan kalibre skor (`computed_confidence`) kullanılmıyor. Event confidence'ı UI'da gösterilmiyor. Eksikse guardrail 0.5 atıyor (`guardrail.py:77-78`), VLM parse hatasında 0.5 (`decision_agent.py:131-133`) — sessiz varsayılanlar gerçek bilgiyle karışıyor.

### K12 — Belirsizlik bayrağı UI'a taşınmıyor

2.9'da ayrıntılı. `null_response` üretiliyor, `DecisionFinal` şemasında yok, `public_view` whitelist'inde yok, UI'da kontrolü yok.

### K13 — VLM flag'leri ana yolda RAG'e girmiyor

`analyze_video_library.py:344` ve `src/main.py:211`: `rag.build_context(observations, event_signals)` — `vlm_flags` **geçilmiyor**. `vlm_flags` yalnızca mikroservis yolunda (`backend/decision/main.py:73-81`) kullanılıyor, orada da `observation_report=[]` verildiği için `structural_match` devre dışı. Yani iki yol birbirinden farklı davranıyor; kütüphane analizinde VLM'in gördüğü yangın/sızıntı RAG sorgusuna hiç girmiyor.

### K14 — Grid ve video prompt'ları arasında dil tutarsızlığı

`vlm_backend.py:23-83` (grid) `scene_summary_tr` alanını **İngilizce** ister; `vlm_backend.py:579-666` (video) Türkçe ister. Ayrıca `src/reasoning/decision_agent.py:33-58`'deki üçüncü prompt hâlâ eski `risk_flags_tr` + `notable_frames` şemasını istiyor. Üç prompt, üç farklı sözleşme.

---

## 4. Ne yapılmalı

### P0 — Riskin sabitliğini kır (bu olmadan diğerleri fark etmez)

**P0.1 · RAG agregasyonunu değiştir.** `rag_layer.py:341-438`

- `threshold`'u ölçülen dağılıma göre kalibre et. Mevcut benzerlikler 0.13–0.73 arasında; 0.1 anlamsız. Önerilen: eşiği sabit vermek yerine **top-k + göreli eşik** kullan (örn. en yüksek benzerliğin %60'ının altındakileri at, en fazla 5 pattern tut).
- `max(risk_score)` yerine **kanıt ağırlıklı agregasyon**: `score = risk_score × similarity × evidence_weight`. `evidence_weight` = sinyal eşleşmesi 1.0, structural_match 0.3, yalnız benzerlik 0.1. Böylece 0.257 benzerlikli `fire_smoke` tek başına konteksti "Yüksek" yapamaz.
- Kanıtsız pattern'i risk seviyesine hiç dahil etme: `matched_patterns` listesinde bilgi olarak kalsın, `risk_level` hesabına yalnızca gerçek sinyali olanlar girsin.

**P0.2 · `structural_match`'i sayı-duyarlı yap.** `rag_layer.py:330-338`

`all(req in detected_classes)` yerine `collections.Counter` ile çokluk kontrolü: `gathering` için gerçekten ≥3 "insan" istensin. Ayrıca structural_match'i "risk" değil "olasılık/bağlam" olarak işaretle — risk seviyesini tek başına belirlemesin.

**P0.3 · Prompt'u simetrik hale getir.** `config.yaml` decision_agent.system_prompt

- "do not hesitate to deviate [higher]" cümlesini iki yönlü yaz: kanıt yetersizse **aşağı** sapmak da zorunlu olsun ve gerekçesi `reasoning`'e yazılsın.
- RAG kontekstini "ground truth" olarak sunmayı bırak. Gerçekte 20/20 eşleşen bir aday listesi; prompt'ta "aday hipotezler, çoğu doğrulanmamış" olarak tanımlanmalı.
- RAG kontekstini prompt'a **kırpılmış** ver: 20 pattern'in tamamı yerine yalnızca kanıtlı olanlar. Şu an model 10 tanesi "Yüksek" olan bir liste görüyor; bu tek başına anchor etkisi yaratıyor.
- Risk seviyesi için açık **karar kriteri** yaz: hangi kanıt kombinasyonu Düşük/Orta/Yüksek demek. Şu an kriter yok, yalnızca öncelik sıralaması var.

**P0.4 · Guardrail'i simetrik yap.** `guardrail.py:102-120`

- "Yüksek" için kanıt zorunluluğu ekle: en az bir event'in `confidence ≥ eşik` olması **ve** o event tipinin gerçekten sinyal veya VLM tarafından desteklenmesi. Aksi hâlde retry.
- RAG'den düşük risk vermeyi uyarı olmaktan çıkar; bu meşru bir sonuç.
- Ters kontrol ekle: kanıt zayıfken "Yüksek" verilmişse retry veya "belirsiz" işaretle.

### P1 — Mirası kes

**P1.1 · Event'e kendi risk/severity'sini ver.** `src/output/schema.py`

`EventEntry`'ye `severity: Literal["low","medium","high","critical"]` ekle ve karar ajanının prompt şemasında **her olay için ayrı** istet. Bu, mirasın kaynağını kurutur: genel riski türetmeye gerek kalmaz.

**P1.2 · `build_event_timestamps` fallback'ini kaldır.** `analyze_video_library.py:210-227`

- `default_severity`'yi genel riskten türetmeyi bırak. Karar ajanı (P1.1 sonrası) event severity'si üretiyorsa onu kullan; üretmediyse `severity=None`/`"unknown"` bırak — uydurma.
- VLM eşleştirmesine **olay tipi kontrolü** ekle. ±5 sn zaman yakınlığı tek başına yeterli değil; `ppe_missing` bir yangın riskinin `critical`'ını devralmamalı.
- `contracts.py:372-377`'deki "timestamp yoksa segment başı" ataması işaretlensin (`timestamp_estimated: true`) ki ±5 sn eşleştirmesi yapay zamanlar üzerinden sessizce çalışmasın.

**P1.3 · Gateway'de atamaya olay severity'sini yaz.** `ops.py:155`, `ops.py:345`, `store.py:55-79`

- `assignments` tablosuna `event_severity TEXT` ekle, atamada `event.get("severity")` yaz.
- Saha kartındaki rozet (`saha.js:83`) olayın severity'sini göstersin; analizin genel riski ayrı ve ikincil bir bilgi olarak (örn. "bağlam: yüksek riskli video") görünsün.
- Araç `urgency` parametresini analiz riskinden değil olay severity'sinden türet.

**P1.4 · `replay.py:420` default'unu düzelt.** `"high"` → `"unknown"` veya damga severity'si yoksa bildirim hiç üretilmesin.

### P2 — Güven ve belirsizlik ifadesini yeniden tasarla

Senin sorgulaman haklı: **LLM'in kendi beyan ettiği 0-1 skoru gösterilecek bir şey değil.** Ölçüm bunu doğruluyor (14 analizde 0.85–0.95, ayırt edici gücü yok). İki ayrı kavramı karıştırmamak lazım:

| Kavram | Kaynak | Nerede kullanılmalı |
|---|---|---|
| Kanıt gücü | kaç bağımsız kanal doğruladı, süre, track sürekliliği | **UI'da metinsel** |
| Model öz-beyanı | LLM'in yazdığı `confidence` | yalnız iç loglama/metrik |
| Kalibre skor | `computed_confidence` (`vlm_backend.py:177-217`) | iç eşikleme |

**P2.1 · Metinsel güven bandı üret — türetilmiş, beyan edilmiş değil.**

Şemaya `evidence_level: Literal["doğrulanmış","tek_kaynak","zayıf","belirsiz"]` ekle ve bunu **kodda** kanıt sayımından türet, LLM'e sorma:

- `doğrulanmış`: hem geometrik sinyal hem VLM risk olayı aynı olayı ±2 sn içinde ve aynı tipte bildiriyor
- `tek_kaynak`: yalnız biri bildiriyor
- `zayıf`: yalnız RAG benzerliği var, sinyal yok
- `belirsiz`: parse hatası, çelişkili kanıt veya `computed_confidence` düşük

UI'da `Güven: 0.95` yerine bu bandın Türkçe etiketi görünsün: "İki kanal doğruladı", "Tek kaynak — doğrulanmadı", "Emin değil".

**P2.2 · "Emin değilim" çıktısını yapısal hale getir.**

- `AnalysisOutput`'a `uncertain: bool` + `uncertainty_reason: str` ekle.
- `DecisionFinal` şemasına (`backend/contracts/messages.py:32-41`) `null_response` **ve** `uncertain` alanlarını ekle — şu an bayrak üretilip yolda kayboluyor.
- `library.public_view()` whitelist'ine (`library.py:227-252`) ekle.
- UI'da ayrı bir görsel durum yap (`index.js:279-282`): belirsizken risk rozeti yerine "Belirsiz — insan incelemesi gerekli" göster. Şu an `summary="Bilmiyorum"` yanında `Yüksek` rozeti ve `Güven: 0.00` görünüyor; bu yanıltıcı.
- Guardrail `_null_response()`'a `uncertain: True` ekle (`guardrail.py:122-131`).

**P2.3 · Event bazlı güveni UI'da göster veya tamamen kaldır.**

`event_timestamps[].confidence` verisi UI'a gidiyor ama `renderTimeline` (`index.js:332-357`) göstermiyor. Ya metinsel bandıyla göster, ya `public_view`'dan çıkar. Şu anki hâli boşa taşınan veri.

**P2.4 · `computed_confidence`'ı akışa sok.** `vlm_backend.py:177-217`'de hesaplanıyor, hiçbir yerde okunmuyor. En azından `evidence_level` türetiminde kullanılsın. Video modunda `supporting_frame_count` prompt'ta istenmediği için `evidence_bonus` sabit — ya prompt'ta istet ya formülden çıkar.

### P3 — Gürültü kaynaklarını temizle

**P3.1 · `ppe_missing` kuralını sıkılaştır.** `rules.py:441-497`. Sinyallerin %58'i buradan (326/580) ve confidence sabit 0.7. Baret/yelek sınıfı düğümü yoksa "eksik" sayılıyor; `best.pt` bu sınıfları güvenilir tespit ediyor mu ayrıca doğrulanmalı. Minimum süre eşiği ve insan bbox kalitesi kontrolü ekle.

**P3.2 · RAG'i zaman pencereli çalıştır.** `analyze_video_library.py:344`. Tüm videoyu tek sorguda değil, 10-15 saniyelik pencerelerde sorgula; risk seviyesi pencere bazında üretilsin. Bu hem K9'u çözer hem event-seviyesi riski doğal olarak mümkün kılar.

**P3.3 · Zamansal boost'u iki yönlü yap.** `rag_layer.py:415-423`. Tekrarlayan ama zararsız durum (sürekli `ppe_missing`) riski yükseltmemeli; tersine, sürekli var olan bir durum "kronik/düşük aciliyet" olarak sınıflanabilir.

**P3.4 · `vlm_flags`'i kütüphane yolunda da geçir.** `analyze_video_library.py:344` ve `src/main.py:211`. İki yolun davranışını birleştir.

**P3.5 · Araç önerisini riskten değil olay tipinden türet.** `mock_tools.py:47-53`. "Yüksek" → 7 araç eşlemesi, `activate_cbrn_protocol` dahil, olay tipinden bağımsız. Pattern'lerin `mock_tool_hints` alanı zaten var; `recommend_tools()` (`rag_layer.py:~440`) kullanılsın.

**P3.6 · Üç VLM prompt'unu tek sözleşmede birleştir.** `vlm_backend.py:23-83`, `vlm_backend.py:579-666`, `decision_agent.py:33-58`. Dil tutarsızlığı (`*_tr` alanların İngilizce doldurulması) ve şema farkı (`risk_flags_tr` vs `risk_events`) giderilsin.

**P3.7 · Parse başarısını S8'e yaz.** `vlm_backend.py:824-833`. `parse_succeeded: bool` alanı ekle; şu an parse hatası "risk yok, güven 0" olarak sessizce akıyor ve `computed_confidence == 0.0` dolaylı ipucundan başka işaret yok.

---

## 5. Doğrulama planı

Değişiklikleri ölçmeden "düzeldi" demek mümkün değil. Öneri:

1. **Referans etiket seti.** 14 videoyu elle etiketle (gerçek risk seviyesi + gerçek olay listesi + zaman). Şu an hiç ground truth yok, dolayısıyla "%100 Yüksek" iyi mi kötü mü matematiksel olarak kanıtlanamıyor — bu videolar İSG kaza derlemesi olduğu için gerçekten çoğu yüksek riskli olabilir. **Bu belirsizliğin çözülmesi P0'dan bile önce gelir.**
2. **Ayrıştırma testi.** Kasıtlı olarak düşük riskli 3-5 video ekle (normal depo operasyonu, boş koridor). Sistem bunlara "Düşük" diyebiliyor mu? Şu anki kodla **diyemez** — RAG deterministik olarak "Yüksek" üretiyor. Bu tek test P0'ın işe yarayıp yaramadığını gösterir.
3. **RAG regresyon testi.** `build_context()` için birim test: tek insan tespitli gözlemde `risk_level != "Yüksek"` olmalı ve `matched_patterns` sayısı < 20 olmalı.
4. **Miras testi.** `build_event_timestamps()` için birim test: `risk="Yüksek"` ve VLM eşleşmesi olmayan bir olay `severity="high"` **almamalı**.
5. **Dağılım izleme.** Her toplu analizden sonra risk dağılımını ve `matched_patterns` histogramını logla. Tek bir sınıfın %90'ı geçmesi alarm olsun.

---

## 6. Doğrulanamayanlar / emin olmadıklarım

- **Ground truth yok.** 14 videonun tamamı "Yüksek" olabilir; bunlar İSG kaza derlemesi videoları. Sistemin yanlış olduğunu kanıtlayan şey çıktının kendisi değil, **RAG'in içeriğe duyarsız olması** (14/14 skor tam 100, 20/20 eşleşme, sinyalsiz videoda dahi Yüksek). Karar ajanının kendi yargısı bağımsız olarak doğru olabilir — ama şu anki kurulumda bunu ayırt etmek imkânsız.
- `src/events/isg_rules_engine.py` içindeki `RiskSeverity` enum'unun ana akışa bağlı olup olmadığından emin değilim; import taramasında yalnızca `tests/test_isg_rules.py` kullanıyor görünüyor.
- RAG'in hangi yolda çalıştığı (sentence-transformers embedding mi, TF-IDF fallback mi) çıktıya yazılmıyor. Ölçülen benzerlik değerleri TF-IDF ile uyumlu görünüyor ama `all-MiniLM-L6-v2` (İngilizce model) yüklenmişse Türkçe dokümanlarda anlamsız benzerlikler üretiyor olabilir. `rag_layer.py:275-286`'daki `except (ImportError, Exception)` her hatayı yutuyor — hangi yolun aktif olduğu loglanmıyor.
- EVREN `vlm` modelinin klipten kaç kare örneklediği bu depoda kontrol edilmiyor (`preprocessing.py:397-419` yorumunda "2.0 fps, 520 kareye kadar" yazıyor, doğrulayamadım).
- `best.pt`'nin baret/yelek sınıflarındaki gerçek başarımını ölçmedim; `ppe_missing`'in %58 pay almasının model kalitesinden mi kural tasarımından mı geldiği bu yüzden kesin değil.
