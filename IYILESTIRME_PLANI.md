# Karar Ajanını Bağlam Temelli Yeniden Tasarlama

Tarih: 2026-08-27  
Kapsam: YOLO / kural motoru (Kanal A), VLM / Kanal B, RAG, karar ajanı, guardrail ve UI

> Bu doküman önceki frekans-temelli öneriyi **geri çeker**. Bir bulgunun videonun büyük bölümünde görülmesi onu normal, güvenli veya önemsiz yapmaz. Süreklilik yalnızca bir **kanıt özelliğidir**; riskin seviyesi mekân, yapılan iş, aktif operasyon, maruziyet, mevcut korumalar ve olası zararın birlikte değerlendirilmesiyle belirlenmelidir.

---

## 1. Hedef davranış

Karar ajanı artık şu soruya cevap vermelidir:

> **Bu mekânda, o anda yürütülen iş sırasında, görülen durum veya değişim insanlara / ekipmana hangi zarar mekanizmasıyla risk yaratıyor; bu risk hangi kanıtla destekleniyor; hangi zaman aralığında geçerli?**

Bu, üç sonuç gerektirir:

1. Video genel olarak tehlikeli bir çalışma içeriği barındırsa dahi, video içindeki her saniye ve her YOLO sinyali riskli event sayılmaz.
2. Sürekli görünen bir uygunsuzluk, otomatik olarak bastırılmaz. Aksine bağlamda geçerliyse sürekli/kronik risk olarak raporlanır; fakat akut kaza ile aynı veri yapısına, aynı zaman penceresine ve aynı müdahale yoluna zorlanmaz.
3. RAG, YOLO ya da VLM tek başına hüküm vermez. Karar ajanı kanıtları karşılaştırır, çelişkileri açıklar ve yeterli kanıt yoksa **belirsiz** sonucunu üretir.

Bu tasarımda "Yüksek" etiketi katalogdan miras alınan bir değer değil; bağlama yerleştirilmiş, kanıtlanmış bir tehlike mekanizmasının sonucudur.

---

## 2. Mevcut mimaride değiştirilmesi gereken yanlış soyutlamalar

### 2.1 Sinyal = olay = risk varsayımı

Mevcut `EventSignal` (`src/events/rules.py:28-60`) her kural çıktısını aynı yapıya koyar: `event_type`, `timestamp`, `description`, `confidence`, `involved_track_ids`, `metadata`. Bu yapı yalnızca bir **algılama hipotezidir**. Henüz olayın gerçekten yaşandığını, bağlamda tehlikeli olduğunu ya da risk seviyesini söylemez.

Karar katmanına ham olarak yüzlerce sinyal gitmesi (`DecisionAgent._build_prompt`, `src/reasoning/decision_agent.py:198-203`) iki hata üretir:

- Tekrarlayan aynı algı, model tarafından bağımsız ve çok sayıda risk olarak yorumlanabilir.
- Algı sinyalinin neden üretildiği ve hangi koşullarda geçerli olduğu kaybolur; model yalnızca `ppe_missing`, `gathering` gibi etiketleri görür.

**Yeni kural:** Algı katmanının çıktısı `candidate_observation`dır. Karar ajanı bunu doğrudan event veya risk olarak kullanıcıya sunmaz.

### 2.2 RAG = ground truth varsayımı

`src/reasoning/rag_layer.py:341-438`, pattern eşleşmelerinden `risk_level` ve `risk_score` seçer; `max(matches, key=risk_score)` ile tek en yüksek katalog puanı bütün bağlamı belirler (`:428-430`). Buna ek olarak system prompt RAG riskini "ground truth" olarak tanımlar (`config.yaml`, `decision_agent.system_prompt`).

Bu iki tasarım kararı hatalıdır. Bir katalog kaydı ancak "incelenmesi gereken olası tehlike"dir. Katalogdaki sabit puan, mekân ve faaliyetten bağımsız olduğu için nihai risk olamaz.

**Yeni kural:** RAG yalnızca şu üç şey üretir:

- bağlama uygun **aday tehlike mekanizmaları**,
- geçerlilik koşulları ve aranacak **kanıt koşulları**,
- doğrulanmış event için prosedür / mevzuat / aksiyon önerileri.

RAG `risk_level`, `risk_score`, genel risk veya event severity üretmez ve karar prompt'unda otorite olarak sunulmaz.

### 2.3 Genel risk = event severity varsayımı

`scripts/analyze_video_library.py:196-246` içindeki `build_event_timestamps()`, eventin kendi severity'si yoksa analiz seviyesindeki `risk` alanını `default_severity` olarak tüm eventlere yansıtır. Ayrıca zaman yakınlığındaki VLM olayının severity'si olay tipi doğrulanmadan ödünç alınır.

Bu doğrudan kaldırılmalıdır. Bir video genel olarak yüksek riskli olabilir; bu bilgi, aynı videodaki her adayın yüksek riskli olduğu anlamına gelmez.

**Yeni kural:** Genel risk yalnızca doğrulanmış eventlerin ve bağlama uygun kalıcı bulguların sonradan oluşturulmuş bir özetidir. Hiçbir zaman event alanına geri yazılmaz.

---

## 3. Yeni karar modeli: Bağlam → Aday → Kanıt → Tehlike mekanizması → Karar

### 3.1 Aşama 0 — Mekân ve faaliyetin bağlam profilini üret

İlk işlem risk sınıflaması değildir. VLM, YOLO gözlemleri ve video zaman çizelgesinden yapılandırılmış bir `SceneContext` üretir.

```json
{
  "environment": {
    "type": "known | inferred | unknown",
    "description": "mekânın gözlenebilir nitelikleri",
    "visibility_limits": ["görüşü kısıtlayan durumlar"]
  },
  "activities": [
    {
      "activity": "gözlenen iş/faaliyet",
      "time_range": {"start_sec": 0.0, "end_sec": 0.0},
      "participants": ["rol veya nesne türü"],
      "operational_state": "active | idle | transition | unknown",
      "evidence": ["visual", "geometric"]
    }
  ],
  "zones": [
    {
      "id": "zone-1",
      "purpose": "gözlenebiliyorsa işlev",
      "active_operations": ["faaliyet kimliği"],
      "access_constraints": "known | unknown"
    }
  ],
  "context_uncertainties": ["bağlamın bilinmeyen tarafları"]
}
```

Bu profil **tek bir site etiketi** değildir. Video boyunca iş değişebilir; her `activity` ve `zone` zaman aralığı taşır. "Bilinmiyor" geçerli bir değerdir; model tahmin ederek iş türü veya zorunlu kural uyduramaz.

Bu aşamada sistem şu ayrımı yapar:

- Görüntüden gözlenebilen gerçek: etkin araç/makine, insan hareketi, üretim akışı, erişim düzeni, tehlikeli enerji/ısı/malzeme görünümü.
- Bilinmeyen yönetim bilgisi: işletmenin prosedürü, o iş için geçerli KKD matrisi, kapalı alan izni, eğitim kaydı, yetki durumu.

İkinci grup görüntüden doğrulanamaz; RAG bunlar için yalnızca **sorgulanması gereken koşullar** önerebilir. Karar ajanı bunları "ihlal" diye yazmaz.

### 3.2 Aşama 1 — Algı çıktısını aday bulguya dönüştür

YOLO/kural motoru ve VLM her biri aday üretir. Adaylar, aynı nesne/kişi/zone ve yakın zaman aralığı üzerinden birleştirilir.

```json
{
  "candidate_id": "c-42",
  "source_observations": ["geometry:fall#12", "vlm:risk_event#3"],
  "time_range": {"start_sec": 226.1, "end_sec": 229.8},
  "subject_refs": ["track:17"],
  "zone_ref": "zone-1",
  "observation_summary": "Henüz yorumlanmamış gözlem özeti",
  "repetition": {
    "occurrence_count": 3,
    "track_continuity": "continuous | switched | unknown",
    "temporal_pattern": "isolated | persistent | intermittent"
  }
}
```

Buradaki `repetition` önemlidir; ancak **risk skoru değildir**. Tekrarlama, aynı olayı tekilleştirmek ve kalıcı durumun kanıt gücünü belirlemek için kullanılır. Sürekli görülen bir uygunsuzluk, bağlamda geçerliyse kronik bir bulgu olarak korunur; yalnızca "video boyunca vardı" diye güvenli sayılmaz.

### 3.3 Aşama 2 — Tehlike mekanizmasını bağlama göre sınama

Her aday için karar ajanı şu bağımsız soruları yanıtlar:

1. **Nedir?** Gözlem gerçekten neyi gösteriyor; hangi alternatif açıklamalar mümkün?
2. **Nerede ve ne zaman?** Hangi zone'da, hangi faaliyetin sırasında, hangi kişi/nesne ilişkisi içinde gerçekleşiyor?
3. **Zarar mekanizması var mı?** Gözlem, bu faaliyet bağlamında bir yaralanma, çarpma, sıkışma, yanma, düşme, maruziyet veya operasyonel zarar zinciri oluşturuyor mu?
4. **Koruyucu / azaltıcı etken var mı?** Görüntüde fiziksel ayrım, güvenli mesafe, koruyucu ekipman, işlem dışı durum ya da tehlikeyi etkisizleştiren başka bir kanıt var mı?
5. **Kanıt yeterli mi?** Aynı iddiayı en az iki bağımsız kanal destekliyor mu? Tek kanalsa, o kanal bu iddia için yetkin mi? Çelişki var mı?
6. **Ne kadar acil?** Zararın şiddeti, maruziyetin süresi, yakınlık, hareket/enerji ve mevcut kontrolün yeterliliği birlikte ne söylüyor?

Böylece karar "etiket görüldü → sabit risk" biçiminden çıkar:

```
Gözlem + mekân + faaliyet + zaman + maruziyet + kontrol kanıtı
                         ↓
                bağlama uygun tehlike mekanizması
                         ↓
                  kanıt düzeyi ve aciliyet
                         ↓
                   event/finding/uncertain
```

### 3.4 Aşama 3 — Üç farklı sonuç türü

Karar ajanı adayları tek `events` dizisine zorlamaz.

| Sonuç | Ne zaman kullanılır? | Zaman penceresi | Bildirim |
|---|---|---|---|
| `incident` | Belirli anda gerçekleşen veya gerçekleşmek üzere olan, bağlama uygun zarar mekanizması | Kanıttan türetilmiş kısa aralık | Severity ve politika uygunsa |
| `contextual_finding` | İş/mekân bağlamında gerçek ve süreklilik gösteren uygunsuzluk veya tehlikeli çalışma koşulu | Başlangıç/bitiş ve `persistence` bilgisi; video çerçevesini sürekli yakmaz | Akut olay değilse bildirim yok; denetim / önleyici aksiyon listesine gider |
| `uncertain_observation` | Tehlike mekanizması, bağlam veya kanıt yeterince doğrulanamadı | Gözlenen zaman | İnsan incelemesine gider; risk etiketi miras almaz |

Örnekler prompt'a sabitlenmez; yukarıdaki kategoriler yalnızca davranış sözleşmesidir. Karar, videonun kendisinden çıkarılan `SceneContext` ve kanıttan gelir.

---

## 4. Risk değerlendirme sözleşmesi

### 4.1 Risk seviyesi nasıl verilir

Risk seviyesi yalnızca `incident` veya bağlama uygun `contextual_finding` için, aşağıdaki beş boyut birlikte değerlendirilerek verilir:

| Boyut | Sorulan soru | Veri kaynağı |
|---|---|---|
| Zarar potansiyeli | Gerçekleşirse olası zarar ne? | VLM gözlemi, RAG prosedür bilgisi |
| Maruziyet | Kim/kaç kişi, ne kadar süre ve ne kadar yakın? | YOLO track/mesafe, VLM, zaman çizelgesi |
| Enerji / etkileşim | Hareket, yük, ısı, yükseklik, kimyasal, sıkışma vb. kaynak var mı? | Geometrik ölçüm + VLM bağlamı |
| Koruyucu kontroller | Tehlikeyi etkisizleştiren veya azaltan kanıt var mı? | VLM/YOLO gözlemi; bilinmeyenler ayrı işaretlenir |
| Kanıt kalitesi | İddia ne kadar bağımsız ve tutarlı kanıtla destekleniyor? | Kanıt füzyonu |

Bu boyutlar **toplanıp sabit sayısal skora çevrilmez**. Sayısal eşik, farklı ortamları yanlış eşitleyeceği için karar ajanının içinde saklı bir risk puanı olmaz. Ajan her kararın `reasoning` ve yapılandırılmış `evidence` alanında hangi boyutlara dayandığını yazmak zorundadır.

Önerilen seviyeler:

- `critical`: Bağlam içinde derhal ciddi zarar tehdidi veya gerçekleşmiş ciddi olay; kanıt güçlü.
- `high`: Zarar potansiyeli ve maruziyet yüksek, koruma yetersiz veya olay sürüyor; kanıt yeterli.
- `medium`: Bağlama uygun bir tehlike / uygunsuzluk var, fakat acil zarar zinciri ya da güçlü doğrulama yok.
- `low`: Bağlama göre sınırlı etki veya düşük maruziyet; izlenmeli ancak acil müdahale gerektirmiyor.
- `unknown`: Bağlam veya kanıt yetersiz; seviye uydurulmaz.

`unknown` bir hata değil, insan denetimi gerektiren doğru sonuçtur.

### 4.2 Kanıt harmanlama kuralları

Kanal ağırlıkları sabit bir puan tablosu değildir; **iddianın türüne göre yetkinlik** kullanılır:

| İddia türü | Birincil kanıt | İkincil kanıt | Dikkat |
|---|---|---|---|
| Mesafe, hız, iz sürekliliği, yön/ölçü | Geometrik kanal | VLM | VLM sayısal ölçüm otoritesi değildir |
| Görsel bağlam, faaliyet türü, duman/alev/sızıntı, görünür kontrol | VLM | YOLO | VLM tek kaynaksa işaretlenir |
| Katalog/prosedür geçerliliği | RAG | SceneContext | RAG, görüntüde olmayan olayı kanıtlamaz |
| Zaman aralığı | Track/sinyal kümeleri | VLM zaman damgası | VLM zaman damgası yaklaşık kabul edilir |

Kanal anlaşması şu biçimde saklanır:

```json
{
  "evidence": {
    "geometric": {"supports": true, "observations": ["..."], "limitations": []},
    "visual": {"supports": true, "observations": ["..."], "limitations": ["zaman damgası yaklaşık"]},
    "rag": {"applicable": true, "references": ["..."], "limitations": ["olay kanıtı değildir"]},
    "agreement": "corroborated | single_source | conflicting | insufficient",
    "resolution": "Çelişki varsa nasıl çözüldüğü"
  }
}
```

- İki kanalın aynı iddiayı doğrulaması `corroborated` olur.
- Bir kanalın tekrar sayısı, ikinci bağımsız kanal sayılmaz.
- RAG hiçbir zaman `corroborated` sayısını artırmaz; o yalnızca bağlam/prosedür bilgisidir.
- Çelişkide model kesinlik taklidi yapmaz; `uncertain_observation` üretir veya severity'yi düşürür.

---

## 5. RAG'i karar vericiden bağlam danışmanına dönüştür

### 5.1 Veri modelini değiştir

`data/risk_patterns.yaml` şu anda her pattern için sabit `risk_score` ve `risk_level` taşır. Bu alanlar `build_context()`te doğrudan nihai riske dönüşür. Bunlar kaldırılmalı veya yalnızca tarihsel/katalog önceliği olarak karar akışından ayrıştırılmalıdır.

Yeni pattern yapısı örneği:

```yaml
patterns:
  some_hazard:
    hazard_mechanism: "zararın nasıl oluştuğu"
    applicability_questions:
      - "Bu mekânda ve aktif faaliyette ilgili tehlike kaynağı var mı?"
      - "Etkilenen kişi/nesne maruz kalıyor mu?"
    required_evidence:
      geometric: ["ölçülebilir kanıt türü"]
      visual: ["görsel kanıt türü"]
      context: ["faaliyet veya zone koşulu"]
    disconfirming_evidence:
      - "tehlikeyi etkisizleştiren görünür kontrol"
    related_controls:
      - "ilgili prosedür / önleyici tedbir"
    action_hints:
      - "kanıtlanırsa uygulanabilecek eylem"
```

Bu bir hard-code örnek listesi değildir; yeni katalog sözleşmesidir. Her kayıt, risk puanı yerine **ne zaman geçerli olduğu** ve **hangi kanıtın onu çürütebileceği** bilgisini taşır.

### 5.2 `RAGLayer.build_context()` davranışı

`src/reasoning/rag_layer.py:341-438` için hedef davranış:

1. `SceneContext` ve aday gözlemden sorgu üret.
2. En fazla `top_k` aday mekanizma getir; 0.1 sabit eşik ve "her şeyi listele" davranışı kaldırılır.
3. `matched_signal` olmayan sonuçları `unverified_hypotheses` olarak ayır; bunlar nihai risk hesabına girmez.
4. Pattern'in `applicability_questions` ve `disconfirming_evidence` alanlarını karar ajanına ilet.
5. RAG çıktısında `risk_level`, `risk_score`, genel aksiyon listesi ve maksimum risk agregasyonu **olmaz**.
6. Karar sonrasında RAG, yalnızca doğrulanmış mechanism için eylem/prosedür getirir.

Bu iki aşamalı kullanım önemlidir: önce RAG, modele "neye bakması gerektiğini" söyler; model gerçekten o tehlikeyi doğruladıktan sonra RAG "ne yapılmalı"yı söyler.

---

## 6. Karar ajanı girdisini yeniden düzenle

### 6.1 Ham sinyal listesi yerine kanıt özeti

Şu an `event_signals` doğrudan JSON olarak prompt'a giriyor. Yerine aşağıdaki gibi kümelenmiş, zaman ve bağlam bağlı bir paket üretilmelidir:

```json
{
  "candidate_observations": [
    {
      "candidate_id": "c-42",
      "time_range": {"start_sec": 0.0, "end_sec": 0.0},
      "subjects": ["track:..."],
      "zone": "zone-...",
      "geometric_evidence": {
        "type": "...",
        "occurrences": 3,
        "representative_measurements": ["..."],
        "track_continuity": "..."
      },
      "visual_evidence": {"...": "..."},
      "context_refs": ["activity-..."],
      "rag_hypotheses": ["..."]
    }
  ]
}
```

Bu özet tekrar sayısını gizlemez; fakat tekrarın **aynı kişiye/aynı zamana ait olduğunu** açıklığa kavuşturur. Modelin 76 tekrar eden sinyali 76 ayrı olay sanması engellenir.

### 6.2 Video zaman çizelgesi zorunlu girdi

Karar ajanına yalnızca son scene graph değil, bağlamın değiştiği zamanları içeren bir özet verilmelidir:

- faaliyet başlangıç/bitişleri,
- zone geçişleri,
- aday gözlemlerin aralıkları,
- VLM'in görsel bağlam değişimi,
- track sürekliliği / ID değişimi uyarıları.

Böylece model bir olayın gerçekleştiği anı, işin normal seyrini ve kanıt bulunmayan aralıkları ayırabilir. Modelden "sakin aralık uydurması" istenmez; sakin aralıklar event aralıklarının tümleyeni olarak kod tarafında hesaplanır.

---

## 7. Yeni system prompt taslağı

Aşağıdaki prompt, belirli sektör veya KKD örneklerini hard-code etmez. Mekân ve iş bilgisini kendisinin ürettiği `SceneContext`ten almasını; kanıtları harmanlamasını zorlar.

```text
You are an occupational health and safety decision agent for video analysis.
Your task is not to repeat detector labels, catalogue entries, or the most
frequent signal. Your task is to determine, from the evidence as a whole,
what work is happening, where it is happening, what changed over time, and
whether a real hazard mechanism exists in that specific context.

EVIDENCE SOURCES
You receive a scene/activity context, clustered geometric observations,
an independent visual interpretation, and RAG hazard references.
No source is ground truth:
- Geometric evidence is strongest for measured motion, spatial relation,
  track continuity, and duration. It can be wrong when objects are occluded,
  boxes are small, or tracker identities switch.
- Visual evidence is strongest for visible context, task, posture, material,
  and controls. It cannot establish exact distance or speed and may have
  approximate timestamps.
- RAG references are candidate hazard mechanisms, applicability questions,
  and controls. They are not evidence that an incident occurred and their
  catalogue priority is never a risk decision.

WORK IN THIS ORDER
1. Read SceneContext first. Identify the observed environment, active work,
   zones, people/equipment involved, and what is unknown. Do not invent an
   industry, procedure, or required control that the evidence does not show.
2. For each candidate observation, decide whether it is relevant to the
   active activity and zone. A detector label alone is not a hazard.
3. Explain the hazard mechanism: who or what is exposed, to which source of
   harm, during which activity, and which visible controls reduce or fail to
   reduce that exposure. If this chain cannot be established, do not promote
   the candidate to a risk event.
4. Fuse evidence by competence. Use geometric evidence for measurements and
   visual evidence for contextual interpretation. Treat RAG as a question to
   test, never as proof. Repeated evidence from one source is persistence,
   not independent corroboration.
5. Resolve conflicts explicitly. If evidence is insufficient or conflicting,
   return uncertain_observation with uncertainty_reason="emin değilim" and
   state what evidence is missing. Do not manufacture a severity to appear
   decisive.
6. Separate an incident from a contextual finding. An incident is a bounded
   harmful or imminently harmful occurrence. A contextual finding is a
   persistent condition that is relevant to the current work but is not itself
   a bounded incident. Preserve relevant persistent findings; do not turn
   them into a full-video alarm or inherit the overall video risk.
7. Deduplicate: records that refer to the same subject, zone, harm mechanism,
   and continuous time range are one observation or incident.
8. Assign severity only after the mechanism, exposure, visible controls, and
   evidence agreement are stated. Overall video risk is a summary of confirmed
   results. Never copy it into an individual result.

OUTPUT REQUIREMENTS
For every result, include: result_type (incident | contextual_finding |
uncertain_observation), time range, affected subjects/zone, hazard mechanism,
severity (critical | high | medium | low | unknown), evidence by source,
evidence agreement, and a concise Turkish explanation.
Use "unknown" and "emin değilim" when the context or evidence is inadequate.
Return only JSON that matches the requested schema.
```

Mevcut prompt'tan kaldırılacak ifadeler:

- `The risk level and actions in the RAG context are the catalog's ground truth`
- `if ... requires a higher risk, do not hesitate to deviate`
- `always take precedence` biçimindeki, kanıt gücünü aşan mutlak öncelik cümleleri

Yerine yukarıdaki gibi **mekanizma + kanıt + bağlam** zorunluluğu gelir.

---

## 8. Yeni çıktı sözleşmesi

`src/output/schema.py` ve karar promptundaki JSON şeması şu modele geçmelidir:

```json
{
  "summary": "Türkçe, bağlamı ve doğrulanmış sonuçları özetleyen metin",
  "scene_context": {
    "environment": "...",
    "activities": [],
    "zones": [],
    "uncertainties": []
  },
  "results": [
    {
      "result_type": "incident | contextual_finding | uncertain_observation",
      "event_type": "...",
      "time": "MM:SS",
      "end_time": "MM:SS",
      "timestamp_sec": 0.0,
      "duration": 0.0,
      "subjects": ["..."],
      "zone": "...",
      "hazard_mechanism": "...",
      "severity": "critical | high | medium | low | unknown",
      "evidence": {
        "geometric": {"supports": false, "observations": [], "limitations": []},
        "visual": {"supports": false, "observations": [], "limitations": []},
        "rag": {"applicable": false, "references": []},
        "agreement": "corroborated | single_source | conflicting | insufficient",
        "resolution": "..."
      },
      "uncertain": false,
      "uncertainty_reason": ""
    }
  ],
  "overall_risk": "critical | high | medium | low | unknown",
  "actions": [],
  "reasoning": "...",
  "uncertain": false,
  "uncertainty_reason": ""
}
```

Kurallar:

- `overall_risk` yalnız üst düzey özet alanıdır; `results[].severity`ye kopyalanmaz.
- `contextual_finding` için `duration` yalnız gözlenen süreyi temsil eder; replay alarm penceresi değildir.
- `uncertain_observation` severity miras almaz; `severity="unknown"` olur.
- `confidence` alanı kullanıcıya gösterilecek bir LLM öz-beyanı olmaktan çıkar. İstenirse sadece iç metrikte kalır. UI'da `evidence.agreement` ve `uncertain` metinsel olarak gösterilir.

---

## 9. Guardrail ve UI davranışı

### 9.1 Guardrail

`src/output/guardrail.py` sadece JSON biçimini değil, aşağıdaki tutarlılıkları denetlemelidir:

- `high`/`critical` sonuçta boş `hazard_mechanism` veya `evidence.agreement="insufficient"` olamaz.
- `incident` sonucu için zaman aralığı, subject/zone veya bunların açıkça bilinmediği notu bulunmalıdır.
- `contextual_finding`, otomatik olarak bildirim ve acil araç çağrısı üretemez.
- `uncertain_observation` için `uncertainty_reason` zorunludur.
- Genel risk, tek başına bir result'un severity'sini yükseltemez.
- RAG referansı `geometric` veya `visual` kanıt yerine sayılmaz.

Guardrail kararın daha yüksek veya daha düşük olmasını zorlamaz; sadece **kanıtsız kesinliği** engeller.

### 9.2 UI

UI üç sonucu ayrıştırmalıdır:

- `incident`: zaman çizelgesinde, yalnız kendi doğrulanmış aralığında çerçeve / bildirim gösterir.
- `contextual_finding`: "Çalışma bağlamı bulguları" bölümünde gösterir; videonun tamamını kırmızı yapmaz.
- `uncertain_observation`: "İnsan incelemesi gerekli" etiketiyle gösterir; Yüksek risk rozeti ile karıştırılmaz.

Saha atamasında (`backend/gateway/routers/ops.py:155,345`) `analysis.risk` yerine seçilen result'un `severity`, `result_type`, `hazard_mechanism` ve kanıt özeti yazılmalıdır. Üst seviye genel risk, yalnız bağlam bilgisi olarak ayrı taşınabilir.

---

## 10. Uygulama sırası

1. **Veri sözleşmesi:** `SceneContext`, aday gözlem kümesi, `results[]`, `evidence`, `uncertain` modellerini ekle (`src/output/schema.py`, `backend/contracts/messages.py`).
2. **Sinyal kümelendirme:** `EventEngine` sonrasında candidate builder oluştur; ham sinyali subject/zone/zaman üzerinde tekilleştir. Bu katman risk seviyesi hesaplamaz.
3. **Bağlam çıkarımı:** Kanal B'nin çıktısını, faaliyet/zone/operasyon durumu/bilinmeyenler içeren `SceneContext`e genişlet. Parse başarısını da açık alan olarak taşı.
4. **RAG ayrıştırması:** `risk_score`/`risk_level` ile maksimum agregasyonu karar yolundan kaldır; patternleri geçerlilik sorusu, çürüten kanıt ve kontrol kaynağına dönüştür.
5. **Yeni prompt ve karar şeması:** Yukarıdaki harmanlama sözleşmesini `config.yaml` ve `DecisionAgent._build_prompt()`a uygula. Prompt'a ham pattern listesi ve ham sinyal seli verme.
6. **Guardrail:** kanıtsız kesinlik, risk mirası ve belirsizlik kaybını yakala.
7. **Event zamanlayıcı:** `build_event_timestamps()`te genel risk → severity fallback'ini ve tip kontrolsüz ±5 sn severity ödünç almayı kaldır.
8. **UI / assignment:** üç sonuç tipini ve metinsel kanıt durumunu göster; atamayı event/finding seviyesinde yap.
9. **Katalog migrasyonu:** patternleri mekanizma/koşul/kanıt/çürüten kanıt sözleşmesine geçir; site/iş bilgisini prompta hard-code etme.

---

## 11. Başarı ölçütleri

Bu tasarım "Yüksek" oranını yapay biçimde düşürmeyi hedeflemez. Riskli videolar riskli kalmalıdır. Başarı şu ölçütlerle doğrulanır:

1. **Bağlam uygunluğu:** Her high/critical sonuçta `SceneContext`teki faaliyet veya zone ile açık ilişki ve zarar mekanizması bulunuyor mu?
2. **Kanıt izi:** Her kesin sonuçta hangi kanalın neyi kanıtladığı ve sınırlılığı görülebiliyor mu?
3. **Miras olmaması:** Aynı videodaki düşük/unknown bir aday, video `overall_risk=high` diye high olmuyor mu?
4. **Zamansal doğruluk:** Incident yalnız doğrulanmış kendi aralığında replay/bildirim oluşturuyor mu?
5. **Kalıcı riskin korunması:** Bağlama uygun sürekli bulgular silinmeden `contextual_finding` olarak kayda geçiyor mu?
6. **Belirsizlik dürüstlüğü:** Görüntüden belirlenemeyen prosedür ve bağlamlar "emin değilim" / `unknown` olarak UI'a kadar ulaşıyor mu?
7. **RAG disiplinı:** Katalog eşleşmesi tek başına event, severity veya acil aksiyon üretmiyor mu?

---

## 12. Bilinçli olarak yapılmayanlar

- "Bir sinyal videonun %60'ından fazlasında görülürse güvenlidir" gibi frekans tabanlı bir kural **yoktur**.
- Belirli bir sektör, mekan veya kişisel koruyucu ekipman örneği system prompt'a risk kuralı olarak **gömülmez**.
- Her katalog patternine sabit nihai risk seviyesi atanmaz.
- VLM'in ya da LLM'in 0-1 öz-güveni kullanıcıya yüzde/ondalık skor olarak sunulmaz.
- Genel analiz riski tekil event/finding severity'sine kopyalanmaz.

Bu sınırlar tasarımın temelidir: sistem, önceden ezberletilmiş etiketlerden değil, o videoda çıkarılan mekân + faaliyet + kanıt ilişkisinden karar vermelidir.
