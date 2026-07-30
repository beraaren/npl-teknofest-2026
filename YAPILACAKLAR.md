# YAPILACAKLAR — Dalga AI / TEKNOFEST 2026 Senaryo 3

Bu liste 30.07.2026 itibarıyla günceldir. Mevcut durum: HF transformers detection
backend'i (geçici) entegre edildi, Kanal B genel VLM yorumu + kritik kare seçimi
pipeline'a bağlandı, timestamp düzeltmesi yapıldı. Aşağıdakiler **henüz yapılmadı**.

Önerilen sorumlular plan 00 §6'daki iş bölümüne göredir:
Bera (mimari/VLM backend), Talha (prompt/LLM ajanları), Hüseyin (veri/rapor), Atagün (algı/olay/test).

---

## 1. YOLO Model Eğitimi — öncelik: KRİTİK (Atagün)

- [ ] İSG özel sınıflarıyla (forklift, insan, palet, baret, yelek) eğitim veri setinin hazırlanması (bkz. §2).
- [ ] Eğitim script'i: `yolo train data=isg.yaml model=yolov8n.pt epochs=... imgsz=640` (Ultralytics CLI veya Python API).
- [ ] Augmentasyon stratejisi: parlaklık/kontrast (gece sahneleri için), motion blur, mozaik, açı çeşitliliği (devrilme durumları için döndürülmüş kareler).
- [ ] Değerlendirme: mAP50 hedefi **≥ 0.877** (rakip The Deep referansı); mAP50-95 ve sınıf bazlı precision/recall raporlanmalı.
- [ ] Eğitilen ağırlığın `config.yaml → perception.yolo_model` alanına bağlanması ve `detector_backend: "ultralytics"`e geri dönüş (şu an `hf_transformers` geçici olarak aktif — PP-DocLayoutV3 İSG sahnelerinde anlamlı tespit üretmez, sadece iskelet içindir).
- [ ] ByteTrack ile ID sürekliliği yeniden doğrulanmalı (HF backend'indeki IoU fallback'ten gerçek trackere geçiş).

## 2. Veri Etiketleme (Hüseyin + Atagün)

- [ ] Kaynak video toplama: depo/saha güvenlik kamerası görüntüleri, açık kaynak İSG videoları, sentetik sahneler.
- [ ] Etiketleme aracı seçimi ve kurulumu: CVAT / Label Studio / Roboflow (yerel çalışma kısıtına dikkat).
- [ ] Sınıf listesi kesinleştirme: forklift, insan, palet, baret, yelek (+ adaylar: yangın/duman, sıvı birikintisi).
- [ ] Zor durumların etiketlenmesi: devrilmiş forklift, yerde yatan insan, bareti olmayan işçi (negatif örnekler önemli).
- [ ] Train/val/test bölünmesi (örn. %70/%15/%15), video bazlı bölünme (aynı videodan kareler hem train hem test'e düşmemeli — veri sızıntısı).
- [ ] Etiket kalite kontrolü: çift etiketleme + uyuşmazlık çözümü, bbox tutarlılık kontrolü.

## 3. VLM Prompt Denemeleri ve Testler (Talha)

- [ ] **Genel vs spesifik prompt A/B ölçümü:** yeni genel-terim yaklaşımı ("araç", "kişi") ile eski spesifik yaklaşımın ("forklift") aynı videolarda doğruluk karşılaştırması. Hipotez: genel tanımlar halüsinasyonu azaltır (ekip kararı, 30.07).
- [ ] Değerlendirme metriği tanımı: insan değerlendirme rubriği (olay doğruluğu, zaman damgası doğruluğu, risk seviyesi uyumu) + JSON şema uyum oranı (guardrail retry sayısı üzerinden).
- [ ] Kritik kare sayısı denemeleri: `critical_frame_count` için 2/4/6 karşılaştırması (`src/preprocessing/critical_frames.py` artık hazır).
- [ ] Sıcaklık taraması: 0.05–0.3 arası kararlılık ölçümü.
- [ ] Kanal B prompt varyantları: maddeli serbest betimleme vs kısıtlı JSON; hangisinin karar ajanıyla daha iyi birleştiğinin ölçümü.
- [ ] Sonuçların `outputs/` altında deney bazlı arşivlenmesi (tarih + config hash'i ile).

## 4. Görüntü Ön İşleme — YOLO ve VLM ayrı ayrı (Atagün)

- [ ] **YOLO tarafı (Kanal A):** letterbox resize, yoğun kare örnekleme (şu an 8 kare uniform — Kanal A için daha yoğun örnekleme plan 01'de öngörülüyor), model imgsz ile uyum.
- [ ] **VLM tarafı (Kanal B):** CLAHE parametre taraması (`clahe_clip_limit`, `clahe_grid_size`), kare çözünürlüğü taraması (384x216 vs 512x288 vs 640x360 — VLM doğruluğu vs token maliyeti).
- [ ] Grid (4x2 tek görüntü) vs ayrı kareler karşılaştırması — mevcut kod ayrı kare gönderiyor; grid alternatifi `DecisionAgent.frames_to_grid`'te hazır, ölçülmedi.
- [ ] Düşük ışık / bulanık sahnelerde enhancement açık-kapalı etkisinin ölçümü.

## 5. RAG İçerik Üretimi (Hüseyin + Talha)

- [ ] `data/risk_patterns.yaml`: gerçekçi örnek risk durumları — mevcut içerik gözden geçirilip kapsam genişletilecek (devrilme, düşme, toplanma, hareketsizlik, KKD eksikliği, yakınlık, yangın/duman, sızıntı başına en az 2-3 varyant).
- [ ] `data/action_catalog.yaml`: her risk pattern'ine eşleşen örnek aksiyonlar; acil/uzun vadeli ayrımı; `data/mock_tools.yaml`'daki 5 araçla (`call_health_team`, `secure_area`, `record_incident`, `notify_supervisor`, `stop_forklift`) uyumlu aksiyon metinleri.
- [ ] Referans mevzuat taraması: `data/savunma_sanayii_guvenligi_yonetmeligi.md` içinden ilgili madde/yükümlülüklerin aksiyon metinlerine işlenmesi.
- [ ] Pattern ↔ aksiyon eşleme kapsam matrisi (hangi pattern hangi aksiyonları tetikliyor; boş pattern kalmasın).
- [ ] Retrieval doğruluk testi: bilinen senaryolarda doğru pattern/aksiyonun geldiğini doğrulayan test seti.

## 6. Diğer Gerekenler (plan 00 §5'ten kalan boşluklar)

- [ ] **`tools.execute()` gerçekten çağrılmıyor** — `main.py`'de karar sonrası seçilen mock tool'lar execute edilip sonucu çıktıya ve log'a yazılmalı (şartname mock fonksiyon kullanımını puanlıyor, %35). (Talha)
- [ ] **Test kapsamı:** events, rag, guardrail, decision agent için birim testler (şu an sadece preprocessing/perception/critical_frames/guardrail/event_engine kısmen var). (Atagün)
- [ ] **vLLM benchmark:** latency / TPS / VRAM ölçümü ve belgelenmesi (rakipten fark yaratma noktası; PagedAttention + continuous batching ile çoklu kare performansı). (Bera)
- [ ] **`tensor_split=[1,0]` çift GPU düzeltmesi** — örn. `[0.67, 0.33]` (RX 9070 16GB + RTX 4060 8GB). (Bera)
- [ ] **ObserverAgent'ın LLM yorumlayıcıya dönüşümü** (plan 02 §2.4): tespit/track/scene-graph ObjectDetector'da kaldı; ObserverAgent'ın LLM ile "işlenmiş gözlem raporu" üretmesi ve bu raporun RAG ana sorgusu + karar ajanı doğrudan girdisi olması. (Talha)
- [ ] **Demo videosu + sunum** hazırlanması (teslim zorunluluğu). (Hüseyin + tüm ekip)
- [ ] **Grader fonksiyonu:** ikinci geçişle çıktı doğrulama skoru; düşük skorda retry, son denemede "Bilmiyorum" (kısmen guardrail'de var, tamamlanacak). (Talha)
- [ ] **Post-hoc cross-validation:** nihai kararın geometrik sinyallerle karşılaştırılıp sapmada revizyon turu (plan 05). (Talha + Bera)
- [ ] **LangGraph değerlendirmesi:** Gözlemci→Orkestratör→Karar ajan grafı anlatısı (opsiyonel, mimari puanı güçlendirir). (Bera)
