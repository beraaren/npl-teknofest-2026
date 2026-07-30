# Rakip Analizi: "The Deep" (Emir-Gemici)

**Repo:** https://github.com/Emir-Gemici/The_Deep
**Yarışma:** TEKNOFEST Yapay Zeka Dil Ajanları Yarışması — 3. Senaryo

---

## Kullanılan Teknolojiler

- **Tespit:** YOLO26s (Ultralytics 8.4), COCO-pretrained, 5 sınıfa fine-tune edilmiş: `forklift`, `insan`, `palet`, `baret`, `yelek`
- **Takip:** `model.track(persist=True)` + **BoT-SORT + ReID**, özel tracker config (`custom_tracker.yaml`), uzun buffer
- **VLM / Reasoning:** **Qwen2.5-VL** (multimodal, kareleri doğrudan görüyor) — anlamlandırma ve Türkçe çıktı üretimi bu katmanda yapılıyor
- **Dil:** %100 Python

---

## Mimari / Çalışma Şeması

Pipeline net bir 3 katmanlı yapı izliyor:

```
Video → [Vision/Detection katmanı] → [VLM/Reasoning katmanı] → Nihai JSON
         (YOLO26 + tracking +            (Qwen2.5-VL: olay
          basit geometrik sinyal)         anlamlandırma + TR özet
                                           + risk + aksiyon)
```

### Vision katmanı çıktısı (`olaylar_cikti.json`)

Sadece "sinyal adayı" üretiyor, VLM'e ham girdi olarak veriliyor:

- `toplanma_adayi`: ≥3 insan track'i aynı anda yakın konumda
- `hareketsizlik_adayi`: bir insan track'inin merkezi ~2.5 sn sabit kalıyor
- Zaman damgası: `frame / fps`

### Önemli tasarım kararı — devrilme tespiti VLM'e devredilmiş

Ekip, forklift devrilmesini bbox geometrisiyle (w/h oranı) tespit etmeyi denemiş; ancak forklift'in dönüşünü (önden → yandan görünüm) gerçek devrilmeyle karıştırdığı için güvenilmez bulmuşlar. Bu fonksiyonu koddan çıkarmadan devre dışı bırakıp "devrilme" ve "tehlikeli yakınlık" kararını tamamen VLM'in görsel yorumuna bırakmışlar.

---

## Ekip Yapısı (görev dağılımı)

| Rol | Sorumluluk |
|---|---|
| Vision/Detection | Tamamlanmış (repo sahibi) |
| Bera / Selim | VLM/Reasoning — nihai JSON (summary/events/risk/actions) |
| Selin | Pipeline/Orkestrasyon + hata yönetimi |
| Doküman/Demo/Ölçümleme | Ayrı görev, henüz tamamlanmamış görünüyor |

---

## Performans / Metrikler

- **mAP50 (genel):** 0.877
  - insan: 0.927
  - forklift: 0.982
  - yelek: 0.881
  - baret: 0.836
  - palet: 0.759
- **Veri:** train 18.004 / valid 3.516 / test 2.267 görsel

---

## Bilinen Zayıflıklar (fırsat alanları)

1. **Gerçek CCTV + devrilmiş forklift zayıf tespit ediliyor** — kendileri de itiraf ediyor, hazır veri seti yok, "gelecek iş" olarak bırakmışlar. Gerçek/grainy CCTV footage'ında da çalışan bir sistem göstermek büyük fark yaratabilir.
2. **Olay tespiti sadece 2 sinyalle sınırlı** (toplanma + hareketsizlik) — devrilme, düşme, yangın, kaçış gibi diğer riskli durumlar sadece VLM'in ham görsel yorumuna bağlı; ayrı bir geometrik/olay sinyali yok. PPE-eksikliği (baretsiz/yeleksiz personel) YOLO sınıflarında var ama olay katmanına henüz bağlanmamış. Daha zengin bir olay taksonomisi avantaj sağlayabilir.
3. **Eşikler tek bir videoda ayarlanmış**, genelleme/robustluk kanıtlanmamış.
4. **3D/CGI animasyon domaininde tutarsız** — sadece gerçekçi/net görsellerde iyi çalışıyor.
5. Devrilme/yakınlık gibi kritik güvenlik olaylarını tamamen VLM'in "görüp yorumlamasına" bırakmak, düşük ışık/açı durumlarında halüsinasyon riski taşır. Buna karşı bir doğrulama/çapraz kontrol mekanizması, "Teknik İmplementasyon" ve "Otonomi" kriterlerinde öne çıkarabilir.
6. Şartnamedeki **vLLM / yerel yüksek performanslı servisleme** vurgusuna dair repo'da açık bir vLLM entegrasyon detayı görünmüyor (sadece "Qwen2.5-VL" adı geçiyor). vLLM'i gerçekten performans odaklı kullanıp bunu ölçümleyerek (gecikme, throughput) belgelemek, "%35 Teknik İmplementasyon" kriterinde fark yaratabilir.
