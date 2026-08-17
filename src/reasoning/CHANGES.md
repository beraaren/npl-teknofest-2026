# RAG Katmanı — ByteTrack Tracking Uyumluluk Düzeltmeleri

**Tarih:** 2026-08-17  
**İlgili commit:** `440bda7` — *ByteTrack nesne takip kimlikleri (track_id) algıdan kural motoruna bağlandı*  
**Düzenlenen dosya:** `src/reasoning/rag_layer.py`  
**Test dosyası:** `tests/test_rag_tracking_compat.py` (10/10 ✅)

---

## Sorun

Son commit, `TrackedObject.to_dict()` çıktı formatını değiştirdi.  
`rag_layer.py` bu değişiklikten habersiz olduğu için tracking verileri hiç işlenmiyordu.

### Eski format (önceki commit'ler)
```python
{
    "id": 1,
    "class": "insan",
    "history": [[x0, y0], [x1, y1], ...],  # koordinat geçmişi
    "velocity": [vx, vy],                   # opsiyonel
    "bbox": [x1, y1, x2, y2]
}
```

### Yeni format (`440bda7` sonrası — `TrackedObject.to_dict()`)
```python
{
    "track_id": 1,
    "class": "insan",
    "history_length": 5,       # sadece uzunluk, liste yok
    "last_center": [cx, cy],   # son merkez koordinatı
    "speed": [vx, vy]          # son iki kare arası piksel farkı
}
```

---

## Yapılan Düzeltmeler

### 1. Hareket Tespiti (`_observations_to_natural_language`)

| | Önce | Sonra |
|---|---|---|
| **Mantık** | `len(track["history"]) >= 2` ise "hareketli" | `speed` vektörünün büyüklüğü `> 1.5 px/frame` ise "hareketli" |
| **Anahtar** | `"history"` (mevcut değil) | `"speed": [vx, vy]` |

### 2. Kinematik Kontrol Bloğu (`_observations_to_natural_language`)

| | Önce | Sonra |
|---|---|---|
| **Hız kaynağı** | `"velocity"` → yoksa `history[-1] - history[-2]` | `"speed": [vx, vy]` doğrudan |
| **Konum kaynağı** | `bbox` merkezi | Önce `"last_center"`, yoksa `bbox`'tan hesapla |
| **Bbox erişimi** | Sadece track dict'ten | Track'te yoksa `track_id` ile `detections` listesinden eşleştir |
| **Değişken adı** | `for f in forklift_tracks` (`f` → `fk` çakışması) | `for fk in forklift_tracks` |

### 3. `match_patterns()` Fonksiyonu

- **Çift format desteği:** `EventSignal` nesnesi veya `.to_dict()` çıktısı (dict) — artık her ikisi kabul edilir.
- **Yapısal eşleşme genişletildi:** Sadece `detections[].class` değil, `tracks[].class` da taranır. ByteTrack bazen aynı nesneyi detection listesine eklemiyor olabilir; tracks her zaman dolu gelir.

### 4. `recommend_tools()` — `stop_forklift` parametresi

| | Önce | Sonra |
|---|---|---|
| **ID kaynağı** | `signal.get("forklift_id")` (EventSignal'de bu alan yok) | `signal["involved_track_ids"][0]` → ilk araç track ID'si |
| **Fallback** | `"bilinmeyen_forklift"` | Önce `"object_id"`, sonra `"forklift_id"`, son olarak `"bilinmeyen_forklift"` |

---

## Test Kapsamı

`tests/test_rag_tracking_compat.py` — 10 test, tümü geçiyor:

| Test | Ne doğrular |
|------|-------------|
| `test_motion_note_from_speed_field` | `speed=[10, 5]` → "hareketli" çıktısı |
| `test_stationary_note_when_speed_zero` | `speed=[0, 0]` → "hareketsiz" çıktısı |
| `test_kinematic_high_relative_speed_detected` | Yakın forklift+insan + yüksek göreceli hız → çarpışma uyarısı |
| `test_kinematic_uses_last_center_not_history` | `"history"` anahtarı olmayan dict → hata vermez |
| `test_aspect_ratio_fall_detection` | Yatık insan bbox → düşme uyarısı |
| `test_aspect_ratio_vehicle_tip_over` | Yatık araç bbox → devrilme uyarısı |
| `test_match_patterns_with_dict_signals` | Dict sinyaller `match_patterns()`'e kabul edilir |
| `test_match_patterns_tracks_included_in_structural` | `tracks[].class` yapısal eşleşmede taranır |
| `test_stop_forklift_uses_involved_track_ids` | `forklift_id` = `involved_track_ids[0]` |
| `test_empty_observations_no_crash` | Boş liste → hata vermez, `""` döner |

---

## Veri Akışı (Güncellendi)

```
ObserverAgent.observe_frame()
    │
    ├── detections: [{"class", "track_id", "bbox", "center", ...}]
    └── tracks:     [{"track_id", "class", "history_length",
                       "last_center", "speed"}]           ← YENİ FORMAT
           │
           ▼
_observations_to_natural_language()
    • speed[vx,vy] → hareket tespiti
    • last_center   → mesafe hesabı (bbox fallback ile)
    • bbox          → aspect ratio anomali
           │
           ▼
RAGLayer.build_context() → TF-IDF / MiniLM sorgulama
           │
           ▼
RAGLayer.match_patterns() → EventSignal (nesne|dict) destekli
           │
           ▼
RAGLayer.recommend_tools()
    • stop_forklift → involved_track_ids[0]
```
