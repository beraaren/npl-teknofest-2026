"""Geometrik ve Uzamsal İSG Olay Tespit Kuralları Modülü.

Bu modül, Kanal A algılama katmanından gelen nesne takipleri (TrackedObject),
durum geçmişleri (TrackStateMachine) ve anlamsal sahne grafiği (SceneGraph)
verilerini geometrik kurallarla değerlendirerek tehlikeli İSG durumlarını tespit eder.

Tespit Edilen Temel Olaylar:
  1. Forklift Devrilmesi (`forklift_tip_over`): En/boy oranının (aspect ratio) değişmesi.
  2. İnsan Düşmesi (`person_fall`): Dikey eksende (Y ekseni) ani ve yüksek hızlı düşüş hareketi.
  3. Tehlikeli Toplanma (`gathering`): Belirli bir mesafe içinde birden fazla personelin kümelenmesi.
  4. KKD Eksikliği (`ppe_missing`): Sahne grafiğinde baret veya yelek giyilme ilişkisinin bulunmaması.
  5. Tehlikeli Yakınlık (`dangerous_proximity`): Forklift ile yaya arasındaki mesafenin risk sınırına inmesi.
  6. Yangın / Duman (`fire_smoke`): `yangin` veya `duman` sınıfının süreklilik eşiğini aşarak tespiti.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..perception.scene_graph import SceneEdge, SceneGraph, SceneNode
from ..perception.tracker import TrackedObject
from .state_machine import TrackStateMachine


@dataclass
class EventSignal:
    """Tespit edilen bir İSG olay sinyalini temsil eden veri sınıfı.

    Attributes:
        event_type (str): Olayın tür kodu (örn. 'forklift_tip_over', 'person_fall', 'ppe_missing').
        timestamp (float): Olayın gerçekleştiği saniye cinsinden zaman damgası.
        description (str): Olayın insan tarafından okunabilir Türkçe açıklaması.
        confidence (float): 0.0 ile 1.0 arasında olayın kesinlik / güven skoru.
        involved_track_ids (List[int]): Olaya karışan nesnelerin takip kimlikleri (track_id).
        metadata (Dict[str, Any]): Olayla ilgili ek sayısal ve geometrik detaylar (hız, mesafe, oran vb.).
    """

    event_type: str
    timestamp: float
    description: str
    confidence: float
    involved_track_ids: List[int]
    metadata: Dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Olay sinyalini JSON uyumlu sözlük formatına dönüştürür.

        Returns:
            dict[str, Any]: 'MM:SS' formatında zaman damgası ve olay detaylarını içeren sözlük.
        """
        return {
            "event_type": self.event_type,
            "timestamp": self._format_time(self.timestamp),
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "involved_track_ids": self.involved_track_ids,
            "metadata": self.metadata,
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Saniye cinsinden zamanı 'MM:SS' (Dakika:Saniye) formatına dönüştürür.

        Args:
            seconds (float): Saniye değeri.

        Returns:
            str: '01:24' gibi biçimlendirilmiş zaman metni.
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


class RuleSet:
    """Yapılandırılabilir İSG geometrik kurallar kümesi yöneticisi.

    `config.yaml` içindeki eşik değerlerini ve aktif kural listesini alarak
    her video karesi için kuralları çalıştırır ve üretilen olay sinyallerini döner.

    Attributes:
        thresholds (Dict[str, Any]): Kurallara ait eşik ve parametre sözlüğü.
        fps (float): Videonun zaman hesaplamalarında kullanılan efektif kare/saniye hızı.
    """

    def __init__(self, thresholds: Dict[str, Any], fps: float = 25.0):
        """RuleSet nesnesini başlatır.

        Args:
            thresholds: config.yaml'daki 'events' bloğu ve eşik değerleri.
            fps: Video kare/saniye hızı (varsayılan 25.0).
        """
        self.thresholds = thresholds
        self.fps = fps
        # Aynı RuleSet/EventEngine oturumu içinde bir kişiye güvenilir biçimde
        # baret veya yelek atanmışsa, ilgili KKD daha sonraki karelerde YOLO
        # tarafından kaçırılsa bile o kişiyi o ekipmandan yoksun sayma. Hafıza
        # yalnız kişi track ID'sine bağlıdır ve yeni RuleSet oluşturulduğunda
        # sıfırlanır. ObserverAgent'ın kısa süreli `disappeared < 5` toleransı
        # bundan bağımsız olarak çalışmaya devam eder.
        self._confirmed_ppe_person_tracks: Dict[str, set[int]] = {
            "baret": set(),
            "yelek": set(),
        }

    def evaluate(
        self,
        tracks: List[TrackedObject],
        states: TrackStateMachine,
        graph: SceneGraph,
    ) -> List[EventSignal]:
        """Tüm aktif kuralları değerlendirir ve tespit edilen olay sinyallerini toplar.

        Args:
            tracks (List[TrackedObject]): Mevcut karedeki aktif takip nesneleri.
            states (TrackStateMachine): Nesnelerin zamansal durum ve hareket geçmişi makinesi.
            graph (SceneGraph): Karedeki nesneler arası anlamsal ve mekansal ilişkiler grafiği.

        Returns:
            List[EventSignal]: Bu karede tetiklenen tüm olay sinyalleri listesi.
        """
        signals: List[EventSignal] = []

        enabled = self.thresholds.get("enabled_rules", [])

        if "tip_over" in enabled:
            signals.extend(self._rule_tip_over(tracks, states))
        if "fall" in enabled:
            signals.extend(self._rule_fall(tracks, states))
        if "gathering" in enabled:
            signals.extend(self._rule_gathering(tracks, states))
        if "ppe_missing" in enabled:
            signals.extend(self._rule_ppe_missing(graph))
        if "proximity" in enabled:
            signals.extend(self._rule_proximity(graph, states))
        if "fire_smoke" in enabled:
            signals.extend(self._rule_fire_smoke(tracks))

        return signals

    def _rule_tip_over(self, tracks: List[TrackedObject], states: TrackStateMachine) -> List[EventSignal]:
        """Forklift Devrilmesi Kuralı: En/Boy Oranı + Yükseklik Çöküşü Analizi.

        `TrackState.update()` (`state_machine.py`) içinde sayaç (`tip_over_frames`)
        artık iki koşulun **birlikte** sağlanmasını gerektiriyor: bbox en/boy
        oranının yüksek olması (`ar >= aspect_ratio_min`, forklift yandan/yatık
        görünüyor) VE aynı pencerede bbox yüksekliğinin hızla düşmüş olması
        (kabinin gerçekten yere çökmesi). Sadece oran yüksekliği tek başına
        yeterli değildir: forklift kameraya dönerken de (yaw) yandan görünüme
        geçip geniş görünür, ama fiziksel olarak yatmamıştır (bkz. proximity.mp4
        gözlemi). Bu ayrımın detayları `state_machine.py` içindedir; bu kural
        sadece sayacın süreklilik eşiğini (`min_duration_frames`) ve son kontrolü
        (`aspect_ratio_min`) uygular.

        Args:
            tracks: Aktif takip listesi.
            states: Nesne durum makinesi.

        Returns:
            List[EventSignal]: Devrilen forkliftler için üretilen sinyaller.
        """
        signals = []
        cfg = self.thresholds.get("tip_over", {})
        min_frames = cfg.get("min_duration_frames", 3)
        ar_min = cfg.get("aspect_ratio_min", 1.45)

        for t in tracks:
            if t.class_name != "arac":
                continue
            state = states.get(t.track_id)
            if state and state.tip_over_frames >= min_frames:
                det = t.last_detection
                if det.aspect_ratio >= ar_min:
                    signals.append(
                        EventSignal(
                            event_type="forklift_tip_over",
                            timestamp=t.last_detection.frame_idx / self.fps,
                            description=f"Araç (track {t.track_id}) devrilme pozisyonunda; en/boy oranı {det.aspect_ratio:.2f}",
                            confidence=min(1.0, det.aspect_ratio / 3.0),
                            involved_track_ids=[t.track_id],
                            metadata={"aspect_ratio": det.aspect_ratio},
                        )
                    )
        return signals

    def _rule_fall(self, tracks: List[TrackedObject], states: TrackStateMachine) -> List[EventSignal]:
        """Personel Düşmesi Kuralı: Ölçeğe Oranlı, Pencereli Kinematik Analiz.

        Gerçek düşüşler her zaman tek karede olmaz; birkaç kareye yayılabilir
        (bkz. video_ile_test/Completely_stationary_overhead.mp4 — ~20 kare/0.67s
        sürüyor, tek karedeki en büyük sıçrama sabit piksel eşiğinin altında
        kalıyordu). Bu yüzden:

          1. Anlık hız yerine `window_seconds`'a denk gelen kare sayısı boyunca
             kümülatif dikey yer değiştirmeye bakılır (`TrackedObject.displacement`).
          2. Yer değiştirme, sabit piksel yerine track'in kendi ölçeğine
             (`TrackState.scale_ema`, bbox yüksekliği ortalaması) oranlanır.
             Bu oran kameraya uzaklıktan bağımsızdır: yakın/büyük görünen bir
             personelin küçük bir gerçek hareketi kadar piksel kaymasını,
             uzak/küçük bir personelin büyük bir gerçek hareketi kadar piksel
             kaymasıyla eşit muameleye tabi tutar.

        `min_drop_ratio` (varsayılan 0.35), pencere boyunca dikey yer
        değiştirmenin track'in kendi ölçeğine oranı için eşiktir (örn. 0.35 =
        kendi bbox yüksekliğinin %35'i kadar düşüş).

        Args:
            tracks: Aktif takip listesi.
            states: Nesne durum makinesi.

        Returns:
            List[EventSignal]: Düşen personel için üretilen sinyaller.
        """
        signals = []
        cfg = self.thresholds.get("fall", {})
        window_seconds = cfg.get("window_seconds", 0.4)
        min_drop_ratio = cfg.get("min_drop_ratio", 0.35)
        window_frames = max(1, round(window_seconds * self.fps))

        for t in tracks:
            if t.class_name != "insan":
                continue
            state = states.get(t.track_id)
            if not state or state.scale_ema <= 0.0:
                continue

            _, dy = t.displacement(window_frames)
            drop_ratio = dy / state.scale_ema

            if drop_ratio >= min_drop_ratio:
                signals.append(
                    EventSignal(
                        event_type="person_fall",
                        timestamp=t.last_detection.frame_idx / self.fps,
                        description=f"Personel (track {t.track_id}) düşme hareketi gösteriyor.",
                        confidence=min(1.0, drop_ratio / (min_drop_ratio * 2)),
                        involved_track_ids=[t.track_id],
                        metadata={
                            "vertical_displacement_pixels": dy,
                            "scale_pixels": state.scale_ema,
                            "drop_ratio": drop_ratio,
                        },
                    )
                )
        return signals

    def _rule_gathering(self, tracks: List[TrackedObject], states: TrackStateMachine) -> List[EventSignal]:
        """Tehlikeli Toplanma Kuralı: Ölçeğe Oranlı Mekansal Kümeleme Analizi.

        Sabit piksel mesafesi (`max_inter_center_distance`) yerine, çiftin
        ortalama ölçeğine (`TrackState.scale_ema`) oranlı bir mesafe eşiği
        kullanılır. Kameraya yakın (büyük bbox) kişiler için piksel mesafesi
        doğal olarak büyük olur; sabit eşik bu durumda kümelenmiş kişileri
        "uzak" sayabilir. Ölçek bilinmiyorsa (scale_ema henüz hesaplanmamışsa)
        güvenli tarafta kalınıp `max_inter_center_distance` piksel eşiğine
        geri dönülür.

        `max_inter_center_ratio` (varsayılan 1.5), çiftin ortalama bbox
        yüksekliğinin kaç katına kadar "yakın" sayılacağını belirler.

        Args:
            tracks: Aktif takip listesi.
            states: Nesne durum makinesi (ölçek bilgisi için).

        Returns:
            List[EventSignal]: Toplanan gruplar için üretilen sinyaller.
        """
        signals = []
        cfg = self.thresholds.get("gathering", {})
        min_persons = cfg.get("min_persons", 3)
        max_dist_fallback = cfg.get("max_inter_center_distance", 120)
        max_dist_ratio = cfg.get("max_inter_center_ratio", 1.5)

        persons = [t for t in tracks if t.class_name == "insan"]
        if len(persons) < min_persons:
            return signals

        def _pair_max_dist(p: TrackedObject, q: TrackedObject) -> float:
            sp = states.get(p.track_id)
            sq = states.get(q.track_id)
            scales = [s.scale_ema for s in (sp, sq) if s and s.scale_ema > 0.0]
            if not scales:
                return max_dist_fallback
            return (sum(scales) / len(scales)) * max_dist_ratio

        # Basit kümeleme: birbirine yakın kişileri bul
        clusters = []
        visited = set()
        for p in persons:
            if id(p) in visited:
                continue
            cluster = [p]
            visited.add(id(p))
            for q in persons:
                if id(q) in visited:
                    continue
                dist = math.hypot(p.last_detection.center[0] - q.last_detection.center[0],
                                   p.last_detection.center[1] - q.last_detection.center[1])
                if dist <= _pair_max_dist(p, q):
                    cluster.append(q)
                    visited.add(id(q))
            if len(cluster) >= min_persons:
                clusters.append(cluster)

        for cluster in clusters:
            signals.append(
                EventSignal(
                    event_type="gathering",
                    timestamp=cluster[0].last_detection.frame_idx / self.fps,
                    description=f"{len(cluster)} personel tehlikeli bölgede toplanmış.",
                    confidence=min(1.0, len(cluster) / 5.0),
                    involved_track_ids=[t.track_id for t in cluster],
                    metadata={"person_count": len(cluster)},
                )
            )
        return signals

    def _rule_fire_smoke(self, tracks: List[TrackedObject]) -> List[EventSignal]:
        """Yangın / Duman Kuralı: Tehlike Sınıfı Varlığı ve Süreklilik Analizi.

        `yangin` (fire) veya `duman` (smoke) sınıfı bir tespit, `min_duration_frames`
        kadar kare boyunca kesintisiz takip edilmişse sinyal üretilir. Diğer
        kuralların aksine geometri veya kinematik gerekmez — bu sınıfların
        **varlığı** başlı başına tehlikedir; bu yüzden ölçek/mesafe normalizasyonu
        uygulanmaz (bkz. `_rule_fall`, `_rule_proximity`).

        Süreklilik koşulunun amacı yanlış pozitifleri elemektir: duman ve alev
        sınıfları renk/doku tabanlı olduğu için tek karelik parlama, yansıma veya
        toz bulutu gibi görüntü artefaktlarında sahte tespit üretmeye yatkındır.
        Ayrıca `min_confidence` ile düşük güvenli tespitler filtrelenir.

        Yangın ve duman ayrı `risk_patterns.yaml` girdisi değil aynı `fire_smoke`
        pattern'ine düşer (risk skoru 100); hangisinin görüldüğü `metadata`
        içindeki `hazard_class` alanında ve açıklamada raporlanır.

        Args:
            tracks: Aktif takip listesi (kalıcı `TrackedObject` örnekleri; geçmiş
                uzunluğu `len(t.history)` kaç karedir görüldüğünü verir).

        Returns:
            List[EventSignal]: Tespit edilen yangın/duman için üretilen sinyaller.
        """
        signals = []
        cfg = self.thresholds.get("fire_smoke", {})
        hazard_classes = cfg.get("classes", ["yangin", "duman"])
        min_frames = cfg.get("min_duration_frames", 3)
        min_confidence = cfg.get("min_confidence", 0.35)

        display_names = {"yangin": "Yangın/alev", "duman": "Duman"}

        for t in tracks:
            if t.class_name not in hazard_classes:
                continue
            if len(t.history) < min_frames:
                continue

            det = t.last_detection
            if det.confidence < min_confidence:
                continue

            label = display_names.get(t.class_name, t.class_name)
            signals.append(
                EventSignal(
                    event_type="fire_smoke",
                    timestamp=det.frame_idx / self.fps,
                    description=f"{label} tespit edildi (track {t.track_id}).",
                    confidence=det.confidence,
                    involved_track_ids=[t.track_id],
                    metadata={
                        "hazard_class": t.class_name,
                        "detection_confidence": det.confidence,
                        "observed_frames": len(t.history),
                    },
                )
            )
        return signals

    def _rule_ppe_missing(self, graph: SceneGraph) -> List[EventSignal]:
        """Kişisel Koruyucu Donanım (KKD) Eksikliği Kuralı: Sahne Grafiği Analizi.

        Karedeki her bir `insan` düğümü için sahne grafiğindeki `wearing` ilişkilerini
        denetler. Gerekli sınıflardan (`baret`, `yelek`) herhangi biriyle `wearing`
        ilişkisi bulunamazsa KKD eksikliği sinyali üretir.

        Bir kişi track ID'sine en az bir karede güvenilir bir `wearing:baret`
        veya `wearing:yelek` ilişkisi kurulmuşsa bu bilgi mevcut kural motoru
        oturumu boyunca korunur. YOLO'nun sonraki karelerde ilgili KKD'yi
        kaçırması o kişiyi yeniden o ekipmandan yoksun yapmaz.

        Yalnızca ilgili personele **bağlı** kenarlar sayılır; başka bir personelin
        ekipman kenarı bu personeli KKD'li saymaz. Kenar yönü hoşgörülü okunur
        (kaynak veya hedef bu personel olabilir), ancak kenarın taraflarından biri
        mutlaka bu personel olmalıdır.

        `wearing` ilişkisinin kendisi `SceneGraph.build_relations()` içinde kapsama
        (bounding box bandı) yöntemiyle kurulur; bu kural piksel eşiği kullanmaz.

        Args:
            graph (SceneGraph): Karenin anlık sahne grafiği.

        Returns:
            List[EventSignal]: Baret veya yeleği eksik olan personel için üretilen sinyaller.
        """
        signals = []
        cfg = self.thresholds.get("ppe_missing", {})
        ppe_classes = cfg.get("classes", ["baret", "yelek"])

        persons = graph.find_nodes("insan")
        for person in persons:
            has_ppe = {ppe: False for ppe in ppe_classes}
            for edge in graph.edges:
                if edge.relation != "wearing":
                    continue
                # Kenarın bu personele ait olduğu doğrulanmalıdır. Aksi halde başka
                # bir personelin ekipman kenarı da bu personeli KKD'li sayar ve
                # karede tek kişi baret takıyorsa herkes muaf hale gelir.
                if edge.source == person.node_id:
                    other_id = edge.target
                elif edge.target == person.node_id:
                    other_id = edge.source
                else:
                    continue
                other = graph.nodes.get(other_id)
                if other and other.class_name in ppe_classes:
                    has_ppe[other.class_name] = True

            # Bir kez kişiyle güvenilir `wearing:baret` veya `wearing:yelek`
            # ilişkisi kurulduğunda ilgili KKD'yi aynı track ID'sinin kalan ömrü
            # boyunca pozitif kanıt olarak koru. Böylece uzun süren YOLO KKD
            # kaçırmaları yanlış eksiklik üretmez. Track ID'si olmayan
            # düğümlerde kişiler arası durum sızıntısını önlemek için hafıza
            # kullanılmaz.
            person_track_id = person.track_id
            if person_track_id is not None:
                for ppe_class, confirmed_tracks in self._confirmed_ppe_person_tracks.items():
                    if ppe_class not in has_ppe:
                        continue
                    if has_ppe[ppe_class]:
                        confirmed_tracks.add(person_track_id)
                    elif person_track_id in confirmed_tracks:
                        has_ppe[ppe_class] = True

            missing = [p for p, v in has_ppe.items() if not v]
            if missing:
                signals.append(
                    EventSignal(
                        event_type="ppe_missing",
                        timestamp=graph.timestamp,
                        description=f"Personel {person.node_id} eksik KKD: {', '.join(missing)}.",
                        confidence=0.7,
                        involved_track_ids=[person.track_id] if person.track_id is not None else [],
                        metadata={"missing_ppe": missing},
                    )
                )
        return signals

    @staticmethod
    def _person_vehicle_overlap_ratio(person: SceneNode, vehicle: SceneNode) -> float:
        """Kişinin bbox alanının araç bbox'ıyla kesişen oranını (0-1) hesaplar.

        Bu oran yüksekse (varsayılan eşik 0.5) kişi aracın içinde/üzerinde
        oturuyor demektir — yani forklift **operatörü**, ayrı bir yaya değil.
        `_rule_proximity` bu durumu tehlikeli yakınlık olarak saymamalıdır:
        merkez mesafesi tabanlı `near` ilişkisi, operatörün bbox'ı aracın
        bbox'ıyla ağır örtüştüğü için sırf bu yüzden çok küçük çıkar ve
        sürücüyü "aşırı yakın yaya" gibi işaretler.

        Kavramsal olarak `wearing` ilişkisinin (`_is_ppe_worn`) kullandığı
        kapsama mantığıyla aynıdır, ama genel `SceneGraph` ilişkilerine
        eklenmez: bu ayrım yalnızca `dangerous_proximity` kuralının kendi iç
        mantığına ait, sahne grafiğinin başka tüketicilerini (VLM prompt'u,
        JSON raporu, diğer kurallar) etkilememesi için burada, yerel olarak
        tutulur.

        Args:
            person: `insan` sınıfındaki düğüm.
            vehicle: `arac` sınıfındaki düğüm.

        Returns:
            Kesişim alanının kişinin bbox alanına oranı. Kesişim yoksa veya
            kişinin bbox'ı dejenere (alan <= 0) ise 0.0.
        """
        px1, py1, px2, py2 = person.bbox
        vx1, vy1, vx2, vy2 = vehicle.bbox
        ix1, iy1 = max(px1, vx1), max(py1, vy1)
        ix2, iy2 = min(px2, vx2), min(py2, vy2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        person_area = (px2 - px1) * (py2 - py1)
        if person_area <= 0.0:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        return intersection / person_area

    def _rule_proximity(self, graph: SceneGraph, states: TrackStateMachine) -> List[EventSignal]:
        """Tehlikeli Yakınlık Kuralı: Forklift ve Yaya Etkileşimi.

        Sahne grafiğindeki `near` ilişkilerini inceler. Eğer birbiriyle yakın olan
        nesne çifti tehlikeli ikili listesindeyse (örn. `['forklift', 'insan']`)
        gerçek mesafe (`estimated_dist`) şu eşiklerden **büyük olanının** altında
        kalmalıdır:
          1. Sabit piksel eşiği (`distance_threshold_pixels`) — ölçek bilgisi
             yoksa (ilk kareler) güvenli taban.
          2. Çiftin ortalama ölçeğine (`TrackState.scale_ema`) oranlı mesafe
             (`distance_threshold_ratio`) — kameraya yakın/büyük görünen bir
             çift için gerçek tehlike mesafesi, sabit pikselden büyük olabilir.

        Önceki tasarım iki eşiğin **küçüğünü** (`min`) kullanıyordu; bu, oranlı
        eşiği anlamsız kılıyordu çünkü sabit eşik neredeyse her zaman daha
        küçük çıkıp oranlı hesabı gölgeliyordu (bkz. proximity.mp4 gözlemi:
        forklift 278px'e kadar yaklaştı — kişilerin kendi ölçeğine göre bu zaten
        "tehlikeli yakın" sayılması gereken bir mesafeydi — ama sabit 100px
        eşiği hep kazanıp sinyali bastırıyordu). `max()` bu ölçek-duyarlı
        davranışı doğru yöne çevirir: kameraya yakın/büyük çiftlerde daha büyük
        (daha erken tetiklenen), uzak/küçük çiftlerde sabit taban eşiği geçerli
        olur.

        Bu kuralın çalışabilmesi için `near` kenarının önce oluşmuş olması
        gerekir; `EventEngine._compute_proximity_graph_threshold` bu kenarın
        sabit eşik yüzünden hiç kurulmamasını önlemek için sahne grafiğine
        geçirilen eşiği o karenin ölçeğine göre büyütür.

        Operatör filtresi: kişi ile aracın bbox'ları ağır örtüşüyorsa (bkz.
        `_person_vehicle_overlap_ratio`, eşik `operator_overlap_ratio`), bu
        kişi aracın içinde/üzerinde oturan operatördür, yaya değildir; merkez
        mesafesi çok küçük olsa bile sinyal üretilmez. Bu ayrım olmadan, bir
        forklift operatörü sürekli "kendi aracına aşırı yakın yaya" olarak
        yanlış pozitif üretirdi (bkz. proximity.mp4 gözlemi).

        Args:
            graph (SceneGraph): Karenin anlık sahne grafiği.
            states (TrackStateMachine): Ölçek bilgisi (`scale_ema`) için nesne durum makinesi.

        Returns:
            List[EventSignal]: Forklift ve insan arasındaki yakınlaşma sinyalleri.
        """
        signals = []
        cfg = self.thresholds.get("proximity", {})
        dangerous_pairs = [set(p) for p in cfg.get("dangerous_pairs", [["forklift", "insan"]])]
        threshold = cfg.get("distance_threshold_pixels", 100)
        distance_ratio = cfg.get("distance_threshold_ratio", 1.0)
        operator_overlap_threshold = cfg.get("operator_overlap_ratio", 0.5)

        for edge in graph.edges:
            if edge.relation != "near":
                continue
            a = graph.nodes.get(edge.source)
            b = graph.nodes.get(edge.target)
            if not a or not b:
                continue
            if {a.class_name, b.class_name} not in dangerous_pairs:
                continue

            # Operatör filtresi: kişinin bbox'ı aracın bbox'ıyla ağır
            # örtüşüyorsa (varsayılan >= %50) bu kişi aracın içinde/üzerinde
            # oturan operatördür, ayrı bir yaya değil — sinyal üretilmez.
            if "insan" in (a.class_name, b.class_name) and "arac" in (a.class_name, b.class_name):
                person_node = a if a.class_name == "insan" else b
                vehicle_node = b if a.class_name == "insan" else a
                overlap_ratio = self._person_vehicle_overlap_ratio(person_node, vehicle_node)
                if overlap_ratio >= operator_overlap_threshold:
                    continue

            # Gerçek mesafe doğrudan düğüm merkezlerinden hesaplanır (weight'ten
            # geri çıkarım YAPILMAZ). `edge.weight`, near kenarını kuran
            # SceneGraph.build_relations()'a o karede geçirilen ÖLÇEKLİ eşiğe
            # (bkz. EventEngine._compute_proximity_graph_threshold) göre
            # normalize edilmiştir; burada sabit `threshold` kullanarak geri
            # hesaplama yapmak, iki farklı taban değeri karıştırıp yanlış bir
            # mesafe üretir (örn. gerçek mesafe 318px iken 98px hesaplanması
            # gibi). Doğrudan hesap bu tutarsızlığı ortadan kaldırır.
            estimated_dist = math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])

            state_a = states.get(a.track_id) if a.track_id is not None else None
            state_b = states.get(b.track_id) if b.track_id is not None else None
            scales = [s.scale_ema for s in (state_a, state_b) if s and s.scale_ema > 0.0]
            scale_limit = (sum(scales) / len(scales)) * distance_ratio if scales else threshold
            effective_limit = max(threshold, scale_limit)

            if estimated_dist <= effective_limit:
                # Güven skoru da gerçek mesafeye göre hesaplanır (edge.weight
                # KULLANILMAZ — yukarıdaki tutarsızlık nedeniyle): mesafe 0'a
                # yaklaştıkça güven 1.0'a, effective_limit'e yaklaştıkça 0'a
                # yakın olur.
                confidence = max(0.0, min(1.0, 1.0 - estimated_dist / effective_limit)) if effective_limit > 0 else 0.0
                signals.append(
                    EventSignal(
                        event_type="dangerous_proximity",
                        timestamp=graph.timestamp,
                        description=f"{a.class_name} ve {b.class_name} arasında tehlikeli yakınlık (~{estimated_dist:.0f} piksel).",
                        confidence=confidence,
                        involved_track_ids=[tid for tid in (a.track_id, b.track_id) if tid is not None],
                        metadata={"estimated_distance_pixels": estimated_dist, "effective_limit_pixels": effective_limit},
                    )
                )
        return signals
