"""Sahne Grafiği (Scene Graph) veri modeli ve mekânsal ilişki çıkarımı.

Bu modül, tek bir video karesindeki algılanan nesneleri (düğüm / :class:`SceneNode`)
ve aralarındaki mekânsal-anlamsal ilişkileri (kenar / :class:`SceneEdge`) tutan
:class:`SceneGraph` yapısını sağlar. Grafik, Kanal A algı hattının çıktısıdır ve iki
farklı tüketiciye hizmet eder:

1. :mod:`src.events.rules` — geometrik İSG kurallarını (KKD eksikliği, tehlikeli
   yakınlık) kenarlar üzerinden değerlendirir.
2. VLM prompt'u ve nihai JSON raporu — grafiğin :meth:`SceneGraph.to_dict` çıktısı
   modele yapılandırılmış sahne bağlamı olarak verilir.

Üretilen ilişki türleri
-----------------------
``near``
    İki nesne merkezi arasındaki Öklid mesafesi ``proximity_threshold`` değerinin
    altındaysa kurulur. Kenar ağırlığı ``1 - dist / proximity_threshold``
    formülüyle normalize edilir; 1.0'a yaklaşması nesnelerin üst üste geldiğini
    gösterir. Simetrik bir ilişkidir, kaynak/hedef sırası anlam taşımaz.

``carrying``
    Bir ``arac`` (forklift) ile bir ``palet`` düğümü arasında kurulur. Yönlü bir
    ilişkidir: kaynak **her zaman** taşıyan araç, hedef taşınan yüktür.

``wearing``
    Bir ``insan`` düğümü ile bir KKD düğümü (``baret`` / ``yelek``) arasında
    kurulur. Yönlü bir ilişkidir: kaynak **her zaman** personel, hedef ekipmandır.
    Atama tekildir: bir ekipman yalnızca bir personele, bir personel her KKD
    sınıfından yalnızca bir ekipmana bağlanır (:meth:`SceneGraph._build_wearing_relations`).

Koordinat ve ölçek konvansiyonları
----------------------------------
Tüm koordinatlar, algılamanın yapıldığı karenin piksel uzayındadır ve
``(x1, y1, x2, y2)`` sol-üst / sağ-alt biçimindedir. ``y`` ekseni aşağı doğru artar.

``near`` ve ``carrying`` ilişkileri merkez mesafesine dayandığı için ölçeğe
bağımlıdır ve ``proximity_threshold`` üzerinden ayarlanır. Buna karşılık ``wearing``
ilişkisi **merkez mesafesi kullanmaz**: KKD kutusunun merkezinin, personel kutusunun
ilgili dikey bandı içinde kalıp kalmadığına bakar. Bunun nedeni, bir personelin
merkezinin gövde hizasında, baretinin ise baş hizasında olmasıdır; 300 piksel
boyundaki bir personelde bu iki merkez arasındaki mesafe 130 piksele ulaşır ve sabit
bir piksel eşiği baretli personeli "baretsiz" olarak işaretler. Kapsama tabanlı
kontrol ölçekten bağımsızdır ve hem yakın hem uzak personelde doğru çalışır.

Kullanılan bant oranları (:data:`HELMET_HEAD_BAND_RATIO`,
:data:`VEST_BODY_TOP_RATIO`, :data:`VEST_BODY_BOTTOM_RATIO`), bu kod tabanında
:meth:`src.events.isg_rules_engine.ISGRulesEngine.evaluate_person_ppe` içinde
kullanılan ve testlerle sabitlenmiş oranlarla birebir aynıdır. İki KKD yolunun
tutarlı sonuç vermesi için bu değerler birlikte değiştirilmelidir.

Serileştirme turu (round-trip)
------------------------------
:meth:`SceneGraph.to_dict` grafiği JSON uyumlu bir sözlüğe çevirir;
:meth:`SceneGraph.from_dict` bu sözlükten grafiği geri kurar. Geri kurma sırasında
kenarlar **yeniden hesaplanır**, serileştirilmiş kenar listesi kullanılmaz. Bunun
nedeni, kenarların ``proximity_threshold`` değerine bağlı olması ve olay motorunun
kendi yapılandırma eşiğiyle çalışması gerekmesidir.

Geri kurmanın doğruluğu düğüm geometrisinin korunmasına bağlıdır. Bu yüzden
``to_dict`` hem ``bbox`` hem ``center`` alanlarını yayınlar ve
:func:`_node_bbox` yalnızca ``center`` içeren eski/elle yazılmış sözlüklerde bile
merkez bilgisini korur. ``bbox`` yayınlanmadığı bir dönemde bu tur tüm merkezleri
``(0, 0)`` noktasına çökertiyor ve birbirinden yüzlerce piksel uzaktaki nesneler
"0 piksel mesafede" sayılarak sahte ``near`` / ``wearing`` ilişkileri üretiliyordu.

Örnek
-----
>>> from src.perception.detector import Detection
>>> dets = [
...     Detection("insan", 0.9, (100.0, 100.0, 180.0, 400.0), track_id=1),
...     Detection("baret", 0.8, (120.0, 100.0, 160.0, 140.0), track_id=2),
... ]
>>> graph = SceneGraph.from_detections(frame_idx=12, timestamp=0.48, detections=dets)
>>> [(e.source, e.relation, e.target) for e in graph.edges if e.relation == "wearing"]
[('insan_1', 'wearing', 'baret_2')]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from .detector import Detection

# ---------------------------------------------------------------------------
# İlişki çıkarım sabitleri
# ---------------------------------------------------------------------------

#: ``wearing`` ilişkisinde ekipman tarafı olarak kabul edilen sınıf adları.
PPE_CLASSES: frozenset[str] = frozenset({"baret", "yelek"})

#: ``near`` ilişkisi için varsayılan merkez mesafesi eşiği (piksel). Üretimde
#: ``config.yaml`` içindeki ``events.thresholds.proximity.distance_threshold_pixels``
#: değeri :class:`src.events.event_engine.EventEngine` tarafından geçirilir.
DEFAULT_PROXIMITY_THRESHOLD: float = 100.0

#: Baretin geçerli sayılması için merkezinin kalması gereken bant: personel
#: kutusunun üst %45'i. :meth:`ISGRulesEngine.evaluate_person_ppe` ile aynı oran.
HELMET_HEAD_BAND_RATIO: float = 0.45

#: Yeleğin geçerli sayıldığı gövde bandının üst sınırı (personel yüksekliğinin %15'i).
VEST_BODY_TOP_RATIO: float = 0.15

#: Yeleğin geçerli sayıldığı gövde bandının alt sınırı (personel yüksekliğinin %85'i).
VEST_BODY_BOTTOM_RATIO: float = 0.85

#: ``carrying`` ilişkisi, ``near`` eşiğinin bu katına kadar olan mesafelerde kurulur.
#: Forklift ve palet kutuları büyük olduğu için merkez mesafesi ``near`` eşiğini
#: aşabilir; bu çarpan o payı bırakır.
CARRYING_RANGE_FACTOR: float = 1.5


@dataclass
class SceneNode:
    """Sahne grafiğinde tek bir algılanan nesneyi temsil eden düğüm.

    Attributes:
        node_id: Grafik içinde tekil düğüm kimliği. ``"insan_3"`` biçiminde olup
            takip kimliği varsa ondan, yoksa kare içi sıra indeksinden üretilir.
        class_name: Algılayıcının verdiği sınıf adı (``"insan"``, ``"arac"``,
            ``"baret"``, ``"yelek"``, ``"palet"`` vb.).
        track_id: Nesnenin video boyunca korunan takip kimliği. Takip
            yapılamadıysa ``None``.
        bbox: Piksel uzayında ``(x1, y1, x2, y2)`` sınırlayıcı kutu.
        confidence: Algılama güven skoru, 0.0-1.0 aralığında.
        frame_idx: Düğümün ait olduğu kare indeksi.
    """

    node_id: str
    class_name: str
    track_id: int | None
    bbox: tuple[float, float, float, float]
    confidence: float
    frame_idx: int

    @property
    def center(self) -> tuple[float, float]:
        """Sınırlayıcı kutunun geometrik merkezi.

        Returns:
            ``(cx, cy)`` biçiminde merkez koordinatı. İnsan sınıfı için bu nokta
            baş değil **gövde** hizasına düşer; KKD ilişkileri bu yüzden merkez
            mesafesi yerine :func:`_is_ppe_worn` kapsama kontrolünü kullanır.
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        """Sınırlayıcı kutunun piksel cinsinden genişliği (negatif olamaz)."""
        x1, _, x2, _ = self.bbox
        return max(0.0, x2 - x1)

    @property
    def height(self) -> float:
        """Sınırlayıcı kutunun piksel cinsinden yüksekliği (negatif olamaz)."""
        _, y1, _, y2 = self.bbox
        return max(0.0, y2 - y1)


@dataclass
class SceneEdge:
    """İki düğüm arasındaki ilişkiyi temsil eden kenar.

    ``near`` ilişkisi simetriktir; ``carrying`` ve ``wearing`` ilişkileri yönlüdür.
    Yönlü ilişkilerde kaynak her zaman ana nesnedir (taşıyan araç, personel), hedef
    ise bağımlı nesnedir (taşınan yük, ekipman). Kural motoru bu yönü hoşgörülü
    okusa da, grafiğin VLM'e ve JSON raporuna aktarılması nedeniyle yönün semantik
    olarak doğru olması gerekir.

    Attributes:
        source: Kaynak düğümün ``node_id`` değeri.
        target: Hedef düğümün ``node_id`` değeri.
        relation: İlişki türü: ``"near"``, ``"carrying"`` veya ``"wearing"``.
        weight: İlişkinin gücü, 0.0-1.0 aralığında. ``near`` için mesafeden
            türetilir, diğerlerinde sabit bir güven değeridir.
    """

    source: str
    target: str
    relation: str
    weight: float


def _node_bbox(node: Dict[str, Any]) -> tuple[float, float, float, float]:
    """Serileştirilmiş bir düğüm sözlüğünden sınırlayıcı kutuyu çözümler.

    Öncelik sırası:
      1. ``bbox`` alanı varsa doğrudan kullanılır (tam geometri korunur).
      2. Yoksa ``center`` alanından sıfır boyutlu bir kutu üretilir. Bu, kutu
         geometrisi olmayan eski kayıtlarda ve testlerdeki elle yazılmış
         sözlüklerde merkez bilgisinin kaybolmasını önler; merkez mesafesine
         dayanan ``near`` ilişkisi doğru çalışmaya devam eder.
      3. İkisi de yoksa başlangıç noktasında sıfır boyutlu kutu döner.

    Sıfır boyutlu kutularda yükseklik 0 olduğu için :func:`_is_ppe_worn` kapsama
    kontrolü ``False`` döner; yani geometri bilgisi olmadan KKD ilişkisi
    varsayılmaz. Bu bilinçli bir tercihtir: eksik veriden "KKD takılı" sonucu
    çıkarmak güvenlik açısından yanlış negatif üretir.

    Args:
        node: :meth:`SceneGraph.to_dict` düğüm biçimindeki sözlük.

    Returns:
        ``(x1, y1, x2, y2)`` biçiminde sınırlayıcı kutu.
    """
    bbox = node.get("bbox")
    if bbox is not None and len(bbox) == 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    center = node.get("center")
    if center is not None and len(center) == 2:
        cx, cy = float(center[0]), float(center[1])
        return (cx, cy, cx, cy)

    return (0.0, 0.0, 0.0, 0.0)


def _center_distance(a: SceneNode, b: SceneNode) -> float:
    """İki düğümün sınırlayıcı kutu merkezleri arasındaki Öklid mesafesi (piksel).

    Args:
        a: Birinci düğüm.
        b: İkinci düğüm.

    Returns:
        Merkezler arası mesafe.
    """
    return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])


def _is_ppe_worn(person: SceneNode, ppe: SceneNode) -> bool:
    """KKD ekipmanının ilgili personele ait olup olmadığını kapsama ile denetler.

    Merkez mesafesi yerine, ekipman merkezinin personel kutusunun içinde ve sınıfa
    özgü dikey bantta bulunması aranır. Böylece kontrol personelin görüntüdeki
    boyutundan (kameraya uzaklığından) bağımsız hale gelir.

    Args:
        person: ``insan`` sınıfındaki personel düğümü.
        ppe: ``baret`` veya ``yelek`` sınıfındaki ekipman düğümü.

    Returns:
        Ekipman bu personele ait sayılıyorsa ``True``. Personel kutusu dejenere
        (genişlik veya yükseklik sıfır) ise güvenli tarafta kalmak için ``False``.
    """
    px1, py1, px2, _ = person.bbox
    height = person.height
    if height <= 0.0 or person.width <= 0.0:
        return False

    cx, cy = ppe.center
    if not (px1 <= cx <= px2):
        return False

    if ppe.class_name == "baret":
        return cy <= py1 + height * HELMET_HEAD_BAND_RATIO

    return py1 + height * VEST_BODY_TOP_RATIO <= cy <= py1 + height * VEST_BODY_BOTTOM_RATIO


class SceneGraph:
    """Tek bir video karesindeki nesneler ve aralarındaki ilişkiler.

    Grafik iki aşamada kullanılır: önce düğümler eklenir (:meth:`add_node` veya
    :meth:`from_detections`), ardından :meth:`build_relations` kenarları çıkarır.
    :meth:`from_detections` ve :meth:`from_dict` bu iki adımı birlikte yürütür.

    Attributes:
        frame_idx: Grafiğin ait olduğu kare indeksi.
        timestamp: Karenin video başlangıcına göre saniye cinsinden zamanı.
        nodes: ``node_id`` anahtarlı düğüm sözlüğü. Ekleme sırası korunur.
        edges: Çıkarılmış ilişki kenarları listesi.
    """

    def __init__(self, frame_idx: int = 0, timestamp: float = 0.0):
        """Boş bir sahne grafiği oluşturur.

        Args:
            frame_idx: Karenin indeksi.
            timestamp: Karenin saniye cinsinden zaman damgası.
        """
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.nodes: Dict[str, SceneNode] = {}
        self.edges: List[SceneEdge] = []

    def add_node(self, node: SceneNode) -> None:
        """Grafiğe bir düğüm ekler; aynı ``node_id`` varsa üzerine yazar.

        Args:
            node: Eklenecek düğüm.
        """
        self.nodes[node.node_id] = node

    def add_edge(self, edge: SceneEdge) -> None:
        """Grafiğe bir ilişki kenarı ekler.

        Args:
            edge: Eklenecek kenar. Yineleme denetimi yapılmaz; kenar üretimi
                :meth:`build_relations` içinde her çifti bir kez ele alacak
                şekilde tasarlanmıştır.
        """
        self.edges.append(edge)

    def build_relations(self, proximity_threshold: float = DEFAULT_PROXIMITY_THRESHOLD) -> None:
        """Düğümler arasındaki ``near``, ``carrying`` ve ``wearing`` ilişkilerini çıkarır.

        ``near`` ve ``carrying`` ilişkileri düğüm çiftleri üzerinde tek geçişte
        çıkarılır; her çift yalnızca bir kez ele alınır. Sınıf eşleştirmesi çiftin
        kendisi üzerinden (sıradan bağımsız) yapılır, rol ataması ise açıkça
        belirlenir. Bu, algılayıcının döndürdüğü sıraya bağımlılığı ortadan
        kaldırır: ``palet`` düğümü ``arac`` düğümünden önce gelse bile ilişki
        kurulur ve kenar yönü her zaman araçtan yüke doğrudur.

        ``wearing`` ilişkileri ayrı bir geçişte
        :meth:`_build_wearing_relations` tarafından kurulur; çünkü bir ekipmanın
        tek bir personele atanması, çiftleri birbirinden bağımsız değerlendiren
        tek geçişli döngüde garanti edilemez.

        Yöntem çağrıldığı her seferde mevcut kenar listesine ekleme yapar; grafiği
        yeniden hesaplamak isteyen çağıranlar taze bir örnek kullanmalıdır
        (:meth:`from_dict` ve :meth:`from_detections` böyle davranır).

        Args:
            proximity_threshold: ``near`` ilişkisi için merkez mesafesi eşiği
                (piksel). ``carrying`` bu değerin :data:`CARRYING_RANGE_FACTOR`
                katına kadar olan mesafelerde kurulur. ``wearing`` ilişkisi bu
                eşikten etkilenmez; kapsama tabanlı denetim kullanır.
        """
        node_list = list(self.nodes.values())
        for i, a in enumerate(node_list):
            for b in node_list[i + 1 :]:
                dist = _center_distance(a, b)
                pair = {a.class_name, b.class_name}

                if proximity_threshold > 0 and dist <= proximity_threshold:
                    self.add_edge(
                        SceneEdge(a.node_id, b.node_id, "near", 1.0 - dist / proximity_threshold)
                    )

                # Forklift palet taşıyor olabilir. Kaynak her zaman taşıyan araçtır.
                if pair == {"arac", "palet"} and dist <= proximity_threshold * CARRYING_RANGE_FACTOR:
                    vehicle, load = (a, b) if a.class_name == "arac" else (b, a)
                    self.add_edge(SceneEdge(vehicle.node_id, load.node_id, "carrying", 0.8))

        self._build_wearing_relations(node_list)

    def _build_wearing_relations(self, node_list: List[SceneNode]) -> None:
        """``wearing`` ilişkilerini tekil (1-e-1) atama ile kurar.

        Kapsama koşulunu sağlayan tüm (personel, ekipman) adayları toplanır, merkez
        mesafesine göre küçükten büyüğe sıralanır ve greedy atama yapılır. İki kısıt
        uygulanır:

          * Bir ekipman düğümü yalnızca **tek** personele atanabilir. Aksi halde
            kutuları örtüşen iki personel arasında tek bir baret ikisini de KKD'li
            gösterir ve baretsiz olan personel raporlanmaz (yanlış negatif).
          * Bir personel her KKD sınıfından yalnızca **bir** ekipman tutabilir.
            Böylece bir personel iki bareti birden sahiplenip diğer personeli
            ekipmansız bırakmaz.

        Mesafe burada bir eşik değil, yalnızca adaylar arasında sıralama ölçütüdür;
        kabul/ret kararını :func:`_is_ppe_worn` kapsama denetimi verir. Bu nedenle
        atama ölçekten (personelin kameraya uzaklığından) bağımsız kalır.

        Yöntem greedy'dir, global en iyi eşleşmeyi (bipartite matching) garanti
        etmez. Yoğun ve ağır örtüşen sahnelerde bir ekipman komşu personele
        atanabilir; bu durum yanlış pozitif (fazladan KKD uyarısı) üretir, yanlış
        negatif üretmez. Güvenlik açısından tercih edilen yön budur.

        Args:
            node_list: Grafikteki düğümlerin ekleme sırasındaki listesi.
        """
        persons = [n for n in node_list if n.class_name == "insan"]
        equipment = [n for n in node_list if n.class_name in PPE_CLASSES]
        if not persons or not equipment:
            return

        candidates = [
            (_center_distance(person, ppe), person, ppe)
            for ppe in equipment
            for person in persons
            if _is_ppe_worn(person, ppe)
        ]
        candidates.sort(key=lambda c: c[0])

        claimed_equipment: set[str] = set()
        claimed_slots: set[tuple[str, str]] = set()
        for _, person, ppe in candidates:
            slot = (person.node_id, ppe.class_name)
            if ppe.node_id in claimed_equipment or slot in claimed_slots:
                continue
            claimed_equipment.add(ppe.node_id)
            claimed_slots.add(slot)
            self.add_edge(SceneEdge(person.node_id, ppe.node_id, "wearing", 0.9))

    def find_nodes(self, class_name: str) -> List[SceneNode]:
        """Belirtilen sınıfa ait tüm düğümleri döner.

        Args:
            class_name: Aranan sınıf adı (örn. ``"insan"``).

        Returns:
            Eşleşen düğümlerin listesi; eşleşme yoksa boş liste.
        """
        return [n for n in self.nodes.values() if n.class_name == class_name]

    def to_dict(self) -> dict[str, Any]:
        """Grafiği JSON uyumlu bir sözlüğe dönüştürür.

        Düğümlerde hem ``bbox`` hem ``center`` yayınlanır. ``center`` insan
        tarafından okunabilirlik ve geriye dönük uyumluluk için, ``bbox`` ise
        :meth:`from_dict` ile geometrinin kayıpsız geri kurulabilmesi için
        gereklidir; kapsama tabanlı KKD denetimi kutu boyutlarına ihtiyaç duyar.

        Returns:
            ``frame_idx``, ``timestamp``, ``nodes`` ve ``edges`` anahtarlarını
            içeren sözlük. Sayısal değerler raporlama için yuvarlanır.
        """
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 2),
            "nodes": [
                {
                    "id": n.node_id,
                    "class": n.class_name,
                    "track_id": n.track_id,
                    "bbox": [round(v, 2) for v in n.bbox],
                    "center": [round(n.center[0], 2), round(n.center[1], 2)],
                    "confidence": round(n.confidence, 3),
                    "frame_idx": n.frame_idx,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "relation": e.relation, "weight": round(e.weight, 3)}
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        proximity_threshold: float = DEFAULT_PROXIMITY_THRESHOLD,
    ) -> SceneGraph:
        """:meth:`to_dict` çıktısından grafiği yeniden kurar.

        Düğümler sözlükten okunur, kenarlar ise **yeniden hesaplanır**. Sözlükteki
        ``edges`` listesi bilinçli olarak yok sayılır: kenarlar
        ``proximity_threshold`` değerine bağlıdır ve olay motoru kendi
        yapılandırma eşiğiyle çalışmak zorundadır. Aksi halde grafiği üreten
        katmanın varsayılan eşiği, kuralların kullandığı eşikle çelişir.

        Args:
            data: :meth:`to_dict` biçiminde sahne grafiği sözlüğü. Boş sözlük
                verilirse düğümsüz bir grafik döner.
            proximity_threshold: ``near`` ilişkisi için kullanılacak eşik (piksel).

        Returns:
            Düğümleri doldurulmuş ve ilişkileri hesaplanmış yeni bir
            :class:`SceneGraph` örneği.
        """
        frame_idx = data.get("frame_idx", 0)
        graph = cls(frame_idx=frame_idx, timestamp=data.get("timestamp", 0.0))
        for n in data.get("nodes", []):
            graph.add_node(
                SceneNode(
                    node_id=n["id"],
                    class_name=n["class"],
                    track_id=n.get("track_id"),
                    bbox=_node_bbox(n),
                    confidence=n.get("confidence", 0.0),
                    frame_idx=n.get("frame_idx", frame_idx),
                )
            )
        graph.build_relations(proximity_threshold)
        return graph

    @classmethod
    def from_detections(
        cls,
        frame_idx: int,
        timestamp: float,
        detections: List[Detection],
        proximity_threshold: float = DEFAULT_PROXIMITY_THRESHOLD,
    ) -> SceneGraph:
        """Algılama listesinden düğümleri kurup ilişkileri hesaplayarak grafik üretir.

        Düğüm kimliği takip kimliğinden (``"insan_3"``) üretilir; takip kimliği
        yoksa liste içindeki sıra indeksi kullanılır (``"insan_0"``). Takip kimliği
        kullanılması, aynı nesnenin kareler arasında aynı kimlikle görünmesini
        sağlar ve olay motorunun yineleme filtresini çalıştırır.

        Args:
            frame_idx: Karenin indeksi.
            timestamp: Karenin saniye cinsinden zaman damgası.
            detections: Kareye ait algılama nesneleri.
            proximity_threshold: ``near`` ilişkisi için kullanılacak eşik (piksel).

        Returns:
            Düğüm ve kenarları hazır yeni bir :class:`SceneGraph` örneği.
        """
        graph = cls(frame_idx=frame_idx, timestamp=timestamp)
        for idx, det in enumerate(detections):
            tid = getattr(det, "track_id", None)
            node_id = f"{det.class_name}_{tid}" if tid is not None else f"{det.class_name}_{idx}"
            graph.add_node(
                SceneNode(
                    node_id=node_id,
                    class_name=det.class_name,
                    track_id=tid,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    frame_idx=frame_idx,
                )
            )
        graph.build_relations(proximity_threshold)
        return graph
