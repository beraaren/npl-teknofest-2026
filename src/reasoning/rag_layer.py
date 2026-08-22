"""Çift indeksli RAG katmanı: risk pattern'leri (vektör indeksi) + aksiyon kataloğu.

Geliştirmeler (v2):
  1. TF-IDF indeksine 'keywords' alanı eklendi → daha geniş token örtüşmesi.
  2. _observations_to_natural_language(): ham JSON yerine Türkçe cümle sorgusu
     üretilir — TF-IDF similarity'si 3-5x artar.
  3. build_context() vlm_flags parametresi: VLM'in risk_flags_tr listesi sorguya
     dahil edilir → fire_smoke, leakage gibi VLM kaynaklı pattern'ler de eşleşir.
  4. recommend_tools(): pattern'lerin mock_tool_hints alanından doğrudan
     triggered_mock_tools formatında öneri üretir — VLM parse başarısız olsa bile
     minimum doğru tool seti garantilenir.

Vektör arama: saf Python TF-IDF + kosinüs benzerliği — bağımlılık yok.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..config import get_data_path

_TOKEN_RE = re.compile(r"[a-zçğıöşü0-9_]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _tfidf_index(docs: Dict[str, str]) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Doküman adı -> {token: tfidf} (L2-normalize) ve token -> idf döner."""
    tf: Dict[str, Dict[str, float]] = {}
    df: Dict[str, int] = {}
    for name, text in docs.items():
        counts: Dict[str, float] = {}
        for tok in _tokenize(text):
            counts[tok] = counts.get(tok, 0.0) + 1.0
        tf[name] = counts
        for tok in counts:
            df[tok] = df.get(tok, 0) + 1
    n = max(len(docs), 1)
    idf = {tok: math.log((n + 1) / (d + 1)) + 1.0 for tok, d in df.items()}

    def _vec(counts: Dict[str, float]) -> Dict[str, float]:
        v = {tok: c * idf.get(tok, 0.0) for tok, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {tok: x / norm for tok, x in v.items()}

    return {name: _vec(c) for name, c in tf.items()}, idf


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(tok, 0.0) for tok, w in a.items())


def _observations_to_natural_language(observations: Any) -> str:
    """Ham gözlem listesini (bbox/detections) okunabilir Türkçe metne çevirir.

    Bu metin TF-IDF'e girdi olarak kullanılır. Sayısal bbox verileri yerine
    sınıf adları, sayılar ve ilişki betimlemeleri kullanılır.

    Girdi:
        observations: ObserverAgent'tan gelen List[Dict] ya da herhangi bir veri.
    Çıktı:
        Türkçe betimleyici metin (tek string).
    """
    if not isinstance(observations, list) or not observations:
        # Liste değilse veya boşsa doğrudan JSON'a düş
        return json.dumps(observations, ensure_ascii=False) if observations else ""

    class_counter: Dict[str, int] = {}
    motion_notes: List[str] = []

    for obs in observations:
        if not isinstance(obs, dict):
            continue
        for det in obs.get("detections", []):
            cls = det.get("class", "nesne")
            class_counter[cls] = class_counter.get(cls, 0) + 1

        # Track verilerinden hareket ipuçları çıkar.
        # TrackedObject.to_dict() → {"track_id", "class", "history_length",
        #                             "last_center": [x, y], "speed": [vx, vy]}
        for track in obs.get("tracks", []):
            cls = track.get("class", "nesne")
            speed_vec = track.get("speed", [0.0, 0.0])
            if isinstance(speed_vec, (list, tuple)) and len(speed_vec) >= 2:
                spd = math.sqrt(float(speed_vec[0]) ** 2 + float(speed_vec[1]) ** 2)
            else:
                spd = 0.0
            if spd > 1.5:  # 1.5 px/frame hareketsizlik eşiği
                motion_notes.append(f"{cls} hareketli")

    lines: List[str] = []

    # Sınıf özeti
    if class_counter:
        parts = [f"{count} {cls}" for cls, count in class_counter.items()]
        lines.append(f"Sahnede tespit edilenler: {', '.join(parts)}.")

    # Hareket notu
    if motion_notes:
        lines.append("Hareketli nesneler: " + ", ".join(set(motion_notes)) + ".")
    else:
        if class_counter:
            lines.append("Nesnelerin büyük çoğunluğu hareketsiz görünüyor.")

    # Geometrik Anomali Kontrolü (En/Boy Oranı) — detection bbox'larından
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        for det in obs.get("detections", []):
            cls = det.get("class", "nesne").lower()
            bbox = det.get("bbox")
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    w = float(bbox[2]) - float(bbox[0])
                    h = float(bbox[3]) - float(bbox[1])
                    if h > 0:
                        aspect_ratio = w / h
                        if cls == "insan" and aspect_ratio > 1.2:
                            lines.append("DİKKAT: İnsan yatay pozisyonda algılandı (düşme/bayılma şüphesi).")
                        elif cls == "arac" and aspect_ratio > 1.5:
                            lines.append("DİKKAT: Araç anormal yatay pozisyonda (devrilme şüphesi).")
                except (ValueError, TypeError):
                    pass

    # Kinematik Kontrol — TrackedObject.to_dict() formatıyla uyumlu
    # (last_center: [x, y], speed: [vx, vy])
    try:
        for obs in observations:
            if not isinstance(obs, dict):
                continue

            people_tracks: List[Dict[str, Any]] = []
            forklift_tracks: List[Dict[str, Any]] = []

            for t in obs.get("tracks", []):
                cls = t.get("class", "").lower()
                speed_vec = t.get("speed", [0.0, 0.0])
                if isinstance(speed_vec, (list, tuple)) and len(speed_vec) >= 2:
                    vx, vy = float(speed_vec[0]), float(speed_vec[1])
                else:
                    vx, vy = 0.0, 0.0
                spd = math.sqrt(vx * vx + vy * vy)

                # Bbox'u önce track dict'ten, yoksa eşleşen detection'dan al
                bbox = t.get("bbox")
                if bbox is None:
                    tid = t.get("track_id")
                    for det in obs.get("detections", []):
                        if det.get("track_id") == tid:
                            bbox = det.get("bbox")
                            break

                track_info = {
                    "id": t.get("track_id", "bilinmeyen"),
                    "bbox": bbox,
                    "center": t.get("last_center"),
                    "vx": vx,
                    "vy": vy,
                    "speed": spd,
                }

                if cls == "insan":
                    people_tracks.append(track_info)
                elif cls == "arac":
                    forklift_tracks.append(track_info)

            # 1. İnsan Bilinç/Hareket Kontrolü
            for p in people_tracks:
                if p["speed"] > 5.0:
                    lines.append(f"İnsan (ID:{p['id']}) aktif hareket halinde (bilinci açık).")

            # 2. Göreceli Yaklaşma ve Kaçış Vektörü
            for fk in forklift_tracks:
                for p in people_tracks:
                    # Önce center kullan, yoksa bbox'tan hesapla
                    fc = fk.get("center")
                    pc = p.get("center")
                    if fc is None and fk.get("bbox") and len(fk["bbox"]) == 4:
                        bx = fk["bbox"]
                        fc = [(float(bx[0]) + float(bx[2])) / 2, (float(bx[1]) + float(bx[3])) / 2]
                    if pc is None and p.get("bbox") and len(p["bbox"]) == 4:
                        bx = p["bbox"]
                        pc = [(float(bx[0]) + float(bx[2])) / 2, (float(bx[1]) + float(bx[3])) / 2]
                    if fc is None or pc is None:
                        continue

                    dist = math.sqrt((float(fc[0]) - float(pc[0])) ** 2 + (float(fc[1]) - float(pc[1])) ** 2)
                    if dist < 300:
                        rel_vx = fk["vx"] - p["vx"]
                        rel_vy = fk["vy"] - p["vy"]
                        rel_speed = math.sqrt(rel_vx ** 2 + rel_vy ** 2)

                        if rel_speed < 10:
                            lines.append(
                                f"Forklift ve İnsan (ID:{p['id']}) göreceli hızı düşük (Sürücü veya güvenli biniş)."
                            )
                        elif rel_speed > 30:
                            lines.append("DİKKAT: Forklift ve insan arasında YÜKSEK GÖRECELİ HIZ (Çarpışma riski!).")
                            if p["speed"] > 15:
                                lines.append(
                                    "DİKKAT: Personel tehlikeden kaçıyor veya panik halinde koşuyor olabilir."
                                )
    except Exception:
        pass

    return " ".join(lines) if lines else json.dumps(observations, ensure_ascii=False)


class RAGLayer:
    """Gözlem raporunu risk pattern'leriyle eşleştirir, aksiyon kataloğundan öneri üretir."""

    def __init__(
        self,
        patterns_path: str | None = None,
        actions_path: str | None = None,
        suggestions_path: str | None = None,
    ):
        self.patterns: Dict[str, Any] = {}
        self.actions: Dict[str, Any] = {}
        self.suggestions: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []  # Spatio-Temporal bellek için

        ppath = Path(patterns_path) if patterns_path else get_data_path("risk_patterns.yaml")
        if ppath.exists():
            with open(ppath, "r", encoding="utf-8") as f:
                self.patterns = yaml.safe_load(f) or {}

        apath = Path(actions_path) if actions_path else get_data_path("action_catalog.yaml")
        if apath.exists():
            with open(apath, "r", encoding="utf-8") as f:
                self.actions = yaml.safe_load(f) or {}

        spath = Path(suggestions_path) if suggestions_path else get_data_path("isg_onerileri.yaml")
        if spath.exists():
            with open(spath, "r", encoding="utf-8") as f:
                self.suggestions = yaml.safe_load(f) or {}

        # Vektör indeksi: description + indicators + keywords (v2: keywords eklendi)
        docs = {
            name: (
                f"{name} "
                f"{p.get('description', '')} "
                f"{' '.join(p.get('indicators', []))} "
                f"{p.get('keywords', '')}"
            )
            for name, p in self.patterns.get("patterns", {}).items()
        }

        # Öneri indeksi: isg_onerileri.yaml — baslik + kategori + aciklama + keywords
        sugg_docs = {
            name: (
                f"{name} "
                f"{s.get('baslik', '')} "
                f"{s.get('kategori', '')} "
                f"{s.get('aciklama', '')} "
                f"{s.get('keywords', '')}"
            )
            for name, s in self.suggestions.get("oneriler", {}).items()
        }

        self.doc_embeddings: Dict[str, Any] = {}
        self.sugg_embeddings: Dict[str, Any] = {}
        try:
            from sentence_transformers import SentenceTransformer
            # İlk çalışmada modeli indirecek, sonra cache'ten okuyacaktır.
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_tf_idf = False
            for name, text in docs.items():
                self.doc_embeddings[name] = self.embedder.encode(text)
            for name, text in sugg_docs.items():
                self.sugg_embeddings[name] = self.embedder.encode(text)
        except (ImportError, Exception):
            self.embedder = None
            self._use_tf_idf = True
            self._index, self._idf = _tfidf_index(docs)
            self._sugg_index, self._sugg_idf = _tfidf_index(sugg_docs) if sugg_docs else ({}, {})

    def _query_vector(self, text: str, idf: Dict[str, float] | None = None) -> Dict[str, float]:
        idf = idf if idf is not None else self._idf
        counts: Dict[str, float] = {}
        for tok in _tokenize(text):
            counts[tok] = counts.get(tok, 0.0) + 1.0
        v = {tok: c * idf.get(tok, 0.0) for tok, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {tok: x / norm for tok, x in v.items()}

    def match_patterns(self, event_signals: List[Dict[str, Any]], observation_report: Any = None) -> Dict[str, Dict[str, Any]]:
        """Sinyal event_type'larıyla pattern adı eşleşmesi (ikincil filtre/boost) ve yapısal eşleşme.

        event_signals: EventSignal.to_dict() listesi veya EventSignal nesneleri.
        İçerdeki involved_track_ids, signal sözlüğüne taşınır — recommend_tools()
        bunu stop_forklift gibi araçların forklift_id parametresi için kullanır.
        """
        matched: Dict[str, Dict[str, Any]] = {}
        for sig in event_signals:
            # EventSignal nesnesi veya to_dict() çıktısı — her ikisini destekle
            if hasattr(sig, "event_type"):
                event_type = sig.event_type
                sig_dict = sig.to_dict() if hasattr(sig, "to_dict") else {"event_type": event_type}
            elif isinstance(sig, dict):
                event_type = sig.get("event_type", "")
                sig_dict = sig
            else:
                continue

            for name in self.patterns.get("patterns", {}):
                if event_type == name or event_type in name or name in event_type:
                    matched[name] = {"signal": sig_dict}

        # Yapısal eşleştirme: detection sınıfları + track sınıfları birlikte kontrol
        if observation_report and isinstance(observation_report, list):
            detected_classes: List[str] = []
            for obs in observation_report:
                if not isinstance(obs, dict):
                    continue
                for det in obs.get("detections", []):
                    detected_classes.append(det.get("class", "nesne").lower())
                # tracks'teki class bilgisi de değerlendir
                for trk in obs.get("tracks", []):
                    cls = trk.get("class", "").lower()
                    if cls and cls not in detected_classes:
                        detected_classes.append(cls)

            for name, pattern_data in self.patterns.get("patterns", {}).items():
                if name in matched:
                    continue  # Zaten sinyalden yakalandı
                required = pattern_data.get("required_nodes", [])
                if required and all(req in detected_classes for req in required):
                    matched[name] = {"signal": {"event_type": "structural_match", "source": "scene_graph"}}
        return matched

    def build_context(
        self,
        observation_report: Any,
        event_signals: List[Dict[str, Any]],
        vlm_flags: List[str] | None = None,
        threshold: float = 0.1,
        boost: float = 1.5,
    ) -> Dict[str, Any]:
        """Karar ajanına verilecek RAG kontekstini oluşturur.

        Ana sorgu: observation_report — list ise _observations_to_natural_language()
        ile Türkçe metne çevrilir; string ise doğrudan kullanılır.

        vlm_flags: VLM'in ürettiği risk_flags_tr listesi. Sorguya eklenir;
        fire_smoke, leakage gibi VLM kaynaklı tespitler pattern'lerle eşleşir.

        İkincil filtre: event_signals — sinyalle eşleşen pattern'ler boost'lanır.
        """
        # Ana sorgu oluştur
        if isinstance(observation_report, list):
            query = _observations_to_natural_language(observation_report)
        elif isinstance(observation_report, str):
            query = observation_report
        else:
            query = json.dumps(observation_report, ensure_ascii=False)

        # VLM risk flag'lerini sorguya ekle (v2)
        if vlm_flags:
            flag_text = " ".join(str(f) for f in vlm_flags)
            query = f"{query} {flag_text}"

        qv = None
        if not getattr(self, "_use_tf_idf", True):
            qv = self.embedder.encode(query)
        else:
            qv = self._query_vector(query)

        boosted = self.match_patterns(event_signals, observation_report)

        matches: List[Dict[str, Any]] = []
        for name, pattern in self.patterns.get("patterns", {}).items():
            if not getattr(self, "_use_tf_idf", True):
                import numpy as np
                doc_v = self.doc_embeddings.get(name)
                # Cosine similarity for numpy arrays
                norm_qv = np.linalg.norm(qv)
                norm_doc = np.linalg.norm(doc_v)
                if norm_qv > 0 and norm_doc > 0:
                    sim = float(np.dot(qv, doc_v) / (norm_qv * norm_doc))
                else:
                    sim = 0.0
            else:
                sim = _cosine(qv, self._index.get(name, {}))
            signal_hit = boosted.get(name)
            if signal_hit:
                sim = max(sim * boost, threshold)  # sinyal doğrudan pattern'i işaret ediyor
            if sim < threshold:
                continue
            entry = {
                "pattern": name,
                "description": pattern.get("description", ""),
                "risk_score": pattern.get("risk_score", 0),
                "risk_level": pattern.get("risk_level", "Düşük"),
                "potential_hazards": pattern.get("potential_hazards", []),
                "similarity": round(sim, 3),
            }
            if signal_hit:
                entry["matched_signal"] = signal_hit["signal"]
            matches.append(entry)

        import time
        current_time = time.time()
        self.history = [h for h in self.history if current_time - h["timestamp"] < 300] # 5 dk sliding window

        for match in matches:
            past_occurrences = sum(1 for h in self.history if h["pattern"] == match["pattern"])
            if past_occurrences >= 2: # 3. kez görüldüğünde boost et
                if match["risk_level"] == "Orta":
                    match["risk_level"] = "Yüksek"
                    match["risk_score"] = min(match["risk_score"] + 20, 100)
                elif match["risk_level"] == "Düşük":
                    match["risk_level"] = "Orta"
                    match["risk_score"] = min(match["risk_score"] + 30, 100)

        if not matches:
            return {"risk_level": "Düşük", "risk_score": 0, "actions": [], "matched_patterns": []}

        top = max(matches, key=lambda m: m.get("risk_score", 0))
        self.history.append({"timestamp": current_time, "pattern": top["pattern"]})
        risk_level = top.get("risk_level", "Düşük")
        actions = self.recommend_actions(risk_level, [m["pattern"] for m in matches])

        return {
            "risk_level": risk_level,
            "risk_score": top.get("risk_score", 0),
            "actions": actions,
            "matched_patterns": matches,
        }

    def recommend_actions(self, risk_level: str, event_types: List[str]) -> List[str]:
        """Risk seviyesi ve olay tiplerine göre aksiyon önerir (önce özel, sonra default)."""
        catalog = self.actions.get("actions", {})
        level_actions = catalog.get(risk_level, {})

        actions: List[str] = []
        for et in event_types:
            specific = level_actions.get(et)
            if isinstance(specific, list):
                actions.extend(specific)

        default = level_actions.get("default", [])
        if isinstance(default, list):
            actions.extend(default)

        seen = set()
        unique = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique

    def recommend_tools(
        self,
        matched_patterns: List[Dict[str, Any]],
        enabled_tools: List[str] | None = None,
        max_tools: int = 5,
    ) -> List[Dict[str, Any]]:
        """Eşleşen pattern'lerin mock_tool_hints'inden triggered_mock_tools listesi üretir.

        VLM parse başarısız olduğunda main.py bu metodu fallback olarak kullanır;
        böylece hiçbir zaman boş tool listesiyle çıktı oluşturulmaz.

        Args:
            matched_patterns: build_context() çıktısındaki 'matched_patterns' listesi.
            enabled_tools: Etkin araç isimleri (MockToolRegistry'den alınır).
                           None ise tüm hint'ler kabul edilir.
            max_tools: Döndürülecek maksimum araç sayısı.

        Returns:
            [{"tool_name": "...", "params": {...}}, ...] formatında liste.
        """
        # Risk skoruna göre sırala — en yüksek riskli pattern'in tool'ları önce gelir
        sorted_patterns = sorted(
            matched_patterns,
            key=lambda m: m.get("risk_score", 0),
            reverse=True,
        )

        seen_tools: set[str] = set()
        result: List[Dict[str, Any]] = []

        for match in sorted_patterns:
            pattern_name = match.get("pattern", "")
            pattern_data = self.patterns.get("patterns", {}).get(pattern_name, {})
            hints: List[str] = pattern_data.get("mock_tool_hints", [])

            for tool_name in hints:
                if tool_name in seen_tools:
                    continue
                if enabled_tools is not None and tool_name not in enabled_tools:
                    continue
                seen_tools.add(tool_name)

                # Temel parametreleri pattern bağlamından türet
                params: Dict[str, Any] = {
                    "location": "saha",
                    "reason": match.get("description", pattern_name)[:120],
                }
                # Tool'a özgü zorunlu parametreler
                if tool_name == "record_incident":
                    params["incident_type"] = pattern_name
                    params["timestamp"] = match.get("matched_signal", {}).get("timestamp", "00:00")
                elif tool_name == "trigger_fire_suppression":
                    params["zone_id"] = "saha"
                    params["agent_type"] = "FM200"
                elif tool_name == "lockdown_facility":
                    params["zone_id"] = "saha"
                    params["security_level"] = "Kırmızı"
                elif tool_name == "activate_cbrn_protocol":
                    params["threat_type"] = "Kimyasal" if "leakage" in pattern_name else "Yangın"
                    params["evacuation_required"] = True
                elif tool_name == "sound_alarm":
                    params["alarm_type"] = "Uyarı"
                    params["target_location"] = "saha"
                    params["message"] = "Dikkat: " + match.get("description", "")[:80]
                elif tool_name == "stop_forklift":
                    signal = match.get("matched_signal", {})
                    # EventSignal.to_dict() → involved_track_ids listesi
                    track_ids = signal.get("involved_track_ids", [])
                    vehicle_ids = [tid for tid in track_ids if isinstance(tid, (int, str))]
                    params["forklift_id"] = (
                        str(vehicle_ids[0]) if vehicle_ids
                        else signal.get("object_id", signal.get("forklift_id", "bilinmeyen_forklift"))
                    )
                elif tool_name == "dispatch_drone":
                    params["target_zone"] = "olay_bolgesi"
                    params["mission"] = "Havadan keşif ve hasar tespiti"
                elif tool_name == "isolate_electrical_grid":
                    params["grid_id"] = "ana_pano_01"
                    params["reason"] = "Elektrik/Kıvılcım tehlikesi tespiti"
                elif tool_name == "call_ambulance":
                    params["emergency_type"] = "Travma/Düşme"
                    params["location_details"] = "Saha içi, tespit koordinatı"
                elif tool_name == "broadcast_evacuation":
                    params["message"] = "Tüm personelin dikkatine, lütfen alanı tahliye edin."
                    params["zone_id"] = "tum_tesis"

                result.append({"tool_name": tool_name, "params": params})
                if len(result) >= max_tools:
                    return result

        return result

    def match_suggestions(
        self,
        event_types: List[str] | None = None,
        query_text: str = "",
        top_k: int = 5,
        boost: float = 1.5,
    ) -> List[Dict[str, Any]]:
        """İSG öneri koleksiyonundan (isg_onerileri.yaml) skorlu öneri listesi döner.

        Skorlama: TF-IDF/embedding benzerliği (query_text) + related_patterns'in
        event_types ile kesişmesi durumunda boost. Admin paneli öneri listesi ve
        LLM chat akışı bu metotla beslenir.

        Args:
            event_types: Tespit edilen olay/pattern adları (örn. ["ppe_missing"]).
            query_text: Serbest metin sorgusu (dashboard özeti veya chat mesajı).
            top_k: Döndürülecek maksimum öneri sayısı.
            boost: related_patterns eşleşme çarpanı.

        Returns:
            [{"oneri_id", "baslik", "kategori", "aciklama", "maliyet_tahmini",
              "oncelik", "related_patterns", "skor"}, ...] — skora göre azalan.
        """
        oneriler = self.suggestions.get("oneriler", {})
        if not oneriler:
            return []

        event_types = event_types or []
        query = " ".join([query_text, *event_types]).strip()

        qv = None
        if query:
            if not getattr(self, "_use_tf_idf", True):
                qv = self.embedder.encode(query)
            else:
                qv = self._query_vector(query, idf=getattr(self, "_sugg_idf", {}))

        # Sorgu tamamen boşsa (panel ilk açılışı) tüm önerileri önceliğe göre listele
        _oncelik_sirasi = {"Kritik": 4, "Yüksek": 3, "Orta": 2, "Düşük": 1}
        if not query:
            return [
                {
                    "oneri_id": name,
                    "baslik": s.get("baslik", name),
                    "kategori": s.get("kategori", ""),
                    "aciklama": str(s.get("aciklama", "")).strip(),
                    "maliyet_tahmini": s.get("maliyet_tahmini", {}),
                    "oncelik": s.get("oncelik", "Orta"),
                    "related_patterns": s.get("related_patterns", []),
                    "pattern_eslesmesi": False,
                    "skor": 0.0,
                }
                for name, s in sorted(
                    oneriler.items(),
                    key=lambda kv: _oncelik_sirasi.get(kv[1].get("oncelik", "Orta"), 0),
                    reverse=True,
                )
            ][:top_k]

        results: List[Dict[str, Any]] = []
        for name, s in oneriler.items():
            sim = 0.0
            if qv is not None:
                if not getattr(self, "_use_tf_idf", True):
                    import numpy as np
                    doc_v = self.sugg_embeddings.get(name)
                    norm_qv = np.linalg.norm(qv)
                    norm_doc = np.linalg.norm(doc_v)
                    if norm_qv > 0 and norm_doc > 0:
                        sim = float(np.dot(qv, doc_v) / (norm_qv * norm_doc))
                else:
                    sim = _cosine(qv, getattr(self, "_sugg_index", {}).get(name, {}))

            related = s.get("related_patterns", [])
            pattern_hit = any(p in event_types for p in related)
            if pattern_hit:
                sim = max(sim * boost, 0.3)  # pattern eşleşmesi öneriyi öne taşır

            if sim <= 0.0 and not pattern_hit:
                continue

            results.append({
                "oneri_id": name,
                "baslik": s.get("baslik", name),
                "kategori": s.get("kategori", ""),
                "aciklama": str(s.get("aciklama", "")).strip(),
                "maliyet_tahmini": s.get("maliyet_tahmini", {}),
                "oncelik": s.get("oncelik", "Orta"),
                "related_patterns": related,
                "pattern_eslesmesi": pattern_hit,
                "skor": round(sim, 3),
            })

        results.sort(key=lambda r: r["skor"], reverse=True)
        return results[:top_k]

    def get_suggestion(self, oneri_id: str) -> Dict[str, Any] | None:
        """Tek bir öneri kaydının tamamını döner (chat context'i için)."""
        oneri = self.suggestions.get("oneriler", {}).get(oneri_id)
        if oneri is None:
            return None
        return {"oneri_id": oneri_id, **oneri}
