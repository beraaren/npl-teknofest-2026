import sqlite3
import json
import uuid
from typing import List, Dict, Optional, Any
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "gateway.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            job_id TEXT PRIMARY KEY,
            camera_id TEXT,
            result_json TEXT NOT NULL,
            risk TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # §2.5: job olaylarını debug/demo için kaydet (GET /analyses/{id}/events)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            stream TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Görev atamaları — süpervizörün belirli bir olayı belirli bir ekibe
    # yönlendirmesi. Saha ekranı ARTIK yalnızca kendisine atanan olayları
    # gösterir; eskiden her uyarı tüm rollere gidiyordu (target_roles sabiti).
    #
    # Atama, olayın karar çıktısındaki alanlarını (ajan özeti, risk, olay anı,
    # önerilen aksiyonlar) kopyalayarak taşır. Kopyalamanın nedeni: saha ekibi
    # ekranının analiz dosyasına bağımlı olmadan çalışması ve atama anındaki
    # bilginin sonradan değişmemesi (denetim izi).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_slug TEXT NOT NULL,
            camera_id TEXT,
            role TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'atandi',
            risk TEXT,
            headline TEXT,
            summary TEXT,
            reasoning TEXT,
            event_type TEXT,
            event_seconds REAL DEFAULT 0,
            event_timestamp TEXT,
            actions_json TEXT,
            video_file TEXT,
            note TEXT,
            assigned_by TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            acknowledged_at DATETIME,
            resolved_at DATETIME
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_assignments_role_status "
        "ON assignments (role, status)"
    )
    # RLHF / DPO Human-in-the-Loop geri bildirim tablosu
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_slug TEXT NOT NULL,
            camera_id TEXT,
            feedback_type TEXT NOT NULL,
            original_risk TEXT,
            original_summary TEXT,
            original_output_json TEXT NOT NULL,
            corrected_risk TEXT,
            corrected_summary TEXT,
            corrected_actions_json TEXT,
            corrected_output_json TEXT NOT NULL,
            prompt_context_json TEXT,
            supervisor_notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedbacks_slug "
        "ON feedbacks (analysis_slug)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedbacks_type "
        "ON feedbacks (feedback_type)"
    )
    conn.commit()
    conn.close()



def seed_cameras(count: int = 9):
    """Duvardaki kamera sayısı kadar kamera kaydı oluşturur (varsa günceller).

    Kamera sayısı analiz kütüphanesinin boyutuna göre değişebildiği için sabit
    9 yerine parametre alır; fazladan kalan kayıtlar silinir ki arayüzde
    oynatacak videosu olmayan kamera görünmesin.

    Args:
        count: Oluşturulacak kamera sayısı.
    """
    conn = get_connection()
    cursor = conn.cursor()
    wanted = {f"cam-{i:02d}" for i in range(1, count + 1)}

    for cam_id in sorted(wanted):
        cursor.execute(
            "INSERT OR IGNORE INTO cameras (id, name, source, status) VALUES (?, ?, ?, ?)",
            (cam_id, f"Kamera {cam_id.split('-')[1]}", "pseudolive", "aktif"),
        )

    cursor.execute("SELECT id FROM cameras WHERE source = 'pseudolive'")
    for row in cursor.fetchall():
        if row["id"] not in wanted:
            cursor.execute("DELETE FROM cameras WHERE id = ?", (row["id"],))

    conn.commit()
    conn.close()

# Cameras
def create_camera(name: str, source: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    cam_id = str(uuid.uuid4())
    cursor.execute("INSERT INTO cameras (id, name, source, status) VALUES (?, ?, ?, ?)",
                   (cam_id, name, source, "aktif"))
    conn.commit()
    conn.close()
    return {"id": cam_id, "name": name, "source": source, "status": "aktif"}

def get_camera(cam_id: str) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cameras WHERE id = ?", (cam_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_cameras() -> List[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cameras ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_camera_status(cam_id: str, status: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE cameras SET status = ? WHERE id = ?", (status, cam_id))
    conn.commit()
    conn.close()

def delete_camera(cam_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cameras WHERE id = ?", (cam_id,))
    conn.commit()
    conn.close()

# Analyses
def save_analysis(job_id: str, camera_id: str, result_data: dict):
    conn = get_connection()
    cursor = conn.cursor()
    risk = result_data.get("risk", "")
    cursor.execute(
        "INSERT OR REPLACE INTO analyses (job_id, camera_id, result_json, risk) VALUES (?, ?, ?, ?)",
        (job_id, camera_id, json.dumps(result_data, ensure_ascii=False), risk)
    )
    conn.commit()
    conn.close()

def get_analyses(
    camera_id: Optional[str] = None,
    risk: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM analyses WHERE 1=1"
    params: list = []
    if camera_id:
        query += " AND camera_id = ?"
        params.append(camera_id)
    if risk:
        query += " AND risk = ?"
        params.append(risk)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [
        {"job_id": r["job_id"], "camera_id": r["camera_id"], "risk": r["risk"], "created_at": r["created_at"]}
        for r in rows
    ]

def get_analysis_by_id(job_id: str) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row["result_json"])
    return None

# Events
def save_event(job_id: str, stream: str, payload: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO events (job_id, stream, payload_json) VALUES (?, ?, ?)",
        (job_id, stream, json.dumps(payload, ensure_ascii=False))
    )
    conn.commit()
    conn.close()

def get_events_by_job(job_id: str) -> List[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT stream, payload_json, created_at FROM events WHERE job_id = ? ORDER BY created_at ASC",
        (job_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"stream": r["stream"], "data": json.loads(r["payload_json"]), "created_at": r["created_at"]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Görev atamaları (assignments)
# ---------------------------------------------------------------------------
# Akış: süpervizör bir olayı bir ekibe atar -> atama kaydı oluşur ->
# ilgili rolün saha ekranı yalnızca kendi atamalarını çeker -> ekip
# "gördüm" / "tamamlandı" işaretler ve bu durum kalıcı olur.

#: Atamanın yaşam döngüsü. Arayüzdeki durum etiketleriyle birebir aynıdır.
ASSIGNMENT_STATUSES = ("atandi", "goruldu", "devam_ediyor", "tamamlandi")


def create_assignment(
    analysis_slug: str,
    role: str,
    camera_id: str = "",
    risk: str = "",
    headline: str = "",
    summary: str = "",
    reasoning: str = "",
    event_type: str = "",
    event_seconds: float = 0.0,
    event_timestamp: str = "",
    actions: Optional[List[str]] = None,
    video_file: str = "",
    note: str = "",
    assigned_by: str = "supervisor",
) -> dict:
    """Bir olayı belirli bir ekibe atar ve kaydı döner.

    Olayın karar çıktısındaki alanları atamaya kopyalanır; böylece saha ekranı
    analiz dosyasını okumak zorunda kalmaz ve atama anındaki bilgi sabit kalır.

    Args:
        analysis_slug: İlgili analizin kimliği (``data/library/analyses``).
        role: Atanan ekip rolü (örn. ``"sağlık"``).
        camera_id: Olayın görüldüğü kamera.
        risk: Karar çıktısındaki risk seviyesi.
        headline: Kısa kart başlığı.
        summary: Karar ajanının yazdığı olay özeti.
        reasoning: Ajanın gerekçesi (saha ekibi isterse ayrıntıyı görür).
        event_type: Olay tipi (örn. ``"person_fall"``).
        event_seconds: Olayın videodaki mutlak saniyesi; klip bu ana konumlanır.
        event_timestamp: Aynı anın ``MM:SS`` biçimi.
        actions: Önerilen aksiyonlar.
        video_file: Videonun repo-göreli yolu.
        note: Süpervizörün eklediği serbest not.
        assigned_by: Atamayı yapan.

    Returns:
        Oluşturulan atama kaydı.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO assignments (
            analysis_slug, camera_id, role, status, risk, headline, summary,
            reasoning, event_type, event_seconds, event_timestamp,
            actions_json, video_file, note, assigned_by
        ) VALUES (?, ?, ?, 'atandi', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_slug, camera_id, role, risk, headline, summary,
            reasoning, event_type, float(event_seconds or 0.0), event_timestamp,
            json.dumps(actions or [], ensure_ascii=False), video_file, note, assigned_by,
        ),
    )
    conn.commit()
    assignment_id = cursor.lastrowid
    cursor.execute("SELECT * FROM assignments WHERE id = ?", (assignment_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_assignment(row)


def get_assignments(
    role: Optional[str] = None,
    status: Optional[str] = None,
    analysis_slug: Optional[str] = None,
    limit: int = 100,
) -> List[dict]:
    """Atamaları filtreleyerek en yeniden eskiye sıralar.

    Args:
        role: Verilirse yalnızca bu rolün atamaları döner. Saha ekranı bunu
            kullanır; rol verilmezse (süpervizör görünümü) tümü döner.
        status: ``atandi`` / ``goruldu`` / ``tamamlandi`` filtresi.
        analysis_slug: Belirli bir analize ait atamalar.
        limit: Azami kayıt sayısı.
    """
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM assignments WHERE 1=1"
    params: list = []
    if role:
        query += " AND role = ?"
        params.append(role)
    if status:
        query += " AND status = ?"
        params.append(status)
    if analysis_slug:
        query += " AND analysis_slug = ?"
        params.append(analysis_slug)
    query += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_assignment(r) for r in rows]


def get_assignment(assignment_id: int) -> Optional[dict]:
    """Tek bir atamayı döner; yoksa ``None``."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM assignments WHERE id = ?", (assignment_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_assignment(row) if row else None


def update_assignment_status(assignment_id: int, status: str) -> Optional[dict]:
    """Atamanın durumunu ilerletir ve ilgili zaman damgasını yazar.

    Saha ekibinin "gördüm" / "tamamlandı" işaretlemesi kalıcı olsun diye
    veritabanına yazılır; eskiden aksiyonlar yalnızca ekranda bir bildirim
    gösterip kayboluyordu.

    Args:
        assignment_id: Atama kimliği.
        status: :data:`ASSIGNMENT_STATUSES` içinden bir değer.

    Returns:
        Güncellenmiş kayıt; atama yoksa ``None``.

    Raises:
        ValueError: Geçersiz durum değeri verilirse.
    """
    if status not in ASSIGNMENT_STATUSES:
        raise ValueError(
            f"Geçersiz atama durumu: {status!r}. "
            f"Geçerli değerler: {', '.join(ASSIGNMENT_STATUSES)}"
        )

    conn = get_connection()
    cursor = conn.cursor()
    if status == "goruldu":
        cursor.execute(
            "UPDATE assignments SET status = ?, "
            "acknowledged_at = COALESCE(acknowledged_at, CURRENT_TIMESTAMP) "
            "WHERE id = ?",
            (status, assignment_id),
        )
    elif status == "devam_ediyor":
        cursor.execute(
            "UPDATE assignments SET status = ?, "
            "acknowledged_at = COALESCE(acknowledged_at, CURRENT_TIMESTAMP) "
            "WHERE id = ?",
            (status, assignment_id),
        )
    elif status == "tamamlandi":
        cursor.execute(
            "UPDATE assignments SET status = ?, "
            "acknowledged_at = COALESCE(acknowledged_at, CURRENT_TIMESTAMP), "
            "resolved_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, assignment_id),
        )
    else:
        cursor.execute(
            "UPDATE assignments SET status = ? WHERE id = ?",
            (status, assignment_id),
        )
    conn.commit()
    cursor.execute("SELECT * FROM assignments WHERE id = ?", (assignment_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_assignment(row) if row else None


def get_assignment_counts() -> Dict[str, int]:
    """Rol/durum bazında atama sayılarını döner (süpervizör özet paneli için)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT status, COUNT(*) AS c FROM assignments GROUP BY status")
    by_status = {r["status"]: r["c"] for r in cursor.fetchall()}
    cursor.execute("SELECT role, COUNT(*) AS c FROM assignments GROUP BY role")
    by_role = {r["role"]: r["c"] for r in cursor.fetchall()}
    conn.close()
    return {"by_status": by_status, "by_role": by_role}


def _row_to_assignment(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "analysis_slug": row["analysis_slug"],
        "camera_id": row["camera_id"],
        "role": row["role"],
        "status": row["status"],
        "risk": row["risk"],
        "headline": row["headline"],
        "summary": row["summary"],
        "reasoning": row["reasoning"],
        "event_type": row["event_type"],
        "event_seconds": row["event_seconds"],
        "event_timestamp": row["event_timestamp"],
        "actions": json.loads(row["actions_json"] or "[]"),
        "video_file": row["video_file"],
        "note": row["note"],
        "assigned_by": row["assigned_by"],
        "created_at": row["created_at"],
        "acknowledged_at": row["acknowledged_at"],
        "resolved_at": row["resolved_at"],
    }


# ---------------------------------------------------------------------------
# RLHF / DPO Human-in-the-Loop Geri Bildirim Havuzu
# ---------------------------------------------------------------------------

FEEDBACK_TYPES = (
    "correct",         # Model kararı doğru onaylandı
    "false_positive",  # Yanlış alarm (olay yok veya zararsız)
    "wrong_risk",      # Hatalı risk seviyesi
    "wrong_event",     # Hatalı olay / nesne tespiti
    "wrong_action",    # Önerilen aksiyonlar uygun değil
    "other",           # Diğer düzeltme
)


def create_feedback(
    analysis_slug: str,
    camera_id: str = "",
    feedback_type: str = "correct",
    original_risk: str = "",
    original_summary: str = "",
    original_output: Optional[Dict[str, Any]] = None,
    corrected_risk: str = "",
    corrected_summary: str = "",
    corrected_actions: Optional[List[str]] = None,
    corrected_output: Optional[Dict[str, Any]] = None,
    prompt_context: Optional[Dict[str, Any]] = None,
    supervisor_notes: str = "",
) -> dict:
    """Süpervizörün analiz kararı üzerine yaptığı değerlendirmeyi/düzeltmeyi kaydeder.

    Args:
        analysis_slug: Analizin slug kimliği.
        camera_id: İlgili kamera kimliği.
        feedback_type: Geri bildirim türü (:data:`FEEDBACK_TYPES`).
        original_risk: Modelin ürettiği ilk risk seviyesi.
        original_summary: Modelin ürettiği ilk özet.
        original_output: Modelin ürettiği tüm JSON kararı (Rejected adayı).
        corrected_risk: Süpervizörün düzelttiği/onayladığı risk seviyesi.
        corrected_summary: Süpervizörün düzelttiği özet.
        corrected_actions: Süpervizörün düzelttiği aksiyon listesi.
        corrected_output: Düzeltilmiş tam karar JSON'u (Chosen adayı).
        prompt_context: Modele verilen kanıt paketi ve prompt bağlamı.
        supervisor_notes: Süpervizörün serbest metin notu.
    """
    if feedback_type not in FEEDBACK_TYPES:
        feedback_type = "other"

    orig_out = original_output or {}
    corr_out = corrected_output or {}
    # Eğer düzeltilmiş çıktı verilmemişse temel alanlarla tamamla
    if not corr_out:
        corr_out = {
            "risk": corrected_risk or original_risk,
            "summary": corrected_summary or original_summary,
            "actions": corrected_actions if corrected_actions is not None else orig_out.get("actions", []),
            "events": orig_out.get("events", []),
            "reasoning": supervisor_notes or orig_out.get("reasoning", ""),
        }

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO feedbacks (
            analysis_slug, camera_id, feedback_type,
            original_risk, original_summary, original_output_json,
            corrected_risk, corrected_summary, corrected_actions_json,
            corrected_output_json, prompt_context_json, supervisor_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_slug,
            camera_id,
            feedback_type,
            original_risk,
            original_summary,
            json.dumps(orig_out, ensure_ascii=False),
            corrected_risk or original_risk,
            corrected_summary or original_summary,
            json.dumps(corrected_actions if corrected_actions is not None else [], ensure_ascii=False),
            json.dumps(corr_out, ensure_ascii=False),
            json.dumps(prompt_context or {}, ensure_ascii=False),
            supervisor_notes,
        ),
    )
    feedback_id = cursor.lastrowid
    conn.commit()
    cursor.execute("SELECT * FROM feedbacks WHERE id = ?", (feedback_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_feedback(row)


def get_feedbacks(
    analysis_slug: Optional[str] = None,
    feedback_type: Optional[str] = None,
    limit: int = 100,
) -> List[dict]:
    """Kaydedilmiş geri bildirimleri en yeniden eskiye listeler."""
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM feedbacks WHERE 1=1"
    params: list = []
    if analysis_slug:
        query += " AND analysis_slug = ?"
        params.append(analysis_slug)
    if feedback_type:
        query += " AND feedback_type = ?"
        params.append(feedback_type)
    query += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_feedback(r) for r in rows]


def get_feedback_by_slug(analysis_slug: str) -> Optional[dict]:
    """Slug'a ait en son geri bildirimi döner; yoksa None."""
    rows = get_feedbacks(analysis_slug=analysis_slug, limit=1)
    return rows[0] if rows else None


def get_feedback_stats() -> Dict[str, Any]:
    """RLHF / DPO havuzu istatistiklerini hesaplar."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS total FROM feedbacks")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT feedback_type, COUNT(*) AS c FROM feedbacks GROUP BY feedback_type")
    by_type = {r["feedback_type"]: r["c"] for r in cursor.fetchall()}

    correct_count = by_type.get("correct", 0)
    correction_count = total - correct_count
    accuracy_rate = (correct_count / total * 100.0) if total > 0 else 100.0

    conn.close()
    return {
        "total": total,
        "correct_count": correct_count,
        "correction_count": correction_count,
        "accuracy_rate": round(accuracy_rate, 1),
        "by_type": by_type,
    }


def export_dpo_dataset_records(only_corrections: bool = False) -> List[Dict[str, Any]]:
    """DPO (Direct Preference Optimization) formatında veri çiftleri üretir.

    Format:
        {
            "prompt": "<Görsel / Geometrik Kanıtlar ve RAG Bağlamı>",
            "chosen": "<Süpervizörün Onayladığı/Düzelttiği Doğru Yanıt>",
            "rejected": "<Modelin Ürettiği İlk Hatalı Yanıt>",
            "metadata": { "slug": ..., "type": ... }
        }
    """
    feedbacks = get_feedbacks(limit=5000)
    records = []
    for f in feedbacks:
        fb_type = f["feedback_type"]
        if only_corrections and fb_type == "correct":
            continue

        prompt_data = f.get("prompt_context") or {}
        prompt_str = (
            prompt_data.get("prompt_text")
            or json.dumps(prompt_data, ensure_ascii=False, indent=2)
            if prompt_data else f"İSG Video Analizi: {f['analysis_slug']}"
        )

        chosen_str = json.dumps(f.get("corrected_output") or {
            "risk": f.get("corrected_risk"),
            "summary": f.get("corrected_summary"),
            "actions": f.get("corrected_actions"),
        }, ensure_ascii=False)

        rejected_str = json.dumps(f.get("original_output") or {
            "risk": f.get("original_risk"),
            "summary": f.get("original_summary"),
        }, ensure_ascii=False)

        records.append({
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str,
            "metadata": {
                "id": f["id"],
                "analysis_slug": f["analysis_slug"],
                "camera_id": f["camera_id"],
                "feedback_type": fb_type,
                "notes": f.get("supervisor_notes", ""),
                "created_at": f["created_at"],
            }
        })
    return records


def export_dpo_dataset_jsonl(only_corrections: bool = False) -> str:
    """Tüm DPO kayıtlarını satır satır JSONL formatında metin olarak döner."""
    records = export_dpo_dataset_records(only_corrections=only_corrections)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    return "\n".join(lines)


def _row_to_feedback(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "analysis_slug": row["analysis_slug"],
        "camera_id": row["camera_id"],
        "feedback_type": row["feedback_type"],
        "original_risk": row["original_risk"],
        "original_summary": row["original_summary"],
        "original_output": json.loads(row["original_output_json"] or "{}"),
        "corrected_risk": row["corrected_risk"],
        "corrected_summary": row["corrected_summary"],
        "corrected_actions": json.loads(row["corrected_actions_json"] or "[]"),
        "corrected_output": json.loads(row["corrected_output_json"] or "{}"),
        "prompt_context": json.loads(row["prompt_context_json"] or "{}"),
        "supervisor_notes": row["supervisor_notes"],
        "created_at": row["created_at"],
    }

