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
    # Saha uyarıları (field alerts) — UI'dan gelen manuel/otomatik uyarılar
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS field_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT,
            risk TEXT,
            headline TEXT,
            summary TEXT,
            actions_json TEXT,
            risk_segment_json TEXT,
            target_roles_json TEXT,
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


# Field alerts

def create_field_alert(
    camera_id: str,
    risk: str,
    headline: str,
    summary: str,
    actions: List[str],
    risk_segment: Dict[str, Any],
    target_roles: List[str],
) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO field_alerts (
            camera_id, risk, headline, summary,
            actions_json, risk_segment_json, target_roles_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            camera_id,
            risk,
            headline,
            summary,
            json.dumps(actions, ensure_ascii=False),
            json.dumps(risk_segment, ensure_ascii=False),
            json.dumps(target_roles, ensure_ascii=False),
        ),
    )
    conn.commit()
    alert_id = cursor.lastrowid
    cursor.execute("SELECT * FROM field_alerts WHERE id = ?", (alert_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_field_alert(row)


def get_field_alerts(role: Optional[str] = None, limit: int = 100) -> List[dict]:
    """Saha uyarılarını created_at DESC sıralar.

    role verilirse; target_roles boş olanlar veya role içerenler döner.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM field_alerts ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()

    alerts = [_row_to_field_alert(r) for r in rows]
    if not role:
        return alerts

    filtered = []
    for alert in alerts:
        target_roles = alert.get("target_roles") or []
        if not target_roles or role in target_roles:
            filtered.append(alert)
    return filtered


def _row_to_field_alert(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "camera_id": row["camera_id"],
        "risk": row["risk"],
        "headline": row["headline"],
        "summary": row["summary"],
        "actions": json.loads(row["actions_json"] or "[]"),
        "risk_segment": json.loads(row["risk_segment_json"] or "{}"),
        "target_roles": json.loads(row["target_roles_json"] or "[]"),
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# Görev atamaları (assignments)
# ---------------------------------------------------------------------------
# Akış: süpervizör bir olayı bir ekibe atar -> atama kaydı oluşur ->
# ilgili rolün saha ekranı yalnızca kendi atamalarını çeker -> ekip
# "gördüm" / "tamamlandı" işaretler ve bu durum kalıcı olur.

#: Atamanın yaşam döngüsü. Arayüzdeki durum etiketleriyle birebir aynıdır.
ASSIGNMENT_STATUSES = ("atandi", "goruldu", "tamamlandi")


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
