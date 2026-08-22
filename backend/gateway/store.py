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
    conn.commit()
    conn.close()


def seed_cameras():
    """Cameras tablosu boşsa 9 adet pseudolive kamerası ekle."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as c FROM cameras")
    row = cursor.fetchone()
    if row and row["c"] > 0:
        conn.close()
        return
    rows = [
        (f"cam-{i:02d}", f"Kamera {i:02d}", "pseudolive", "aktif")
        for i in range(1, 10)
    ]
    cursor.executemany(
        "INSERT INTO cameras (id, name, source, status) VALUES (?, ?, ?, ?)",
        rows,
    )
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
