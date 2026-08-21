import sqlite3
import json
import uuid
from typing import List, Dict, Optional
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
