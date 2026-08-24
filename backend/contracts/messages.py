from pydantic import BaseModel
from typing import Literal
from datetime import datetime

class FrameChunk(BaseModel):
    job_id: str
    camera_id: str
    frame_paths: list[str]
    frame_indices: list[int]
    fps: float
    is_last: bool = False
    # Kaynak videonun yolu. Kanal B (vlm-service) videoyu EVREN'in video
    # modeline BÜTÜN olarak gönderir; kare yollarından video yeniden
    # oluşturulamayacağı için kaynak yol zincirde taşınır. Varsayılan boş
    # bırakılmıştır: yol verilmezse Kanal B kare/grid yoluna düşer.
    video_path: str = ""

class EventDetected(BaseModel):
    job_id: str
    camera_id: str
    event_type: str
    timestamp: str
    confidence: float
    description: str
    # FrameChunk'tan taşınır; vlm-service video modunu bununla kullanır.
    video_path: str = ""

class VlmInterpreted(BaseModel):
    job_id: str
    camera_id: str
    interpretation: dict
    critical_indices: list[int]

class DecisionFinal(BaseModel):
    job_id: str
    camera_id: str
    summary: str
    events: list[dict]
    risk: Literal["Düşük", "Orta", "Yüksek"]
    actions: list[str]
    reasoning: str
    confidence: float
    triggered_mock_tools: list[dict]

class ToolExecuted(BaseModel):
    job_id: str
    tool_name: str
    params: dict
    status: str
    mock_result: str

class NotificationPush(BaseModel):
    job_id: str
    camera_id: str
    risk: str
    headline: str
    summary: str
    actions: list[str]
    created_at: datetime
