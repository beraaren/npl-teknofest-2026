"""active_risk_window() için birim testleri.

Bu fonksiyon kamera duvarındaki kırmızı çerçevenin ne zaman yanıp ne zaman
sönük kalacağını belirler; hatalı davranışı doğrudan görsel bir kusra
(sürekli yanan veya hiç yanmayan çerçeve) dönüşür.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.gateway.replay import active_risk_window, _event_window  # noqa: E402


def test_event_window_uses_timestamp_sec_and_duration():
    stamp = {"timestamp_sec": 10.0, "duration": 5.0, "seconds": 10.0}
    start, end = _event_window(stamp)
    assert start == 10.0
    assert end == 15.0


def test_event_window_zero_duration_is_zero_width():
    stamp = {"timestamp_sec": 44.0, "duration": 0.0, "seconds": 44.0}
    start, end = _event_window(stamp)
    assert start == end == 44.0


def test_active_window_none_outside_all_windows():
    stamps = [
        {"timestamp_sec": 3.0, "duration": 51.0, "severity": "high", "event_type": "ppe_missing"},
    ]
    # Pencere disinda: baslangictan once
    assert active_risk_window(stamps, 1.0) is None
    # Pencere disinda: bitisten sonra
    assert active_risk_window(stamps, 60.0) is None


def test_active_window_active_inside_range():
    stamps = [
        {"timestamp_sec": 3.0, "duration": 51.0, "severity": "high", "event_type": "ppe_missing"},
    ]
    win = active_risk_window(stamps, 30.0)
    assert win is not None
    assert win["start_sec"] == 3.0
    assert win["end_sec"] == 54.0
    assert win["risk"] == "Yüksek"


def test_active_window_zero_duration_event_never_active():
    stamps = [
        {"timestamp_sec": 44.0, "duration": 0.0, "severity": "high", "event_type": "person_fall"},
    ]
    # Sifir genislikli pencere hicbir zaman aktif olmaz (uydurulmus sure yok).
    assert active_risk_window(stamps, 44.0) is None


def test_active_window_picks_highest_severity_when_overlapping():
    stamps = [
        {"timestamp_sec": 0.0, "duration": 60.0, "severity": "low", "event_type": "ppe_missing"},
        {"timestamp_sec": 10.0, "duration": 20.0, "severity": "critical", "event_type": "gathering"},
    ]
    win = active_risk_window(stamps, 15.0)
    assert win["severity"] == "critical"
    assert win["risk"] == "Yüksek"


def test_active_window_critical_severity_maps_to_yuksek_risk():
    stamps = [
        {"timestamp_sec": 0.0, "duration": 10.0, "severity": "critical", "event_type": "fire_smoke"},
    ]
    win = active_risk_window(stamps, 5.0)
    assert win["risk"] == "Yüksek"
