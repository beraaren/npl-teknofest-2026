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


def test_event_window_positive_short_duration_has_three_second_minimum():
    stamp = {"timestamp_sec": 10.0, "duration": 0.25, "seconds": 10.0}
    start, end = _event_window(stamp)
    assert start == 10.0
    assert end == 13.0


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


# ---------------------------------------------------------------------------
# ReplayEngine._draw_analysis: aynı videonun iki kamerada birden oynamaması
# ---------------------------------------------------------------------------

from unittest.mock import patch

from backend.gateway.replay import ReplayEngine, CameraStream


async def _noop_broadcast(_payload):
    return None


def _noop_save_event(_job_id, _stream, _payload):
    return None


def _make_engine(camera_count: int, pool_slugs: list[str]) -> ReplayEngine:
    """library'yi sahteleyip belirli boyutta bir havuzla motor kurar."""
    with patch("backend.gateway.replay.library.count", return_value=len(pool_slugs)):
        engine = ReplayEngine(_noop_broadcast, _noop_save_event, camera_count=camera_count)
    return engine


def test_draw_analysis_returns_none_on_temporary_clash_when_pool_exceeds_cameras():
    """Havuz kameradan büyükken, deneme bütçesi tükenip hâlâ boşta slug
    bulunamazsa (geçici çakışma) zorla tekrar seçilmez; None dönülür ki iki
    kamera aynı videoyu aynı anda göstermesin. Gerçek video kütüphanesinde bu,
    deste ortada yeniden karıştırıldığında ekrandaki slug'ların tekrar
    çekilmesiyle oluşabilir; burada deste doğrudan "hepsi ekranda" durumuna
    sabitlenerek deterministik test edilir."""
    engine = _make_engine(camera_count=3, pool_slugs=["a", "b", "c", "d", "e"])
    cam_ids = list(engine.cameras)

    with patch("backend.gateway.replay.library.count", return_value=5), \
         patch("backend.gateway.replay.library.slugs", return_value=[]), \
         patch("backend.gateway.replay.library.get", side_effect=lambda s: {"slug": s}):
        # Deste sadece ekranda olan slug'ı içeriyor; boşalınca yeniden
        # karıştırılacak yeni slug da yok (arama bütçesi tükendi).
        engine.cameras[cam_ids[1]].analysis = {"slug": "on-screen-1"}
        engine._deck = ["on-screen-1"]
        result = engine._draw_analysis(for_camera=cam_ids[0])
        assert result is None


def test_draw_analysis_forces_repeat_when_pool_not_larger_than_cameras():
    """Havuz kamera sayısına eşit/küçükken tekrar kaçınılmazdır; None dönmez."""
    engine = _make_engine(camera_count=2, pool_slugs=["a", "b"])
    cam_ids = list(engine.cameras)

    with patch("backend.gateway.replay.library.count", return_value=2), \
         patch("backend.gateway.replay.library.slugs", return_value=[]), \
         patch("backend.gateway.replay.library.get", side_effect=lambda s: {"slug": s}):
        engine.cameras[cam_ids[1]].analysis = {"slug": "on-screen-1"}
        engine._deck = ["on-screen-1"]
        result = engine._draw_analysis(for_camera=cam_ids[0])
        assert result is not None
        assert result["slug"] == "on-screen-1"
