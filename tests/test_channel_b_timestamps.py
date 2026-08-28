"""Kanal B video timestamp normalizasyonu regresyon testleri."""
import sys
from pathlib import Path

import pytest

KANAL_B_DIR = Path(__file__).resolve().parents[1] / "Kanal_B"
sys.path.insert(0, str(KANAL_B_DIR))

from vlm_backend import _clamp_timestamp  # noqa: E402


def test_preserves_absolute_timestamp_in_ambiguous_range():
    """Mutlak/geçerli değer, klibe göreli sanılıp ikinci kez ötelenmemeli."""
    assert _clamp_timestamp(45.0, start_sec=30.0, end_sec=90.0) == 45.0


@pytest.mark.parametrize("timestamp", [30.0, 90.0])
def test_preserves_absolute_segment_boundaries(timestamp):
    assert _clamp_timestamp(timestamp, start_sec=30.0, end_sec=90.0) == timestamp


def test_preserves_unambiguous_absolute_timestamp():
    assert _clamp_timestamp(145.0, start_sec=120.0, end_sec=180.0) == 145.0


def test_offsets_only_unambiguous_clip_relative_timestamp():
    # 25, 120-180 mutlak aralığında olamaz; klibin 25. saniyesi kabul edilir.
    assert _clamp_timestamp(25.0, start_sec=120.0, end_sec=180.0) == 145.0


def test_first_clip_timestamp_needs_no_offset():
    assert _clamp_timestamp(25.0, start_sec=0.0, end_sec=60.0) == 25.0


@pytest.mark.parametrize("value", [None, "00:45", "gecersiz"])
def test_rejects_non_numeric_timestamp(value):
    assert _clamp_timestamp(value, start_sec=30.0, end_sec=90.0) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-5.0, 30.0), (999.0, 90.0)],
)
def test_clamps_out_of_range_timestamp(value, expected):
    assert _clamp_timestamp(value, start_sec=30.0, end_sec=90.0) == expected
