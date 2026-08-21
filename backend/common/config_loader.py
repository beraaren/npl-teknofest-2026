"""Merkezi config yukleyici - tum backend servisleri bu modulu kullanir."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from functools import lru_cache

_APP_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

@lru_cache(maxsize=1)
def load_app_config():
    """AppConfig yukler; basarisiz olursa varsayilanlarla doner."""
    try:
        from src.config import load_config
        candidates = [
            os.environ.get("TEKNOFEST_CONFIG"),
            "/app/config.yaml",
            str(_APP_ROOT / "config.yaml"),
        ]
        for path in candidates:
            if path and Path(path).exists():
                return load_config(path)
        return load_config(None)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            f"config.yaml yuklenemedi, varsayilanlar kullaniliyor: {exc}"
        )
        try:
            from src.config import AppConfig
            return AppConfig()
        except Exception:
            return None
