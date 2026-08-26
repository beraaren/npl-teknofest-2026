"""Merkezi loglama."""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path


def _utf8_console_stream():
    """Konsol akışını UTF-8'e ayarlar; başarısız olursa güvenli bir sarmalayıcı döner.

    Öncelik sırası:

    1. ``sys.stdout.reconfigure(...)`` — akışı **yerinde** UTF-8'e alır. Aynı
       alttaki tampon kullanıldığı için diğer yazıcılarla (``print`` vb.)
       çıktı sırası bozulmaz. Tercih edilen yol budur.
    2. ``sys.stdout.buffer`` üzerine yeni bir ``TextIOWrapper`` — akış
       yeniden yapılandırmayı desteklemiyorsa (bazı sarmalanmış/yakalanmış
       akışlar) kullanılır.
    3. Hiçbiri olmazsa ``sys.stdout`` olduğu gibi döner; bu durumda davranış
       eskisi gibi olur ama en azından bir istisna yükseltilmez.

    Returns:
        Yazmaya hazır bir metin akışı.
    """
    stream = sys.stdout

    # 1) Yerinde yeniden yapılandırma (Python 3.7+)
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
            return stream
        except (ValueError, OSError):
            pass

    # 2) Alttaki ikili tamponu UTF-8 ile sarmala
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        try:
            return io.TextIOWrapper(
                buffer, encoding="utf-8", errors="backslashreplace", line_buffering=True
            )
        except (ValueError, OSError):
            pass

    # 3) Son çare: olduğu gibi kullan
    return stream


def get_logger(name: str, log_dir: str | Path = "logs") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Konsol
    #
    # Windows konsolu varsayılan olarak cp1254 kod sayfasını kullanır ve bu
    # sayfada bulunmayan karakterler (log metinlerindeki "→", video adlarındaki
    # emoji gibi) yazılmaya çalışıldığında logging katmanı UnicodeEncodeError
    # fırlatıp ekrana traceback basıyordu. Analizi durdurmuyordu ama çıktıyı
    # kirletiyordu.
    #
    # Kök çözüm karakterleri değiştirmek değil, akışı UTF-8'e almaktır; ayrıca
    # temsil edilemeyen bir karakter kalırsa `backslashreplace` ile hata yerine
    # kaçış dizisi yazılır, böylece loglama hiçbir koşulda çökmez.
    stream = logging.StreamHandler(_utf8_console_stream())
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    # Dosya
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
