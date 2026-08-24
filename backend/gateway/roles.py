"""Saha ekibi rolleri ve rol adı normalizasyonu.

NEDEN NORMALİZASYON GEREKLİ
---------------------------
Roller arayüzde Türkçe görünür ("Sağlık Ekibi") ama veritabanında anahtar
olarak kullanılır. Aynı rol farklı yazımlarla gelirse (``sağlık`` / ``saglik`` /
``Sağlık``) ayrı kayıtlar oluşur ve saha ekibi kendisine atanan görevi
görmez — atama sessizce kaybolur. Bu modül tek bir kanonik yazım dayatır:
ASCII, küçük harf (``saglik``). Görünen etiket ayrı tutulur.
"""
from __future__ import annotations

import unicodedata

#: Kanonik rol kimliği -> arayüzde görünen etiket.
FIELD_ROLES: dict[str, str] = {
    "saglik": "Sağlık Ekibi",
    "temizlik": "Temizlik Görevlisi",
    "teknisyen": "Teknisyen",
    "guvenlik": "Güvenlik Ekibi",
}

#: Kanonik olmayan yazımların eşlemesi. Türkçe karakter dönüşümü zaten
#: yapıldığı için burada yalnızca eş anlamlılar tutulur.
_ALIASES: dict[str, str] = {
    "saglikekibi": "saglik",
    "saglik-ekibi": "saglik",
    "health": "saglik",
    "temizlikgorevlisi": "temizlik",
    "cleaning": "temizlik",
    "technician": "teknisyen",
    "tekniker": "teknisyen",
    "security": "guvenlik",
}

_TURKISH = str.maketrans("çğıİöşüÇĞÖŞÜ", "cgiiosuCGOSU")


def normalize_role(value: str | None) -> str:
    """Serbest yazımlı rol adını kanonik kimliğe çevirir.

    Args:
        value: Arayüzden veya API'den gelen rol adı.

    Returns:
        Kanonik rol kimliği. Tanınmayan ad, sadeleştirilmiş hâliyle döner
        (bilinmeyen roller de tutarlı biçimde saklanır); boş girdi ``""``.

    Examples:
        >>> normalize_role("Sağlık Ekibi")
        'saglik'
        >>> normalize_role("saglik")
        'saglik'
    """
    if not value:
        return ""
    text = str(value).strip().translate(_TURKISH)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    compact = "".join(ch for ch in text if ch.isalnum())

    if compact in FIELD_ROLES:
        return compact
    if compact in _ALIASES:
        return _ALIASES[compact]

    # Tire/boşluk içeren varyantları da dene
    dashed = text.replace(" ", "-").strip("-")
    if dashed in _ALIASES:
        return _ALIASES[dashed]

    # "Güvenlik Ekibi" / "Temizlik Görevlisi" gibi unvan soneklerini at ve
    # yeniden dene. Her varyantı tek tek listelemek yerine sonek soymak,
    # arayüz etiketleri değiştiğinde de çalışmaya devam eder.
    for suffix in ("ekibi", "gorevlisi", "personeli", "birimi"):
        if compact.endswith(suffix) and len(compact) > len(suffix):
            stripped = compact[: -len(suffix)]
            if stripped in FIELD_ROLES:
                return stripped
            if stripped in _ALIASES:
                return _ALIASES[stripped]

    return compact


def role_label(role: str | None) -> str:
    """Kanonik rol kimliğinin arayüzde görünecek etiketini döner."""
    canonical = normalize_role(role)
    return FIELD_ROLES.get(canonical, canonical or "Bilinmeyen")


def all_roles() -> list[dict]:
    """Arayüzün rol seçicisini kurmak için rol listesini döner."""
    return [{"role": key, "label": label} for key, label in FIELD_ROLES.items()]
