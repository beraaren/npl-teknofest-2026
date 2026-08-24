"""VLM/LLM çıkarım backend'leri.

Bu paket, karar ve algı katmanlarını çıkarım sağlayıcısından yalıtır:
çağıranlar yalnızca :class:`~src.models.vlm_backend.VLMBackend` arayüzünü
(``name()`` ve ``generate()``) bilir, hangi sağlayıcının kullanıldığını
bilmez.
"""
