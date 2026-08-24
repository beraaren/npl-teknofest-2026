"""VLM/LLM çıkarım backend'leri — çağıranları sağlayıcıdan yalıtan katman.

Bu modül, karar ve algı katmanlarının çıkarım sağlayıcısıyla konuştuğu **tek**
noktadır. Çağıranlar (``src/main.py``, :mod:`src.reasoning.decision_agent`,
:mod:`src.perception.vehicle_labeler`, ``test_akis.py``) yalnızca iki şey bilir:

    backend = create_backend(config.vlm, force="server")
    metin   = backend.generate(kareler, prompt, temperature=0.15, max_tokens=4096)
    ad      = backend.name()

Hangi sağlayıcının kullanıldığı, isteğin nasıl paketlendiği ve sağlayıcıya özgü
kısıtların nasıl aşıldığı bu modülün içinde kalır.

Sağlayıcı kısıtlarının şeffaf yönetimi
--------------------------------------
TEKNOFEST EVREN servisi istek başına **en fazla iki görüntü** kabul eder;
üçüncü görüntü ``HTTP 400 "At most 2 image(s) may be provided"`` döndürür.
Buna karşılık bu projenin akışları doğal olarak daha fazla kare gönderir
(4 kritik kare, 8 örneklenmiş kare, N araç kırpıntısı). Kısıt çağıranlara
yansıtılmak yerine burada çözülür: kare sayısı sınırı aşarsa kareler tek bir
**grid görüntüsünde** birleştirilir ve prompt'a gridin okunma düzenini
açıklayan bir not eklenir. Böylece hem kısıt aşılır hem de modelin zamansal
sıralamayı anlaması korunur.

Ayrıca EVREN'in belgelenmiş üç sessiz tuzağı burada karşılanır:

* **Tanınmayan model adı 404 döndürmez**, sessizce ``llm-fast``e yönlenir. Bu
  yüzden ilk çağrıda model adı ``GET /v1/models`` çıktısına karşı doğrulanır ve
  uyuşmazlıkta uyarı loglanır.
* **Boş yanıt + ``finish_reason="length"``**: akıl yürütme izi bütçeyi
  tüketmişse içerik boş döner. Bu durum sessizce geçilmez, açık uyarı loglanır.
* **İstemci zaman aşımı**: sunucu yığını 1800 sn kullanır; daha kısa bir istemci
  zaman aşımı bağlantıyı modelden önce koparır. Varsayılan
  :attr:`~src.config.ServerConfig.timeout_sec` bu nedenle 1800'dür.
"""
from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from numpy.typing import NDArray

from ..config import VLMConfig
from ..utils.logger import get_logger

logger = get_logger("VLMBackend")

# EVREN gövde sınırı 256 MB'dır; base64 kodlaması boyutu ~1/3 artırdığından ham
# dosya için güvenli üst sınır yaklaşık 190 MB'a karşılık gelir.
_BODY_LIMIT_BYTES = 256 * 1024 * 1024
_RAW_VIDEO_LIMIT_BYTES = int(_BODY_LIMIT_BYTES / 1.34)

# Grid hücrelerinin azami kenar uzunluğu. Kareler bundan büyükse küçültülür;
# amaç görüntü token maliyetini öngörülebilir tutmaktır.
_MAX_GRID_CELL_PX = 512


# ---------------------------------------------------------------------------
# Görüntü yardımcıları
# ---------------------------------------------------------------------------

def _encode_jpeg_b64(image: NDArray[np.uint8], quality: int = 85) -> str:
    """RGB ``ndarray``i JPEG'e sıkıştırıp base64 metnine çevirir.

    OpenAI-uyumlu API'ler görsel veriyi ``data:image/jpeg;base64,...`` biçiminde
    bir veri URI'si olarak bekler.

    Args:
        image: RGB, ``HWC`` düzeninde ``uint8`` görüntü.
        quality: JPEG kalite değeri (1-95).

    Returns:
        Base64 kodlanmış JPEG içeriği (veri URI öneki olmadan).
    """
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _letterbox(image: NDArray[np.uint8], target_w: int, target_h: int) -> NDArray[np.uint8]:
    """Görüntüyü en/boy oranını koruyarak hedef boyuta yerleştirir.

    Basit yeniden boyutlandırma yerine oran korunarak kenarlara dolgu eklenir.
    Bunun nedeni gridin araç türü tanımlama gibi **biçime duyarlı** görevlerde
    de kullanılmasıdır: gerilmiş bir forklift görüntüsü modelin şekil
    çıkarımını bozar.

    Args:
        image: RGB ``uint8`` görüntü.
        target_w: Hedef hücre genişliği.
        target_h: Hedef hücre yüksekliği.

    Returns:
        Tam olarak ``(target_h, target_w, 3)`` boyutunda, ortalanmış ve siyah
        dolgulu görüntü.
    """
    from PIL import Image

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = np.asarray(
        Image.fromarray(np.ascontiguousarray(image)).resize((new_w, new_h), Image.BILINEAR)
    )

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
    return canvas


def build_frame_grid(
    frames: Sequence[NDArray[np.uint8]],
    cols: int = 4,
) -> tuple[NDArray[np.uint8], int, int]:
    """Birden fazla kareyi tek bir grid görüntüsünde birleştirir.

    Kareler soldan sağa, yukarıdan aşağıya yerleştirilir; bu düzen prompt'a
    :func:`_grid_note` ile açıkça bildirilir. Farklı boyuttaki kareler (örneğin
    araç kırpıntıları) ortak hücre boyutuna :func:`_letterbox` ile oturtulur.

    Args:
        frames: En az bir RGB ``uint8`` kare içeren dizi.
        cols: Sütun sayısı; satır sayısı kare sayısından türetilir.

    Returns:
        ``(grid, rows, cols)`` üçlüsü. ``grid`` birleştirilmiş RGB görüntüdür.

    Raises:
        ValueError: ``frames`` boşsa.
    """
    if not frames:
        raise ValueError("Grid oluşturmak için en az bir kare gerekir.")

    cols = max(1, min(cols, len(frames)))
    rows = (len(frames) + cols - 1) // cols

    cell_w = min(_MAX_GRID_CELL_PX, max(int(f.shape[1]) for f in frames))
    cell_h = min(_MAX_GRID_CELL_PX, max(int(f.shape[0]) for f in frames))

    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r, c = divmod(idx, cols)
        grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = _letterbox(
            frame, cell_w, cell_h
        )
    return grid, rows, cols


def _grid_note(rows: int, cols: int, n_frames: int) -> str:
    """Grid düzenini modele açıklayan prompt önekini üretir.

    Kareler tek bir görüntüde birleştirildiğinde model, bunun bir mozaik
    olduğunu ve hücre sırasının zaman akışını temsil ettiğini bilmezse
    görüntüyü tek bir sahne sanır. Bu not o bilgiyi verir.

    Args:
        rows: Grid satır sayısı.
        cols: Grid sütun sayısı.
        n_frames: Gridde yer alan gerçek kare sayısı (boş hücreler hariç).

    Returns:
        Prompt'un başına eklenecek açıklama metni.
    """
    return (
        f"The image you are given is a {rows}x{cols} grid (mosaic) containing "
        f"{n_frames} frames sampled from a single video, in chronological order. "
        "Cells are numbered starting from 0, left-to-right and top-to-bottom; "
        "cell 0 is the earliest moment and the last cell is the latest. "
        "Read the grid as a time sequence, not as one single scene. "
        "Any empty (black) cells at the end are padding and must be ignored.\n\n"
    )


# ---------------------------------------------------------------------------
# Ortak arayüz
# ---------------------------------------------------------------------------

class VLMBackend(ABC):
    """Tüm çıkarım backend'lerinin uyduğu soyut arayüz.

    Çağıranlar bu iki metodun dışına çıkmaz; yeni bir sağlayıcı eklemek için
    bu sınıftan türetip :meth:`generate` uygulanır ve
    :func:`create_backend` içine bir dal eklenir.
    """

    #: :meth:`name` tarafından döndürülen kimlik. :mod:`src.reasoning.decision_agent`
    #: üretim parametrelerini (sıcaklık, token bütçesi) bu değere göre seçtiği için
    #: değerler ``"server" | "vllm" | "llama_cpp" | "transformers"`` kümesinden olmalıdır.
    backend_name: str = "base"

    def name(self) -> str:
        """Backend'in kimliğini döner."""
        return self.backend_name

    @abstractmethod
    def generate(
        self,
        images: Sequence[NDArray[np.uint8]],
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Verilen kareler ve prompt için modelin ham metin çıktısını üretir.

        Args:
            images: RGB ``uint8`` kare listesi. Boş liste metin-only çağrı
                anlamına gelir (karar ajanı kanıtları metin olarak da verebilir).
            prompt: Modele gönderilecek talimat metni.
            temperature: Örnekleme sıcaklığı; ``None`` ise backend'in
                yapılandırılmış varsayılanı kullanılır.
            max_tokens: Üretilecek azami token sayısı; ``None`` ise backend'in
                yapılandırılmış varsayılanı kullanılır.

        Returns:
            Modelin ham metin yanıtı (JSON ayrıştırma çağırana bırakılır).
        """


# ---------------------------------------------------------------------------
# OpenAI-uyumlu HTTP backend'leri
# ---------------------------------------------------------------------------

class _OpenAICompatibleBackend(VLMBackend):
    """OpenAI ``/v1/chat/completions`` sözleşmesini kullanan backend'lerin tabanı.

    TEKNOFEST EVREN servisi, yerel ``llama-server`` ve ``vllm serve`` aynı
    sözleşmeyi konuşur; aralarındaki fark yalnızca adres, model adı ve kısıt
    değerleridir. Bu sınıf ortak istek kurulumunu, görüntü paketlemesini ve
    hata yorumlamasını barındırır.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        temperature: float = 0.15,
        max_tokens: int = 4096,
        timeout_sec: int = 1800,
        api_key: str | None = None,
        max_images_per_request: int = 2,
        grid_columns: int = 4,
        video_model: str | None = None,
        enable_thinking: bool | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.api_key = api_key
        self.max_images_per_request = max(0, max_images_per_request)
        self.grid_columns = max(1, grid_columns)
        self.video_model = video_model or model_name
        self.enable_thinking = enable_thinking
        self._model_verified = False

    # -- HTTP altyapısı ---------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def available_models(self) -> List[str]:
        """Sağlayıcının sunduğu model aliaslarını listeler.

        Returns:
            Model kimlikleri listesi; istek başarısız olursa boş liste.
        """
        import requests

        try:
            resp = requests.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=15
            )
            resp.raise_for_status()
            return [m.get("id", "") for m in resp.json().get("data", [])]
        except Exception as exc:
            logger.debug(f"Model listesi alınamadı: {exc}")
            return []

    def _verify_model_once(self) -> None:
        """Model adını sağlayıcının listesine karşı bir kez doğrular.

        EVREN'de tanınmayan bir model adı hata döndürmez; istek sessizce
        ``llm-fast`` hedefine yönlendirilir. Bu durumda ölçümler yanlış modele
        atfedilir. Sessiz sapmayı görünür kılmak için ilk çağrıda ad denetlenir;
        denetim başarısız olsa bile çağrı engellenmez.
        """
        if self._model_verified:
            return
        self._model_verified = True  # başarısız olsa da tekrar denenmez

        models = self.available_models()
        if not models:
            return
        if self.model_name not in models:
            logger.warning(
                f"Model adı '{self.model_name}' sağlayıcı listesinde yok "
                f"(mevcut: {', '.join(models)}). Sağlayıcı isteği sessizce "
                f"varsayılan modele yönlendirebilir; ad doğrulanmalı."
            )

    def _build_payload(
        self,
        model: str,
        content: List[Dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> Dict[str, Any]:
        """İstek gövdesini kurar ve akıl yürütme bayrağını ekler.

        :attr:`enable_thinking` ``None`` değilse ``chat_template_kwargs`` olarak
        iletilir. Bu bir vLLM uzantısı olduğundan, desteklemeyen sağlayıcılarda
        yapılandırma ``None`` bırakılarak parametre hiç gönderilmez.

        Args:
            model: Kullanılacak model aliası.
            content: OpenAI çok modlu içerik parçaları.
            temperature: Örnekleme sıcaklığı veya ``None`` (varsayılan kullanılır).
            max_tokens: Azami üretim token'ı veya ``None`` (varsayılan kullanılır).

        Returns:
            ``/chat/completions`` için hazır istek gövdesi.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        return payload

    def _post_chat(self, payload: Dict[str, Any]) -> str:
        """``/chat/completions`` çağrısını yapar ve metin içeriğini döner.

        Args:
            payload: OpenAI sözleşmesine uygun istek gövdesi.

        Returns:
            Modelin ürettiği metin.

        Raises:
            RuntimeError: HTTP hatası veya beklenmeyen yanıt yapısı durumunda.
        """
        import requests

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_sec,
            )
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Çıkarım isteği {self.timeout_sec} sn içinde tamamlanmadı. "
                f"Uzun video/çok kare isteklerinde bu değerin artırılması gerekir."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Çıkarım sunucusuna bağlanılamadı ({self.base_url}): {exc}") from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"Çıkarım sunucusu HTTP {resp.status_code} döndürdü: {resp.text[:400]}"
            )

        try:
            data = resp.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"Yanıt yapısı beklenmedik: {resp.text[:400]}") from exc

        # Belgelenmiş tuzak: akıl yürütme izi token bütçesini tüketirse içerik
        # boş döner ve HTTP 200 alınır. Sessizce geçmek, üst katmanda anlamsız
        # bir "ayrıştırılamadı" hatasına dönüşür; nedeni burada söylenir.
        if not text.strip() and finish == "length":
            logger.warning(
                f"Model boş içerik döndürdü (finish_reason='length', "
                f"max_tokens={payload.get('max_tokens')}). Token bütçesi yanıt için "
                f"yetersiz kalmış olabilir; akıl yürütme açıksa en az 2048 önerilir."
            )
        elif not text.strip():
            logger.warning(f"Model boş içerik döndürdü (finish_reason={finish!r}).")

        return text

    # -- İstek kurulumu ---------------------------------------------------

    def _build_image_content(
        self, images: Sequence[NDArray[np.uint8]], prompt: str
    ) -> tuple[List[Dict[str, Any]], str]:
        """Kareleri sağlayıcı kısıtına uyacak şekilde içerik parçalarına çevirir.

        Kare sayısı :attr:`max_images_per_request` sınırını aşarsa kareler tek
        bir grid görüntüsünde birleştirilir ve prompt'a grid düzeni notu eklenir.

        Args:
            images: RGB kare listesi.
            prompt: Özgün prompt metni.

        Returns:
            ``(içerik_parçaları, güncellenmiş_prompt)`` ikilisi.
        """
        if not images:
            return [], prompt

        limit = self.max_images_per_request

        if limit <= 0:
            # Sağlayıcı hiç görüntü kabul etmiyor (EVREN'de "vlm" aliası böyledir).
            raise RuntimeError(
                f"'{self.model_name}' modeli görüntü kabul etmiyor. Görüntü içeren "
                f"istekler için görüntü destekli bir model (EVREN'de llm-fast veya "
                f"llm-large) yapılandırılmalıdır."
            )

        if len(images) <= limit:
            payload_images = list(images)
            effective_prompt = prompt
        else:
            grid, rows, cols = build_frame_grid(images, self.grid_columns)
            payload_images = [grid]
            effective_prompt = _grid_note(rows, cols, len(images)) + prompt
            logger.info(
                f"{len(images)} kare tek {rows}x{cols} grid görüntüsünde birleştirildi "
                f"(sağlayıcı sınırı: istek başına {limit} görüntü)."
            )

        parts: List[Dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_jpeg_b64(img)}"},
            }
            for img in payload_images
        ]
        return parts, effective_prompt

    # -- Genel API --------------------------------------------------------

    def generate(
        self,
        images: Sequence[NDArray[np.uint8]],
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Kareler + prompt için ham metin üretir (bkz. :meth:`VLMBackend.generate`)."""
        self._verify_model_once()

        image_parts, effective_prompt = self._build_image_content(list(images), prompt)
        content: List[Dict[str, Any]] = [{"type": "text", "text": effective_prompt}]
        content.extend(image_parts)

        payload = self._build_payload(self.model_name, content, temperature, max_tokens)
        return self._post_chat(payload)

    def generate_video(
        self,
        video_path: str | Path,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Videoyu doğrudan modele göndererek ham metin üretir.

        Kare örneklemeye alternatif yoldur: sağlayıcının video-özel modeli
        (EVREN'de ``vlm``) zamansal bağlamı karelerden daha iyi değerlendirir.
        :attr:`video_model` bu çağrıda kullanılır.

        Args:
            video_path: Gönderilecek video dosyası.
            prompt: Modele verilecek talimat.
            temperature: Örnekleme sıcaklığı; ``None`` ise varsayılan.
            max_tokens: Azami üretim token'ı; ``None`` ise varsayılan.

        Returns:
            Modelin ham metin yanıtı.

        Raises:
            FileNotFoundError: Dosya yoksa.
            RuntimeError: Dosya gövde sınırını aşarsa veya istek başarısız olursa.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video bulunamadı: {path}")

        raw = path.read_bytes()
        if len(raw) > _RAW_VIDEO_LIMIT_BYTES:
            raise RuntimeError(
                f"{path.name} {len(raw) / 1e6:.1f} MB — base64 kodlamasından sonra "
                f"{_BODY_LIMIT_BYTES / 1e6:.0f} MB gövde sınırını aşar. Klip bölünmeli "
                f"veya yeniden kodlanmalıdır."
            )

        video_b64 = base64.b64encode(raw).decode("utf-8")
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            },
        ]
        payload = self._build_payload(self.video_model, content, temperature, max_tokens)
        return self._post_chat(payload)


class ServerBackend(_OpenAICompatibleBackend):
    """Harici OpenAI-uyumlu çıkarım servisi (TEKNOFEST EVREN dahil).

    Anahtar yapılandırmaya yazılmaz; :attr:`~src.config.ServerConfig.api_key_env`
    ile belirtilen ortam değişkeninden okunur (bkz. ``.env``).
    """

    backend_name = "server"

    @classmethod
    def from_config(cls, config: VLMConfig) -> "ServerBackend":
        """``config.vlm.server`` bloğundan backend oluşturur."""
        cfg = config.server
        api_key = os.environ.get(cfg.api_key_env, "").strip()
        if not api_key:
            logger.warning(
                f"'{cfg.api_key_env}' ortam değişkeni tanımlı değil; istek kimlik "
                f"doğrulaması olmadan gönderilecek. Kimlik doğrulaması gerektiren "
                f"servisler HTTP 401 döndürür."
            )
        return cls(
            base_url=cfg.base_url,
            model_name=cfg.model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_sec=cfg.timeout_sec,
            api_key=api_key or None,
            max_images_per_request=cfg.max_images_per_request,
            grid_columns=cfg.grid_columns,
            video_model=cfg.video_model,
            enable_thinking=cfg.enable_thinking,
        )


class LlamaCppBackend(_OpenAICompatibleBackend):
    """Yerel ``llama-server`` (GGUF model) üzerinden çıkarım.

    ``llama-server`` OpenAI-uyumlu bir arayüz sunduğundan istek mantığı
    :class:`_OpenAICompatibleBackend` ile aynıdır; yalnızca yapılandırma
    kaynağı (``config.vlm.llama_cpp``) farklıdır.

    Not: Yerel sunucunun ayakta olması gerekir; bu yol bu depoda EVREN
    sağlayıcısı kadar doğrulanmamıştır.
    """

    backend_name = "llama_cpp"

    @classmethod
    def from_config(cls, config: VLMConfig) -> "LlamaCppBackend":
        """``config.vlm.llama_cpp`` bloğundan backend oluşturur."""
        cfg = config.llama_cpp
        return cls(
            base_url=cfg.base_url,
            model_name=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_sec=getattr(cfg, "timeout_sec", 1800),
            api_key=None,
            # Yerel llama-server tek görüntüyle en güvenilir çalışır; fazlası
            # grid'e indirilir.
            max_images_per_request=1,
            grid_columns=config.server.grid_columns,
            # chat_template_kwargs bir vLLM uzantısıdır; llama-server bunu
            # tanımadığı için gönderilmez.
            enable_thinking=None,
        )


# ---------------------------------------------------------------------------
# Süreç-içi (yerel) backend'ler
# ---------------------------------------------------------------------------

class VLLMBackend(VLMBackend):
    """Süreç içinde vLLM ile çıkarım (çevrimdışı, yerel GPU).

    Model ilk :meth:`generate` çağrısında yüklenir; backend nesnesi
    oluşturulduğunda bellek tüketilmez.

    Not: Bu yol yerel GPU ve indirilmiş model ağırlığı gerektirir; bu depoda
    EVREN sağlayıcısı kadar doğrulanmamıştır.
    """

    backend_name = "vllm"

    def __init__(self, config: VLMConfig):
        self.cfg = config.vllm
        self.grid_columns = config.server.grid_columns
        self._llm: Any = None

    def _load(self) -> Any:
        if self._llm is None:
            from vllm import LLM

            self._llm = LLM(
                model=self.cfg.model,
                tensor_parallel_size=self.cfg.tensor_parallel_size,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                max_model_len=self.cfg.max_model_len,
                dtype=self.cfg.dtype,
                trust_remote_code=self.cfg.trust_remote_code,
            )
        return self._llm

    def generate(
        self,
        images: Sequence[NDArray[np.uint8]],
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Bkz. :meth:`VLMBackend.generate`."""
        from PIL import Image
        from vllm import SamplingParams

        llm = self._load()
        sampling = SamplingParams(
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=self.cfg.max_new_tokens if max_tokens is None else max_tokens,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
        )

        effective_prompt = prompt
        request: Dict[str, Any] = {}
        if images:
            # Süreç içi çok modlu çağrıda tek bir görüntü beklenir; çok kare
            # gridle tek görüntüye indirilir.
            if len(images) == 1:
                merged = images[0]
            else:
                merged, rows, cols = build_frame_grid(images, self.grid_columns)
                effective_prompt = _grid_note(rows, cols, len(images)) + prompt
            request["multi_modal_data"] = {"image": Image.fromarray(np.ascontiguousarray(merged))}

        request["prompt"] = f"USER: <image>\n{effective_prompt}\nASSISTANT:" if images else effective_prompt
        outputs = llm.generate([request], sampling)
        return outputs[0].outputs[0].text


class TransformersBackend(VLMBackend):
    """Süreç içinde HuggingFace ``transformers`` ile çıkarım.

    Not: Bu yol yerel model indirmesi gerektirir ve yavaştır; bu depoda EVREN
    sağlayıcısı kadar doğrulanmamıştır.
    """

    backend_name = "transformers"

    def __init__(self, config: VLMConfig):
        self.cfg = config.transformers
        self.grid_columns = config.server.grid_columns
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = getattr(torch, self.cfg.torch_dtype, torch.float32)
        self._processor = AutoProcessor.from_pretrained(self.cfg.model)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.model,
            torch_dtype=dtype,
            attn_implementation=self.cfg.attn_implementation,
            device_map="auto",
        )

    def generate(
        self,
        images: Sequence[NDArray[np.uint8]],
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Bkz. :meth:`VLMBackend.generate`."""
        from PIL import Image

        self._load()

        effective_prompt = prompt
        pil_images: List[Any] = []
        if images:
            if len(images) == 1:
                merged = images[0]
            else:
                merged, rows, cols = build_frame_grid(images, self.grid_columns)
                effective_prompt = _grid_note(rows, cols, len(images)) + prompt
            pil_images = [Image.fromarray(np.ascontiguousarray(merged))]

        content: List[Dict[str, Any]] = []
        if pil_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": effective_prompt})

        chat_prompt = self._processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True
        )
        inputs = self._processor(
            text=chat_prompt,
            images=pil_images or None,
            return_tensors="pt",
        ).to(self._model.device)

        output = self._model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens if max_tokens is None else max_tokens,
            temperature=0.15 if temperature is None else temperature,
            repetition_penalty=self.cfg.repetition_penalty,
            do_sample=True,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._processor.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Backend kimliği -> oluşturucu. ``create_backend`` bu haritayı kullanır.
_BUILDERS: Dict[str, Any] = {
    "server": ServerBackend.from_config,
    "llama_cpp": LlamaCppBackend.from_config,
    "vllm": VLLMBackend,
    "transformers": TransformersBackend,
}


def create_backend(config: VLMConfig, force: str | None = None) -> VLMBackend:
    """Yapılandırmaya göre uygun çıkarım backend'ini oluşturur.

    Bu, çıkarım katmanının tek giriş noktasıdır; çağıranlar hangi sağlayıcının
    kullanıldığını bilmeden aynı arayüzle çalışır.

    Seçim sırası:

    1. ``force`` verilmişse (``--backend`` bayrağı) yalnızca o denenir ve
       başarısız olursa hata yükseltilir — kullanıcı açıkça bir sağlayıcı
       istediğinde sessizce başkasına düşmek yanıltıcı olur.
    2. ``default_backend`` ``"auto"`` dışında bir değerse o kullanılır.
    3. ``"auto"`` ise ``auto_preference`` sırası denenir; ilk başarıyla
       oluşturulan backend döner.

    Args:
        config: ``config.vlm`` bloğu.
        force: İstenen backend kimliği veya ``None``.

    Returns:
        Kullanıma hazır :class:`VLMBackend`.

    Raises:
        ValueError: Bilinmeyen bir backend adı istenirse.
        RuntimeError: Hiçbir backend oluşturulamazsa.
    """
    if force:
        if force not in _BUILDERS:
            raise ValueError(
                f"Bilinmeyen backend: {force!r}. Geçerli değerler: {', '.join(_BUILDERS)}"
            )
        backend = _BUILDERS[force](config)
        logger.info(f"VLM backend oluşturuldu (açıkça istendi): {backend.name()}")
        return backend

    requested = config.default_backend
    candidates: List[str] = (
        list(config.auto_preference) if requested == "auto" else [requested]
    )

    errors: List[str] = []
    for candidate in candidates:
        builder = _BUILDERS.get(candidate)
        if builder is None:
            errors.append(f"{candidate}: bilinmeyen backend adı")
            continue
        try:
            backend = builder(config)
        except Exception as exc:  # sıradaki adaya düşülmesi beklenen durum
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            logger.warning(f"'{candidate}' backend'i oluşturulamadı ({exc}); sıradaki deneniyor.")
            continue
        logger.info(f"VLM backend oluşturuldu: {backend.name()}")
        return backend

    raise RuntimeError(
        "Hiçbir VLM backend'i oluşturulamadı. Denenenler:\n  " + "\n  ".join(errors)
    )
