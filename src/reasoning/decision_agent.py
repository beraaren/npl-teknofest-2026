import json
import logging
from typing import Any, Dict, List

# import logging
logger = logging.getLogger(__name__)

class DecisionAgent:
    """
    TEKNOFEST Faz 05: Karar Ajanı (VLM Backend).
    
    Bu ajan, ObserverAgent'tan gelen görüntü (veya frame analizi) ile 
    RAGLayer'dan gelen bağlamı (kurallar, tehlikeler, mock_tool_hints) alarak 
    nihai kararı verir. Kararı her zaman JSON (Structured Output) formatında döndürür.
    
    Eğer VLM hata yaparsa veya çok yavaşlarsa, Fallback olarak RAGLayer'ın
    önerdiği tool'ları direkt çalıştırır.
    """

    def __init__(self, model_name: str = "local-vlm-10b"):
        self.model_name = model_name
        self.system_prompt = (
            "Sen TEKNOFEST Endüstriyel Tesis İSG Uzmanı bir yapay zeka ajanısın. "
            "Gördüğün resim ve sana sunulan RAG raporuna göre durum değerlendirmesi yap ve aksiyon seç. "
            "Hayal kurma, sadece sana verilen RAG (Kurallar) veritabanına sadık kal. "
            "Çıktını KESİNLİKLE aşağıdaki JSON formatında ver:\n"
            '{"reasoning": "...", "risk_level": "Yüksek", "actions": [{"tool": "tool_adi", "params": {}}]}'
        )
        logger.info(f"DecisionAgent başlatıldı. Model: {self.model_name}")

    def decide(self, rag_context: Dict[str, Any], image_path: str = None) -> Dict[str, Any]:
        """
        Görüntü ve RAG içeriğine bakarak nihai kararı (Structured Output) verir.
        
        Args:
            rag_context: RAGLayer.build_context() çıktısı.
            image_path: İncelenecek kameranın anlık frame'i.
            
        Returns:
            JSON formatında VLM kararı.
        """
        
        # 1. Fallback / Otonom Güvenlik Kontrolü
        # VLM API'sini mockluyoruz. Gerçek model (Llama-3-Vision vb.) burada çağırılır.
        vlm_response = self._call_vlm_api(rag_context, image_path)
        
        if vlm_response:
            try:
                # JSON formatını doğrula
                decision = json.loads(vlm_response)
                if "actions" in decision and "risk_level" in decision:
                    logger.info("VLM kararı başarıyla alındı.")
                    return decision
            except json.JSONDecodeError:
                logger.warning("VLM JSON üretemedi. Fallback mekanizması devreye giriyor.")
                
        # 2. VLM Çöktüyse veya Hata Yaptıysa -> RAG Fallback
        logger.warning("VLM API yanıt vermedi veya format hatalı. RAG fallback araçları kullanılıyor.")
        fallback_actions = self._extract_fallback_actions(rag_context)
        
        return {
            "reasoning": "VLM bypass edildi. Karar doğrudan RAG kural motorundan (Fallback) otonom olarak alındı.",
            "risk_level": self._calculate_max_risk(rag_context),
            "actions": fallback_actions
        }

    def _call_vlm_api(self, rag_context: Dict[str, Any], image_path: str) -> str | None:
        """
        10B parametreli lokal modele (vLLM veya llama.cpp) istek atacak fonksiyon.
        Şimdilik Mock/Placeholder.
        """
        # TODO: integrate huggingface transformers or vLLM here.
        # Example pseudo-code:
        # prompt = f"{self.system_prompt}\n\nRAG Context:\n{json.dumps(rag_context)}"
        # response = self.model.generate(prompt, image=image_path)
        
        # Simüle edilmiş (Mock) VLM cevabı:
        return None  # Fallback'i test etmek için None dönüyoruz.

    def _extract_fallback_actions(self, rag_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """RAG'dan önerilen araçları (mock_tool_hints) çıkarır."""
        # Şimdilik temsili, rag_layer.py içinde recommend_tools var.
        # Ajan bu fonksiyonu kullanarak rag_layer'ın recommend_tools çıktısını doğrudan paslar.
        return rag_context.get("recommended_tools", [{"tool": "notify_supervisor", "params": {"reason": "Fallback"}}])
        
    def _calculate_max_risk(self, rag_context: Dict[str, Any]) -> str:
        # RAG context içindeki patternlerden en yüksek riski bul
        return "Yüksek" # Placeholder
