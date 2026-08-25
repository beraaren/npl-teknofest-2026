"""RLHF / DPO Veri Seti Dışa Aktarma Scripti.

Süpervizör arayüzünden toplanan insan geri bildirimlerini (Human-in-the-Loop)
Hugging Face TRL / DPO standartlarına uygun `{"prompt": ..., "chosen": ..., "rejected": ...}`
JSONL formatında dışa aktarır.

Kullanım:
    python scripts/export_dpo_dataset.py --output outputs/dpo_dataset.jsonl
    python scripts/export_dpo_dataset.py --only-corrections --output outputs/dpo_corrections.jsonl
    python scripts/export_dpo_dataset.py --stats
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.gateway import store


def main():
    parser = argparse.ArgumentParser(
        description="Süpervizör geri bildirimlerini RLHF/DPO eğitim veri setine dönüştürür."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/dpo_dataset.jsonl",
        help="Çıktı JSONL dosya yolu (varsayılan: outputs/dpo_dataset.jsonl)",
    )
    parser.add_argument(
        "--only-corrections",
        action="store_true",
        help="Yalnızca süpervizörün düzelttiği hatalı kararları (chosen != rejected) dışa aktar",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Veri seti istatistiklerini ekrana yazdırır",
    )

    args = parser.parse_args()

    # DB'yi başlat (yoksa tablolar oluşturulsun)
    store.init_db()

    stats = store.get_feedback_stats()
    total = stats.get("total", 0)

    print("==================================================")
    print("🎯 RLHF / DPO Human-in-the-Loop Veri Seti Aracı")
    print("==================================================")
    print(f"Toplam Geri Bildirim: {total}")
    print(f"Doğrulanmış Kararlar: {stats.get('correct_count', 0)}")
    print(f"Düzeltilmiş Kararlar: {stats.get('correction_count', 0)}")
    print(f"Model Başarı Oranı : %{stats.get('accuracy_rate', 100.0)}")
    print("Geri Bildirim Dağılımı:")
    for k, v in stats.get("by_type", {}).items():
        print(f"  • {k}: {v}")
    print("==================================================")

    if args.stats and not args.output:
        return

    records = store.export_dpo_dataset_records(only_corrections=args.only_corrections)
    if not records:
        print("ℹ️ Dışa aktarılacak geri bildirim kaydı bulunamadı.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ {len(records)} adet DPO tercih çifti başarıyla yazıldı:")
    print(f"📁 {out_path.resolve()}")


if __name__ == "__main__":
    main()
