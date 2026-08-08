#!/usr/bin/env python3
"""4 ayrı klasörden rastgele video seçip src.main akışını konsolda canlı gösteren test betiği.

Beklenen yapı:
    /videos/
        klasor_1/*.mp4
        klasor_2/*.mp4
        klasor_3/*.mp4
        klasor_4/*.mp4

Varsayılan davranış:
    1. /videos altındaki klasörleri bulur.
    2. İlk 4 klasör yerine rastgele 4 klasör seçer.
    3. Her klasörden rastgele 1 video seçer.
    4. Seçilen videolar için şu komutu çalıştırır:

        ~/.venvs/nlp2026/bin/python -m src.main --video <video> --detector ultralytics --backend server --save-grid

Çıktıyı satır satır konsola basar; böylece her aşama izlenebilir.
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def log(title: str) -> None:
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")


def list_videos(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def choose_random_videos(videos_root: Path, folder_count: int = 4) -> list[Path]:
    if not videos_root.exists():
        raise FileNotFoundError(f"Video kökü bulunamadı: {videos_root}")

    subfolders = [path for path in videos_root.iterdir() if path.is_dir()]
    if len(subfolders) < folder_count:
        raise RuntimeError(
            f"En az {folder_count} klasör gerekli. Bulunan klasör sayısı: {len(subfolders)}"
        )

    chosen_folders = random.sample(sorted(subfolders), folder_count)
    selected_videos: list[Path] = []

    print("Seçilen klasörler:")
    for folder in chosen_folders:
        videos = list_videos(folder)
        if not videos:
            print(f"  - [ATLANDI] {folder} içinde video yok")
            continue

        chosen_video = random.choice(videos)
        selected_videos.append(chosen_video)
        print(f"  - {folder.name}: {chosen_video.name}")

    if len(selected_videos) != folder_count:
        raise RuntimeError(
            f"{folder_count} video seçilemedi. Seçilen video sayısı: {len(selected_videos)}"
        )

    return selected_videos


def resolve_videos_root(explicit_root: Path | None, repo_root: Path) -> Path:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root)
    candidates.extend(
        [
            repo_root / "videos",
            repo_root.parent / "videos",
            repo_root.parent / "INDIR_BENCHMARK" / "videos",
            repo_root.parent.parent / "INDIR_BENCHMARK" / "videos",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = "\n  - ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"Video kökü bulunamadı. Denenen yollar:\n  - {tried}")


def stream_process(command: list[str], cwd: Path) -> int:
    print("\nÇalıştırılan komut:")
    print(" ".join(command))

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    return process.wait()


def preflight_python_env(python_bin: Path) -> None:
    check_code = (
        "import importlib.util as u; "
        "missing=[m for m in ['numpy','av','PIL','huggingface_hub'] if u.find_spec(m) is None]; "
        "raise SystemExit('MISSING:' + ','.join(missing)) if missing else None"
    )
    result = subprocess.run(
        [str(python_bin), "-c", check_code],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stdout.strip() or result.stderr.strip() or "bilinmeyen hata"
        raise RuntimeError(
            "Seçilen Python ortamı eksik bağımlılıklar içeriyor. "
            f"Durum: {message}. "
            "Bu betik, src.main'i çalıştırmadan önce numpy/av/PIL/huggingface_hub kontrolü yapıyor."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video klasörlerini otomatik keşfedip dört rastgele video üzerinden src.main akışını çalıştırır."
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=None,
        help="İçinde video klasörleri bulunan kök dizin. Verilmezse otomatik aranır.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Proje kökü",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path.home() / ".venvs" / "nlp2026" / "bin" / "python",
        help="Kullanılacak Python yorumlayıcısı",
    )
    parser.add_argument(
        "--backend",
        default="server",
        help="src.main için VLM backend parametresi",
    )
    parser.add_argument(
        "--detector",
        default="ultralytics",
        help="src.main için detector parametresi",
    )
    parser.add_argument(
        "--save-grid",
        action="store_true",
        default=True,
        help="VLM'e gönderilen grid'i kaydet (varsayılan: açık)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Tekrarlanabilir seçim için rastgele tohum",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sadece seçimleri ve çalıştırılacak komutları yazdır; src.main'i çalıştırma",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    repo_root = args.repo_root.resolve()
    videos_root = resolve_videos_root(args.videos_root, repo_root)
    python_bin = args.python_bin.resolve()

    log("AŞAMA 1 — Video klasörleri taranıyor")
    print(f"Video kökü: {videos_root}")
    print(f"Repo kökü : {repo_root}")
    print(f"Python    : {python_bin}")

    try:
        preflight_python_env(python_bin)
        print("Python ortamı: gerekli temel paketler bulundu")
    except Exception as exc:
        print(f"[HATA] {exc}")
        raise SystemExit(1)

    selected_videos = choose_random_videos(videos_root, folder_count=4)

    log("AŞAMA 2 — Seçilen videolar")
    for index, video in enumerate(selected_videos, start=1):
        print(f"{index}. {video}")

    if args.dry_run:
        log("DRY-RUN — src.main çalıştırılmadı")
        for index, video in enumerate(selected_videos, start=1):
            command = [
                str(python_bin),
                "-m",
                "src.main",
                "--video",
                str(video),
                "--detector",
                args.detector,
                "--backend",
                args.backend,
            ]
            if args.save_grid:
                command.append("--save-grid")
            print(f"[{index}] {' '.join(command)}")
        raise SystemExit(0)

    failures = 0
    for index, video in enumerate(selected_videos, start=1):
        log(f"AŞAMA 3.{index} — src.main çalıştırılıyor")
        print(f"Video: {video}")

        command = [
            str(python_bin),
            "-m",
            "src.main",
            "--video",
            str(video),
            "--detector",
            args.detector,
            "--backend",
            args.backend,
        ]
        if args.save_grid:
            command.append("--save-grid")

        exit_code = stream_process(command, cwd=repo_root)
        if exit_code != 0:
            failures += 1
            print(f"\n[HATA] Çıkış kodu: {exit_code}")
        else:
            print("\n[OK] Çalışma tamamlandı")

    log("ÖZET")
    print(f"Toplam video: {len(selected_videos)}")
    print(f"Başarısız: {failures}")
    print(f"Başarılı : {len(selected_videos) - failures}")

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()