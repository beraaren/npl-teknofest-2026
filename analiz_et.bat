@echo off
REM Videos klasorune eklenen yeni videolari analiz eder ve JSON uretir.
REM Kullanim: Bu dosyaya cift tiklamak yeterlidir; terminal komutu yazmaya
REM gerek yoktur. Zaten analiz edilmis videolar tekrar islenmez.
REM
REM Not: EVREN_API_KEY burada YAZILMAZ; venv\..env dosyasindan otomatik
REM okunur (scripts\analyze_video_library.py -> python-dotenv ile).

cd /d "%~dp0"
echo Video kutuphanesi analiz ediliyor...
echo.
venv\Scripts\python.exe scripts\analyze_video_library.py
echo.
echo Bitti. Bu pencereyi kapatabilirsiniz.
pause
