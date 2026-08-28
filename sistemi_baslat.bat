@echo off
REM Gateway'i (UI + API) baslatir. Cift tikla, tarayicida ac:
REM   http://127.0.0.1:8000        (supervizor ekrani)
REM   http://127.0.0.1:8000/saha.html   (saha ekibi ekrani)
REM
REM Kapatmak icin bu pencerede Ctrl+C, ardindan bir tusa basin.
REM
REM Not: EVREN_API_KEY burada YAZILMAZ; .env dosyasindan otomatik okunur
REM (src\config.py -> load_config -> python-dotenv).

cd /d "%~dp0"
echo Gateway baslatiliyor... (kapatmak icin Ctrl+C)
echo Hazir olunca tarayicida http://127.0.0.1:8000 adresini acin.
echo.
python -m uvicorn backend.gateway.main:app --host 127.0.0.1 --port 8000
pause
