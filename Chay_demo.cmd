@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Khong tim thay moi truong Python trong thu muc venv.
    pause
    exit /b 1
)
powershell -NoProfile -Command "try { if ((Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 2).Content -eq 'ok') { exit 0 }; exit 1 } catch { exit 1 }"
if not errorlevel 1 (
    start "" "http://127.0.0.1:8501"
    exit /b 0
)
"venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless false
if errorlevel 1 pause
