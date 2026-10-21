@echo off
rem Start the LSTM API locally. Uses the "py" launcher because the bare
rem "python" command may be a Microsoft Store alias that exits silently.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -m uvicorn api.app:app --host 0.0.0.0 --port 8000
) else (
    "C:\Users\CTX\AppData\Local\Python\bin\python.exe" -m uvicorn api.app:app --host 0.0.0.0 --port 8000
)
