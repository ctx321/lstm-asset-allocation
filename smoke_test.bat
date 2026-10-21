@echo off
rem Contract smoke test against the locally running API.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py tools\api_smoke_test.py --url http://127.0.0.1:8000/predict
) else (
    "C:\Users\CTX\AppData\Local\Python\bin\python.exe" tools\api_smoke_test.py --url http://127.0.0.1:8000/predict
)
