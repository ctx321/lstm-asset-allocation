@echo off
rem Stop any process listening on port 8000 (the LSTM API).
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ('Stopping PID ' + $_.OwningProcess); Stop-Process -Id $_.OwningProcess -Force }"
echo Port 8000 is now free.
