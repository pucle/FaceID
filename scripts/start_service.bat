@echo off
REM Chay dich vu diem danh (Windows). Dat PORTAL_TOKEN bang `setx PORTAL_TOKEN "..."` (cua so moi) hoac sua dong duoi.
cd /d "%~dp0.."
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
:loop
python attendance_service.py --pilot-dir pilot %*
echo [%date% %time%] service dung, khoi dong lai sau 10 giay... >> data\restart.log
timeout /t 10 /nobreak >nul
goto loop
