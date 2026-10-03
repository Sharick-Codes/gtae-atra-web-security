@echo off
title GTAE-IDS + ATRA Web Dashboard & Terminal Runner
echo ======================================================================
echo   Starting GTAE-IDS + ATRA Web Dashboard & Terminal Engine...
echo ======================================================================
cd /d %~dp0
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
python app.py
pause

