@echo off
title GTAE-ATRA Web Security Suite (Production)
cls

:: ── Read API key from logs/.api_key ──────────────────────
set API_KEY_FILE=logs\.api_key
if exist %API_KEY_FILE% (
    set /p MONITOR_API_KEY=<%API_KEY_FILE%
) else (
    echo [!] API key not found. Start the monitor server first to generate one.
    set MONITOR_API_KEY=
)

echo ======================================================================
echo   GTAE-ATRA-NIDS: Real-Time Web Security Monitoring (PRODUCTION)
echo ======================================================================
echo   1. Start All (Monitor Server + Web App + React Dashboard)
echo   2. Start Python ML Monitor Server (Port 8765)
echo   3. Start Express Demo Web Application (Port 3000)
echo   4. Start React Security Operations Dashboard (Port 3001)
echo   5. Train / Retrain Web Security ML Models (GTAE + ATRA)
echo   6. Run Attack Traffic Simulation (Test ATRA Response)
echo   7. Run Normal Traffic Simulation
echo   8. Start Legacy CIC-IDS2017 Flask Dashboard (Port 5000)
echo   9. Show API Key (for integrating external websites)
echo   0. Exit
echo ======================================================================
if defined MONITOR_API_KEY (
    echo   API Key: %MONITOR_API_KEY:~0,8%...
) else (
    echo   API Key: [not generated yet - start monitor server first]
)
echo ======================================================================
set /p opt="Select an option (0-9): "

if "%opt%"=="1" (
    echo Starting Python ML Monitor Server...
    start "GTAE-ATRA Monitor Server (8765)" cmd /k "python src/web_monitor_server.py"
    timeout /t 4 /nobreak >nul
    :: Re-read API key after server generates it
    if exist %API_KEY_FILE% (
        set /p MONITOR_API_KEY=<%API_KEY_FILE%
    )
    echo Starting Demo Web Application with API key...
    start "Express Web App (3000)" cmd /k "set MONITOR_API_KEY=%MONITOR_API_KEY%&& set MONITOR_SITE_ID=shopnow-demo&& cd webapp && npm start"
    timeout /t 2 /nobreak >nul
    echo Starting React Security Dashboard...
    start "React Security Dashboard (3001)" cmd /k "cd dashboard && npm start"
    echo.
    echo ======================================================================
    echo   All services started in separate windows!
    echo   API Key: %MONITOR_API_KEY%
    echo ======================================================================
    pause
    goto end
)
if "%opt%"=="2" (
    python src/web_monitor_server.py
    pause
    goto end
)
if "%opt%"=="3" (
    set MONITOR_API_KEY=%MONITOR_API_KEY%
    set MONITOR_SITE_ID=shopnow-demo
    cd webapp && npm start
    goto end
)
if "%opt%"=="4" (
    cd dashboard && npm start
    goto end
)
if "%opt%"=="5" (
    python src/web_train.py
    pause
    goto end
)
if "%opt%"=="6" (
    python tests/traffic/generate_attack_traffic.py --scenario brute_force
    pause
    goto end
)
if "%opt%"=="7" (
    python tests/traffic/generate_normal_traffic.py --duration 30 --rate 2
    pause
    goto end
)
if "%opt%"=="8" (
    python app.py
    pause
    goto end
)
if "%opt%"=="9" (
    echo.
    echo ======================================================================
    if exist %API_KEY_FILE% (
        type %API_KEY_FILE%
        echo.
    ) else (
        echo   API key not generated yet. Run the monitor server first.
    )
    echo ======================================================================
    echo.
    echo   To integrate any Express.js website, set these env variables:
    echo     MONITOR_URL=http://127.0.0.1:8765
    echo     MONITOR_API_KEY=^<key above^>
    echo     MONITOR_SITE_ID=your-site-name
    echo.
    echo   Then add the middleware:
    echo     const { monitor } = require('./middleware/monitor');
    echo     app.use(monitor());
    echo ======================================================================
    pause
    goto end
)

:end
