@echo off
REM Quick setup script for BanglaRAG LMS Chatbot

echo ========================================
echo BanglaRAG LMS Chatbot - Quick Setup
echo ========================================
echo.

echo [1/3] Installing dependencies...
pip install flask flask-cors requests
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo [2/3] Loading course knowledge base...
python "%~dp0load_course_database.py"
if errorlevel 1 (
    echo ERROR: Failed to load course database
    pause
    exit /b 1
)
echo.

echo [3/3] Setup complete!
echo.
echo ========================================
echo Next Steps:
echo ========================================
echo.
echo 1. Start the API server:
echo    python "%~dp0chatbot_api.py"
echo.
echo 2. Open the course page:
echo    %~dp0course-example.html
echo.
echo ========================================
echo.
pause
