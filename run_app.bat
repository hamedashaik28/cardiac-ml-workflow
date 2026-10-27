@echo off
setlocal
echo =======================================================
echo   Cardiac ML Workflow: Streamlit Clinical Dashboard
echo =======================================================

where python >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=python"
) else if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PY_CMD=%USERPROFILE%\anaconda3\python.exe"
) else (
    echo [ERROR] Python not found in PATH or at %USERPROFILE%\anaconda3\python.exe
    pause
    exit /b 1
)

echo Starting Streamlit application...
"%PY_CMD%" -m streamlit run app.py
pause
