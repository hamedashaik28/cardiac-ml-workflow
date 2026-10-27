@echo off
setlocal
echo =======================================================
echo   Cardiac ML Workflow: Pytest Test Suite
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

echo Running unit tests...
"%PY_CMD%" -m pytest -v
pause
