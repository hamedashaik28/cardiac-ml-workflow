@echo off
setlocal
echo =======================================================
echo   Cardiac ML Workflow: Model Training & Evaluation
echo =======================================================

:: Check if python is in PATH, otherwise use Anaconda path
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

echo Using Python: %PY_CMD%
"%PY_CMD%" src/train.py

echo.
echo Training complete. Artifacts saved in models/ and reports/.
pause
