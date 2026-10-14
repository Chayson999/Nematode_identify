@echo off
setlocal
cd /d "%~dp0backend"
title 线虫识别系统 - 服务运行中（关闭本窗口即停止）

if exist ".venv\Scripts\python.exe" (set PY=.venv\Scripts\python.exe) else (set PY=python)
if "%PORT%"=="" set PORT=5000
if "%HOST%"=="" set HOST=127.0.0.1

echo ============================================================
echo   线虫识别系统启动中 ...
echo   页面地址：http://127.0.0.1:%PORT%
echo   首次启动需加载模型（约 10~30 秒），稍后会弹出浏览器
echo   停止服务：直接关闭本窗口
echo ============================================================
echo.

start "" /b cmd /c "timeout /t 20 /nobreak >nul && start http://127.0.0.1:%PORT%"

%PY% app.py
pause
