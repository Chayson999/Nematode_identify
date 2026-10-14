@echo off
setlocal
cd /d "%~dp0frontend-src"
title 线虫识别系统 - 前端开发模式（需要 Node.js）

where npm >nul 2>nul
if errorlevel 1 (
    echo [错误] 没有检测到 Node.js / npm。仅做二次开发时才需要：
    echo        https://nodejs.org/  （建议 LTS 18 或 20）
    pause
    exit /b 1
)
echo 注意：请先用 "2-启动.bat" 启动后端（5000 端口）。
if not exist node_modules (
    echo 首次运行需要安装前端依赖 npm install ...
    call npm install
)
set REACT_APP_BACKEND_URL=http://127.0.0.1:5000
start "" http://localhost:3000
call npm start
