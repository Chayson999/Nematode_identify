@echo off
setlocal
cd /d "%~dp0backend"
title 线虫识别系统 - 安装依赖

echo ============================================================
echo   第 1 步：安装运行依赖（首次使用只需做一次）
echo   需要联网，约 400 MB，请耐心等待 5~15 分钟
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 没有检测到 Python。
    echo        请先安装 Python 3.10 或更高版本：https://www.python.org/downloads/
    echo        安装时务必勾选 "Add python.exe to PATH"，装完重新运行本脚本。
    echo.
    pause
    exit /b 1
)

echo [1/4] 创建独立虚拟环境 .venv ...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败，请确认 Python 安装完整。
        pause
        exit /b 1
    )
)
set PY=.venv\Scripts\python.exe

echo [2/4] 升级 pip ...
%PY% -m pip install --upgrade pip

echo [3/4] 安装 CPU 版 PyTorch（约 200 MB，无需显卡）...
%PY% -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo [4/4] 安装其余依赖 ...
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败。可改用国内源重试：
    echo   %PY% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   安装完成！接下来双击 "2-启动.bat" 即可使用。
echo ============================================================
pause
