@echo off
setlocal
cd /d "%~dp0backend"
title 线虫识别系统 - 安装 GPU 训练依赖

echo ============================================================
echo   可选步骤：安装 GPU 版 PyTorch（用来自己训练模型）
echo   要求：NVIDIA 显卡 + 驱动，首次约 2~3 GB 下载
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 没有检测到 Python，请先安装 Python 3.10 以上版本。
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo 创建虚拟环境 .venv ...
    python -m venv .venv
)
set PY=.venv\Scripts\python.exe

echo [1/3] 升级 pip ...
%PY% -m pip install --upgrade pip

echo [2/3] 安装 CUDA 版 PyTorch（cu128，适配 RTX 50 系 / Blackwell）...
%PY% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo [3/3] 安装其余依赖 ...
%PY% -m pip install -r requirements.txt

echo.
echo 检查 GPU 是否可用 ...
%PY% -c "import torch;print('CUDA 可用:', torch.cuda.is_available());print('显卡:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"

echo.
echo ============================================================
echo   安装完成。下一步：双击 "5-训练模型.bat" 开始训练
echo ============================================================
pause
