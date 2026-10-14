@echo off
setlocal
cd /d "%~dp0backend"
title 线虫识别系统 - 训练模型

if exist ".venv\Scripts\python.exe" (set PY=.venv\Scripts\python.exe) else (set PY=python)

echo ============================================================
echo   训练线虫检测模型（需要 GPU 依赖，见 4-安装GPU训练依赖.bat）
echo ============================================================
echo.
echo 数据集要求：YOLO 格式，目录里有 data.yaml
echo   示例： D:\data\worm\data.yaml
echo.

set /p DATA=请粘贴 data.yaml 的完整路径，然后回车：
if "%DATA%"=="" (
    echo [取消] 没有输入路径。
    pause
    exit /b 1
)

echo.
echo 开始训练：epochs=100, imgsz=960, batch=8（显存不足可用记事本改本文件改小）
echo.
%PY% tools\train_yolo.py --data "%DATA%" --epochs 100 --imgsz 960 --batch 8 --device 0

echo.
echo 训练流程结束。如果成功，backend\models\best.pt 已更新，重启服务即生效。
pause
