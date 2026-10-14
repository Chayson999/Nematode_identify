# -*- coding: utf-8 -*-
"""用 GPU 训练线虫检测模型（Ultralytics YOLO）。

准备：
  1) 安装 GPU 版 PyTorch：运行 release 根目录的 1-安装依赖(GPU版).bat
  2) 准备 YOLO 格式数据集：见 backend/docs/training.md

用法示例：
  python tools/train_yolo.py --data D:\data\worm\data.yaml --epochs 100 --imgsz 960 --batch 8
训练结束自动把 best.pt 复制到 backend/models/best.pt（加 --no-install 可关闭）。
"""
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description='训练线虫检测模型（GPU）')
    ap.add_argument('--data', required=True, help='数据集 data.yaml 路径')
    ap.add_argument('--model', default='yolov8n.pt', help='初始权重（默认 yolov8n.pt，首次自动下载）')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=960, help='训练分辨率；显存不足可降到 640')
    ap.add_argument('--batch', type=int, default=8, help='批大小；显存不足可降到 4')
    ap.add_argument('--device', default='0', help='0=第一块 GPU，cpu=用 CPU 训练')
    ap.add_argument('--project', default=str(BACKEND_DIR / 'runs' / 'train'))
    ap.add_argument('--name', default='worm')
    ap.add_argument('--no-install', action='store_true', help='不自动替换 backend/models/best.pt')
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print('[错误] 找不到数据集配置：%s' % args.data)
        sys.exit(1)

    import torch
    if args.device != 'cpu':
        if not torch.cuda.is_available():
            print('[错误] 当前 PyTorch 不支持 CUDA（装的是 CPU 版）。')
            print('       请先运行 release 根目录的 1-安装依赖(GPU版).bat 后重试。')
            sys.exit(1)
        print('[信息] 使用 GPU：%s，显存 %.1f GB' % (
            torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    else:
        print('[信息] 使用 CPU 训练（会很慢）')

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    best = sorted(glob.glob(os.path.join(args.project, args.name, 'weights', 'best.pt')))
    if not best:
        print('[警告] 训练结束但没有找到 best.pt，请检查 runs 目录')
        return
    best_path = best[-1]
    print('[完成] 最佳权重：%s' % best_path)
    if not args.no_install:
        target = BACKEND_DIR / 'models' / 'best.pt'
        if target.exists():
            shutil.copy2(str(target), str(target) + '.bak')
            print('[提示] 原有权重已备份为 best.pt.bak')
        shutil.copy2(best_path, str(target))
        print('[完成] 已替换 %s，重启服务即生效' % target)


if __name__ == '__main__':
    main()
