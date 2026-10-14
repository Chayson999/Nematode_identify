import os
import shutil
import base64
import csv
import json
import re
import threading
import zipfile
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, join_room
from werkzeug.utils import secure_filename
from worm_processing import (
    init_model,
    process_frame_variants,
    EnhancedWormTracker,
    INFER_IMGSZ,
    SWING_DIAG,
    OVERLAY_MODE_DEBUG,
    OVERLAY_MODE_RELEASE,
)

# ==================== 配置 ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'          # 运行数据统一放在 backend/data 下


def resolve_data_dir(name):
    """返回数据目录：优先 backend/data/<name>，兼容历史 backend/<name> 结构。"""
    new_dir = DATA_DIR / name
    old_dir = BASE_DIR / name
    if new_dir.exists() or not old_dir.exists():
        return str(new_dir.resolve())
    return str(old_dir.resolve())


app = Flask(__name__, static_folder=None)   # 静态资源统一由下方 serve_frontend 托管
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = resolve_data_dir('uploads')
app.config['RESULT_FOLDER'] = resolve_data_dir('results')
app.config['SUB_VIDEO_FOLDER'] = os.path.join(app.config['RESULT_FOLDER'], 'sub_videos')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# ==================== 运动识别配置 ====================
APP_START_TIME = time.time()
TRACKING_CONFIG = {
    # 活性判断参数
    'activity_threshold': 0.25,         # 最小活性分数（0-1），低于此值的目标被过滤
    'activity_window_sec': 1.0,         # 活性评估窗口（秒）
    
    # 运动感知跟踪参数
    'motion_consistency_weight': 0.45,  # 运动一致性权重（提高到45%）
    'iou_weight': 0.25,                 # IoU权重
    'appearance_weight': 0.15,          # 外观特征权重
    'direction_weight': 0.15,           # 方向一致性权重
}

# 画面叠加模式：
#   实时预览（网页“实时处理区”逐帧推送）走 debug —— 残留目标不上屏，画面里只出现真实检测；
#   结果视频（处理完成后下载的主视频）走 release —— 残留目标与真实检测一样画实线框。
# 两者共用同一次推理与跟踪结果，见 worm_processing.process_frame_variants。
PREVIEW_OVERLAY_MODE = OVERLAY_MODE_DEBUG
RESULT_OVERLAY_MODE = OVERLAY_MODE_RELEASE

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['SUB_VIDEO_FOLDER'], exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*")

# ==================== 前端静态页面（可选） ====================
# 若存在前端构建产物（backend/static 或 frontend/build），由后端直接托管。
# 这样使用者只需启动后端、浏览器打开 http://127.0.0.1:5000 即可，无需安装 Node。
FRONTEND_BUILD = None
for _cand in (BASE_DIR / 'static', BASE_DIR.parent / 'frontend' / 'build'):
    if _cand.is_dir() and (_cand / 'index.html').is_file():
        FRONTEND_BUILD = _cand
        break

if FRONTEND_BUILD is not None:

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_frontend(path):
        """托管前端构建产物；未知路径回退到 index.html（单页应用）。"""
        build_root = FRONTEND_BUILD.resolve()
        target = (build_root / path).resolve() if path else None
        if target is not None and str(target).startswith(str(build_root)) and target.is_file():
            return send_from_directory(str(build_root), path)
        return send_from_directory(str(build_root), 'index.html')


# 全局模型（在启动时加载一次）
model, device = None, None

# 存储每个任务的状态（可选）
processing_tasks = {}
task_lock = threading.Lock()


def update_task_state(task_id, **kwargs):
    with task_lock:
        if task_id not in processing_tasks:
            return
        processing_tasks[task_id].update(kwargs)


def get_task_state(task_id):
    with task_lock:
        state = processing_tasks.get(task_id)
        if state is None:
            return None
        return dict(state)


def get_task_control(task_id):
    with task_lock:
        state = processing_tasks.get(task_id)
        if state is None:
            return None
        return state.get('control', 'running')


def wait_or_cancel(task_id):
    """返回 False 表示任务被取消，True 表示可以继续处理。"""
    while True:
        control = get_task_control(task_id)
        if control is None:
            return False
        if control == 'cancelled':
            update_task_state(task_id, status='cancelled', error_message='任务已取消')
            socketio.emit('cancelled', {'task_id': task_id, 'message': '任务已取消'}, room=task_id)
            return False
        if control == 'paused':
            update_task_state(task_id, status='paused')
            time.sleep(0.15)
            continue
        if control == 'running':
            state = get_task_state(task_id)
            if state and state.get('status') == 'paused':
                update_task_state(task_id, status='processing')
            return True
        return True


def build_id_pose_summary(tracker):
    """
    构建ID和姿态摘要
    包含：track_id、周期数、频率、运动类型、活性分数
    """
    summary = []
    for track_id, cycle_info in sorted(tracker.cycle_counts.items(), key=lambda x: int(x[0])):
        avg_freq = cycle_info.get('avg_freq', cycle_info.get('average_frequency', cycle_info.get('freq', 0.0)))
        live_freq = cycle_info.get('live_freq', cycle_info.get('realtime_freq', avg_freq))
        summary.append({
            'track_id': int(track_id),
            'posture_count': int(cycle_info.get('cycles', 0)),
            'swing_count': int(cycle_info.get('swing_count', cycle_info.get('cycles', 0))),
            'swings': int(cycle_info.get('swings', cycle_info.get('cycles', 0))),
            'half_swing_count': int(cycle_info.get('half_swing_count', cycle_info.get('swings', 0))),
            'cycle_count': int(cycle_info.get('cycle_count', 0)),
            'count_mode': str(cycle_info.get('count_mode', 'swing')),
            'observed_seconds': float(cycle_info.get('observed_seconds', 0.0)),
            'swing_freq_hz': float(cycle_info.get('swing_freq_hz', avg_freq)),
            'cycle_freq': float(cycle_info.get('cycle_freq', 0.0)),
            'inst_freq': float(cycle_info.get('inst_freq', 0.0)),
            'live_swing_freq': float(cycle_info.get('live_swing_freq', live_freq)),
            'bends_per_min': float(cycle_info.get('bends_per_min', 0.0)),
            'swing_activity': float(cycle_info.get('swing_activity', 0.0)),
            'freq': float(avg_freq),
            'avg_freq': float(avg_freq),
            'average_frequency': float(avg_freq),
            'live_freq': float(live_freq),
            'realtime_freq': float(live_freq),
            'motion_type': str(cycle_info.get('motion_type', 'unknown')),
            'activity_score': float(cycle_info.get('activity_score', 0.0)),
            'is_ghost': bool(cycle_info.get('is_ghost', False)),
            'lost_frames': int(cycle_info.get('lost_frames', 0)),
            'count_mode': str(cycle_info.get('count_mode', 'swing')),
            'elapsed_seconds': float(cycle_info.get('elapsed_seconds', 0.0)),
            'unobserved_seconds': float(cycle_info.get('unobserved_seconds', 0.0)),
            'is_stale': bool(cycle_info.get('is_stale', False)),
            'avg_freq_observed': float(cycle_info.get('avg_freq_observed', 0.0)),
            'motion_activity': float(cycle_info.get('activity_score', 0.0)),
        })
    return summary




def write_swing_summary_csv(id_pose_summary, result_folder, stem):
    """把每只虫的摆动汇总写成 CSV（Excel 可直接打开），返回文件名。"""
    filename = f"{stem}_swing_summary.csv"
    path = ensure_unique_path(os.path.join(result_folder, filename))
    filename = os.path.basename(path)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', '有效观测秒数', '轨迹存续秒数', '未观测秒数',
                         '摆动次数(单方向)', '平均摆动频率(Hz)', '平均频率(仅观测,Hz)',
                         '实时摆动频率(Hz)', '完整周期数', '完整周期频率(Hz)', '摆动(次/分钟)',
                         '摆动活性(频率/5)', '运动类型', '运动活性(非静止占比)'])
        for item in id_pose_summary:
            writer.writerow([
                item.get('track_id', ''),
                item.get('observed_seconds', 0.0),
                item.get('elapsed_seconds', 0.0),
                item.get('unobserved_seconds', 0.0),
                item.get('swing_count', item.get('posture_count', 0)),
                item.get('swing_freq_hz', item.get('freq', 0.0)),
                item.get('avg_freq_observed', 0.0),
                item.get('live_swing_freq', item.get('live_freq', 0.0)),
                item.get('cycle_count', 0),
                item.get('cycle_freq', 0.0),
                item.get('bends_per_min', 0.0),
                item.get('swing_activity', 0.0),
                item.get('motion_type', 'unknown'),
                item.get('motion_activity', item.get('activity_score', 0.0)),
            ])
    return filename


def write_swing_events_csv(tracker, result_folder, stem):
    """导出每只虫的摆动事件明细（视频真实帧号 / 时间 / 类型 / 间隔）。

    用途：与视频逐帧核对摆动次数是否准确——次数偏多会看到过短的间隔，
    次数偏少会看到过长的间隔或整段没有事件。
    """
    filename = f"{stem}_swing_events.csv"
    path = ensure_unique_path(os.path.join(result_folder, filename))
    filename = os.path.basename(path)
    rows = []
    for track_id, hist in sorted(tracker.track_histories.items(), key=lambda x: int(x[0])):
        sc = hist.swing_counter
        fps = max(float(getattr(sc, 'fps', 30.0)), 1e-6)
        prev_t = None
        for idx, (frame_no, kind) in enumerate(getattr(sc, 'event_log', []), 1):
            t = float(frame_no) / fps
            gap = '' if prev_t is None else round(t - prev_t, 3)
            rows.append([int(track_id), idx, int(frame_no), round(t, 3),
                         'peak' if kind == 'peak' else 'valley', gap])
            prev_t = t
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', '事件序号', '视频帧号', '时间(秒)', '类型', '与上次间隔(秒)'])
        writer.writerows(rows)
    return filename


# ==================== 历史结果管理（磁盘清理） ====================
TASK_ID_RE = re.compile(r'(\d{8}_\d{6}_[0-9a-fA-F]{6})')


def _disk_task_summary():
    """扫描结果目录，按任务号聚合产物（文件数、大小、时间、对应上传视频）。"""
    result_dir = app.config['RESULT_FOLDER']
    sub_dir = app.config['SUB_VIDEO_FOLDER']
    upload_dir = app.config['UPLOAD_FOLDER']
    tasks = {}
    if os.path.isdir(result_dir):
        for name in os.listdir(result_dir):
            path = os.path.join(result_dir, name)
            if not os.path.isfile(path):
                continue
            m = TASK_ID_RE.search(name)
            if not m:
                continue
            tid = m.group(1)
            info = tasks.setdefault(tid, {'task_id': tid, 'files': 0, 'bytes': 0, 'mtime': 0.0, 'base': None})
            try:
                info['files'] += 1
                info['bytes'] += os.path.getsize(path)
                info['mtime'] = max(info['mtime'], os.path.getmtime(path))
            except OSError:
                pass
            if name.endswith('_result_%s.mp4' % tid):
                info['base'] = name.split('_result_')[0]
    if os.path.isdir(sub_dir):
        for d in os.listdir(sub_dir):
            dp = os.path.join(sub_dir, d)
            if d not in tasks or not os.path.isdir(dp):
                continue
            for root, _, files in os.walk(dp):
                for f in files:
                    try:
                        tasks[d]['files'] += 1
                        tasks[d]['bytes'] += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
    for tid, info in tasks.items():
        info['upload'] = None
        info['upload_bytes'] = 0
        if info.get('base'):
            for ext in ('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'):
                up = os.path.join(upload_dir, info['base'] + ext)
                if os.path.isfile(up):
                    info['upload'] = os.path.basename(up)
                    info['upload_bytes'] = os.path.getsize(up)
                    break
        info['mb'] = round(info['bytes'] / 1024 / 1024, 1)
        info['upload_mb'] = round(info['upload_bytes'] / 1024 / 1024, 1)
        info['time'] = time.strftime('%Y-%m-%d %H:%M', time.localtime(info['mtime'])) if info['mtime'] else ''
        info['running'] = any(t.get('status') == 'processing' and tid in str(t.get('video_path', ''))
                              for t in processing_tasks.values())
    return sorted(tasks.values(), key=lambda x: x['mtime'], reverse=True)


def _safe_remove(path, acc):
    """删除文件/目录并累计 (数量, 字节)。"""
    try:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    try:
                        acc[0] += 1
                        acc[1] += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            acc[0] += 1
            acc[1] += os.path.getsize(path)
            os.remove(path)
    except OSError:
        pass


def _delete_task_files(task_id, with_upload=False):
    """删除某任务的结果文件（可选连同上传原视频），返回 (删除数, 释放字节)。"""
    acc = [0, 0]
    result_dir = app.config['RESULT_FOLDER']
    upload_dir = app.config['UPLOAD_FOLDER']
    if os.path.isdir(result_dir):
        for name in os.listdir(result_dir):
            path = os.path.join(result_dir, name)
            m = TASK_ID_RE.search(name)
            if not m or m.group(1) != task_id or not os.path.isfile(path):
                continue
            if with_upload and name.endswith('_result_%s.mp4' % task_id):
                base = name.split('_result_')[0]
                for ext in ('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'):
                    up = os.path.join(upload_dir, base + ext)
                    if os.path.isfile(up):
                        _safe_remove(up, acc)
                        break
            _safe_remove(path, acc)
    sub = os.path.join(app.config['SUB_VIDEO_FOLDER'], task_id)
    if os.path.isdir(sub):
        _safe_remove(sub, acc)
    return acc[0], acc[1]


def _diag_fmt(value):
    """诊断 CSV 的数值格式化：缺失或无效值留空。"""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return ''
    return '' if fv != fv else round(fv, 3)


def _draw_diag_png(folder, stem, tid, frames, signals, events, fused, fps):
    """把某只虫的三路信号与各路/融合事件画成一张 PNG（cv2 绘制，无额外依赖）。"""
    import numpy as _np
    n = len(frames)
    if n < 2:
        return None
    W, H = 1400, 460
    pad_l, pad_t, pad_r, pad_b = 96, 58, 20, 30
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    canvas = _np.full((H, W, 3), 255, _np.uint8)
    span = max(1, frames[-1] - frames[0])

    def x_of(fr):
        return int(pad_l + (fr - frames[0]) / span * plot_w)

    for idx, item in enumerate((('bend', (60, 60, 220)), ('mid', (60, 165, 60)), ('curv', (210, 120, 30)))):
        name, color = item
        vals = _np.asarray(signals.get(name, []), dtype=float)
        if vals.size == 0:
            continue
        good = vals[_np.isfinite(vals)]
        if good.size == 0:
            continue
        vmin, vmax = float(good.min()), float(good.max())
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0
        base = pad_t + plot_h * (idx + 0.5) / 3.0
        amp = plot_h / 3.0 * 0.42
        prev = None
        for i, fr in enumerate(frames):
            v = vals[i] if i < vals.size else _np.nan
            if not _np.isfinite(v):
                prev = None
                continue
            y = int(base - (v - (vmin + vmax) / 2.0) / (vmax - vmin) * 2 * amp)
            x = x_of(fr)
            if prev is not None:
                cv2.line(canvas, prev, (x, y), color, 1, cv2.LINE_AA)
            prev = (x, y)
        for fr in events.get(name, ()):
            cv2.line(canvas, (x_of(fr), pad_t), (x_of(fr), pad_t + plot_h), color, 1)
        cv2.putText(canvas, '%s %.1f~%.1f' % (name, vmin, vmax), (6, int(base) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for fr in fused:
        cv2.circle(canvas, (x_of(fr), pad_t - 12), 4, (0, 0, 0), -1, cv2.LINE_AA)
    counts = {k: len(v) for k, v in events.items()}
    cv2.putText(canvas, 'ID %d   events per signal: %s   fused: %d' % (tid, counts, len(fused)),
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'black dots = fused events    frames %d..%d' % (frames[0], frames[-1]),
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), (225, 225, 225), 1)
    path = ensure_unique_path(os.path.join(folder, '%s_swing_diag_track%d.png' % (stem, tid)))
    cv2.imwrite(path, canvas)
    return os.path.basename(path)


def write_swing_diag(tracker, result_folder, stem):
    """导出摆动诊断：逐帧三路信号 + 各路/融合事件标记 CSV，以及每只虫的曲线图。"""
    rows, pngs = [], []
    for track_id, hist in sorted(tracker.track_histories.items(), key=lambda x: int(x[0])):
        frames = getattr(hist, 'diag_frames', None)
        if not frames or len(frames) < 20:
            continue
        signals = getattr(hist, 'diag_signals', {}) or {}
        sc = hist.swing_counter
        events = {'bend': set(), 'mid': set(), 'curv': set()}
        counters = getattr(sc, 'counters', None)
        if counters:
            for name, counter in counters.items():
                events[name] = {f for f, _ in counter.event_log}
        else:
            events['bend'] = {f for f, _ in getattr(sc, 'event_log', [])}
        fused = {f for f, _ in getattr(sc, 'fused_events', [])}
        if not fused:
            fused = {f for f, _ in getattr(sc, 'event_log', [])}
        fps = max(float(getattr(tracker, 'fps', 30.0)), 1e-6)
        for idx, fr in enumerate(frames):
            row = [int(track_id), int(fr), round(fr / fps, 3)]
            for key in ('bend', 'mid', 'curv'):
                seq = signals.get(key, [])
                row.append(_diag_fmt(seq[idx] if idx < len(seq) else None))
            for key in ('bend', 'mid', 'curv'):
                row.append(1 if fr in events[key] else 0)
            row.append(1 if fr in fused else 0)
            rows.append(row)
        png = _draw_diag_png(result_folder, stem, int(track_id), frames, signals, events, fused, fps)
        if png:
            pngs.append(png)
    if not rows:
        return None, pngs
    path = ensure_unique_path(os.path.join(result_folder, f"{stem}_swing_diag.csv"))
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'frame', 'time_sec', 'bend_deg', 'mid_percent', 'curv_deg',
                         'event_bend', 'event_mid', 'event_curv', 'event_fused'])
        writer.writerows(rows)
    return os.path.basename(path), pngs


def generate_task_id():
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def ensure_unique_path(path):
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f"{root}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def safe_rename(src, dst):
    unique_dst = ensure_unique_path(dst)
    os.rename(src, unique_dst)
    return unique_dst


def safe_upload_path(filename):
    return ensure_unique_path(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename)))


def create_compatible_video_writer(path, fps, frame_size):
    codecs = ['avc1', 'H264', 'X264', 'mp4v']
    last_error = None
    for codec in codecs:
        try:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, frame_size)
            if writer is not None and writer.isOpened():
                print(f"🎬 VideoWriter codec: {codec}")
                return writer
            if writer is not None:
                writer.release()
        except Exception as e:
            last_error = e
    raise RuntimeError(f"无法创建视频写入器: {last_error}")

# ==================== 辅助函数 ====================
def resolve_safe_path(base_dir, filename):
    """将传入文件名解析为指定目录下的安全路径，阻止路径穿透。"""
    base_path = Path(base_dir).resolve()
    candidate = (base_path / Path(str(filename))).resolve()
    base_str = str(base_path)
    candidate_str = str(candidate)
    if candidate_str != base_str and not candidate_str.startswith(base_str + os.sep):
        raise ValueError(f"Invalid path: {filename}")
    return candidate


def sanitize_upload_filename(original_filename):
    """
    生成安全的上传文件名（兼容中文/含特殊字符的文件名）。

    背景：werkzeug 的 secure_filename() 会丢弃非 ASCII 字符，例如
    “线虫实验视频.mp4” 会被清洗成 “mp4”（扩展名丢失），随后 allowed_file()
    会误判为不支持的文件类型导致上传被拒。这里先清洗主干名，再重新拼接
    原始扩展名；主干清洗为空时使用随机名兜底。
    """
    raw_name = str(original_filename or '').replace('\\', '/').rsplit('/', 1)[-1].strip()
    if not raw_name:
        return ''
    stem, ext = os.path.splitext(raw_name)
    ext = (ext or '').lower()
    allowed_ext = {'.' + e for e in app.config['ALLOWED_EXTENSIONS']}
    if ext not in allowed_ext:
        ext = ''
    safe_stem = secure_filename(stem) if stem else ''
    if not safe_stem:
        ascii_parts = re.findall(r'[A-Za-z0-9_-]+', stem)
        safe_stem = secure_filename('_'.join(ascii_parts)) if ascii_parts else ''
    if not safe_stem:
        safe_stem = f'upload_{uuid.uuid4().hex[:10]}'
    return f'{safe_stem}{ext}'


def validate_upload_file(file_storage):
    """校验上传文件并返回安全文件名。"""
    if file_storage is None:
        raise ValueError('No file provided')
    if not getattr(file_storage, 'filename', ''):
        raise ValueError('No selected file')

    filename = sanitize_upload_filename(file_storage.filename)
    if not filename:
        raise ValueError('Invalid file name')
    if not allowed_file(filename):
        raise ValueError('File type not allowed')

    content_length = getattr(file_storage, 'content_length', None)
    if content_length is not None and content_length > app.config['MAX_CONTENT_LENGTH']:
        raise ValueError('File is too large')

    return filename


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_image_file(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in {'jpg', 'jpeg', 'png', 'bmp'}

def process_video_task(task_id, video_path):
    """后台处理视频，通过 WebSocket 逐帧发送结果"""
    print(f"🚀 开始处理任务 {task_id}, 视频路径: {video_path}")
    update_task_state(task_id, status='processing', current_frame=0, total_frames=0, active_tracks=0)
    try:
        if is_image_file(video_path):
            frame = cv2.imread(video_path)
            if frame is None:
                update_task_state(task_id, status='failed', error_message='无法读取图片文件')
                socketio.emit('error', {'task_id': task_id, 'message': '无法读取图片文件'}, room=task_id)
                return

            height, width = frame.shape[:2]
            fps = 12.0
            total_frames = 36

            task_sub_dir = os.path.join(app.config['SUB_VIDEO_FOLDER'], task_id)
            os.makedirs(task_sub_dir, exist_ok=True)
            tracker = EnhancedWormTracker(fps=fps, frame_size=(width, height), sub_video_dir=task_sub_dir)

            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_result_{task_id}.mp4"
            output_path = ensure_unique_path(os.path.join(app.config['RESULT_FOLDER'], output_filename))
            output_filename = os.path.basename(output_path)
            out_writer = create_compatible_video_writer(output_path, fps, (width, height))

            log_entries = []
            for frame_idx in range(1, total_frames + 1):
                if not wait_or_cancel(task_id):
                    if out_writer is not None:
                        out_writer.release()
                    for track in tracker.tracks:
                        if track.sub_video_writer is not None:
                            track.sub_video_writer.release()
                    return

                # 使用活性过滤处理帧：实时预览走 debug（不画残留框），结果视频走 release（残留框画实线）
                frame_variants, track_data = process_frame_variants(
                    frame.copy(), model, device, tracker, frame_idx,
                    min_activity=TRACKING_CONFIG['activity_threshold']
                )
                preview_frame = frame_variants[PREVIEW_OVERLAY_MODE]
                result_frame = frame_variants[RESULT_OVERLAY_MODE]
                _, buffer = cv2.imencode('.jpg', preview_frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                frame_info = {
                    'frame': frame_idx,
                    'active_tracks': len(tracker.tracks),
                    'tracks': track_data
                }
                log_entries.append(frame_info)

                socketio.emit('frame', {
                    'task_id': task_id,
                    'frame': frame_idx,
                    'total_frames': total_frames,
                    'image': frame_b64,
                    'stats': frame_info
                }, room=task_id)

                update_task_state(
                    task_id,
                    status='processing',
                    current_frame=frame_idx,
                    total_frames=total_frames,
                    active_tracks=len(tracker.tracks),
                    latest_frame=frame_b64,
                    tracks=track_data,
                )

                out_writer.write(result_frame)

            out_writer.release()

            for track in tracker.tracks:
                if track.sub_video_writer is not None:
                    track.sub_video_writer.release()

            log_path = os.path.join(app.config['RESULT_FOLDER'], f"{os.path.splitext(output_filename)[0]}_log.json")
            with open(log_path, 'w') as f:
                json.dump(log_entries, f)

            for track_id, cycle_info in tracker.cycle_counts.items():
                cycles = cycle_info.get('cycles', 0)
                old_path = os.path.join(task_sub_dir, f"worm_{track_id}.mp4")
                if os.path.exists(old_path):
                    new_name = f"worm_{track_id}_cycles_{cycles}.mp4"
                    new_path = os.path.join(task_sub_dir, new_name)
                    safe_rename(old_path, new_path)

            zip_path = ensure_unique_path(os.path.join(app.config['RESULT_FOLDER'], f"{task_id}_sub_videos.zip"))
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for root, _, files in os.walk(task_sub_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        zf.write(full_path, arcname=file)

            id_pose_summary = build_id_pose_summary(tracker)
            summary_filename = f"{os.path.splitext(output_filename)[0]}_id_pose_summary.json"
            summary_path = ensure_unique_path(os.path.join(app.config['RESULT_FOLDER'], summary_filename))
            summary_filename = os.path.basename(summary_path)
            with open(summary_path, 'w') as f:
                json.dump(id_pose_summary, f)
                csv_filename = write_swing_summary_csv(id_pose_summary, app.config['RESULT_FOLDER'], os.path.splitext(output_filename)[0])
                events_filename = write_swing_events_csv(tracker, app.config['RESULT_FOLDER'], os.path.splitext(output_filename)[0])
                diag_filename = ''
                if SWING_DIAG:
                    diag_filename, _diag_pngs = write_swing_diag(tracker, app.config['RESULT_FOLDER'], os.path.splitext(output_filename)[0])

            socketio.emit('complete', {
                'task_id': task_id,
                'video_url': f"/media/{output_filename}",
                'log_url': f"/download/{os.path.basename(log_path)}",
                'sub_videos_url': f"/download/{os.path.basename(zip_path)}",
                'summary_url': f"/download/{summary_filename}",
                'swing_csv_url': f"/download/{csv_filename}",
                'swing_events_url': f"/download/{events_filename}",
                'swing_diag_url': f"/download/{diag_filename}" if diag_filename else '',
                'id_pose_summary': id_pose_summary,
            }, room=task_id)

            update_task_state(
                task_id,
                status='completed',
                current_frame=total_frames,
                total_frames=total_frames,
                video_url=f"/media/{output_filename}",
                log_url=f"/download/{os.path.basename(log_path)}",
                sub_videos_url=f"/download/{os.path.basename(zip_path)}",
                summary_url=f"/download/{summary_filename}",
                swing_csv_url=f"/download/{csv_filename}",
                swing_events_url=f"/download/{events_filename}",
                swing_diag_url=f"/download/{diag_filename}" if diag_filename else '',
                id_pose_summary=id_pose_summary,
            )
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 错误：无法打开视频文件 {video_path}")
            update_task_state(task_id, status='failed', error_message='无法打开视频文件')
            socketio.emit('error', {'task_id': task_id, 'message': '无法打开视频文件'}, room=task_id)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 视频信息: {width}x{height}, {fps:.2f} FPS, 总帧数: {total_frames}")
        update_task_state(task_id, total_frames=total_frames)

        # 为当前任务创建子视频目录（任务完成后会打包删除）
        task_sub_dir = os.path.join(app.config['SUB_VIDEO_FOLDER'], task_id)
        os.makedirs(task_sub_dir, exist_ok=True)

        # 初始化跟踪器（传入子视频目录）
        tracker = EnhancedWormTracker(fps=fps, frame_size=(width, height), sub_video_dir=task_sub_dir)

        # 输出完整视频
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_result_{task_id}.mp4"
        output_path = ensure_unique_path(os.path.join(app.config['RESULT_FOLDER'], output_filename))
        output_filename = os.path.basename(output_path)
        out_writer = create_compatible_video_writer(output_path, fps, (width, height))

        frame_idx = 0
        log_entries = []   # 存储每帧统计，用于最后保存 JSON

        while True:
            if not wait_or_cancel(task_id):
                cap.release()
                out_writer.release()
                for track in tracker.tracks:
                    if track.sub_video_writer is not None:
                        track.sub_video_writer.release()
                return

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # 调用整合模块处理单帧（使用活性过滤）：
            # 实时预览走 debug（不画残留框），结果视频走 release（残留框画实线）
            try:
                frame_variants, track_data = process_frame_variants(
                    frame, model, device, tracker, frame_idx,
                    min_activity=TRACKING_CONFIG['activity_threshold']
                )
                preview_frame = frame_variants[PREVIEW_OVERLAY_MODE]
                result_frame = frame_variants[RESULT_OVERLAY_MODE]
            except Exception as e:
                print(f"处理帧 {frame_idx} 时出错: {e}")
                import traceback
                traceback.print_exc()
                preview_frame = frame
                result_frame = frame
                track_data = []

            # 将标注帧编码为 base64 发送（推送的是实时预览帧，不含残留框）
            _, buffer = cv2.imencode('.jpg', preview_frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # 收集该帧统计信息
            frame_info = {
                'frame': frame_idx,
                'active_tracks': len(tracker.tracks),
                'tracks': track_data
            }
            log_entries.append(frame_info)

            # 通过 WebSocket 发送给前端
            socketio.emit('frame', {
                'task_id': task_id,
                'frame': frame_idx,
                'total_frames': total_frames,
                'image': frame_b64,
                'stats': frame_info
            }, room=task_id)

            update_task_state(
                task_id,
                status='processing',
                current_frame=frame_idx,
                total_frames=total_frames,
                active_tracks=len(tracker.tracks),
                latest_frame=frame_b64,
                tracks=track_data,
            )

            # 写入完整视频（结果视频：残留目标画实线框）
            out_writer.write(result_frame)

            # 每50帧打印一次进度
            if frame_idx % 50 == 0:
                print(f"任务 {task_id}: 已处理 {frame_idx}/{total_frames} 帧, 活跃目标数: {len(tracker.tracks)}")

        cap.release()
        out_writer.release()
        print(f"✅ 任务 {task_id} 视频读取完成，共 {frame_idx} 帧")

        # ========== 关键修复：关闭所有子视频写入器 ==========
        print("正在关闭子视频写入器...")
        for track in tracker.tracks:
            if track.sub_video_writer is not None:
                track.sub_video_writer.release()
                print(f"已关闭 track {track.track_id} 的子视频写入器")
        # ==================================================

        # 保存统计日志
        log_path = os.path.join(app.config['RESULT_FOLDER'], f"{os.path.splitext(output_filename)[0]}_log.json")
        with open(log_path, 'w') as f:
            json.dump(log_entries, f)

        # 重命名子视频文件，加入最终周期数
        for track_id, cycle_info in tracker.cycle_counts.items():
            cycles = cycle_info.get('cycles', 0)
            old_path = os.path.join(task_sub_dir, f"worm_{track_id}.mp4")
            if os.path.exists(old_path):
                new_name = f"worm_{track_id}_cycles_{cycles}.mp4"
                new_path = os.path.join(task_sub_dir, new_name)
                final_path = safe_rename(old_path, new_path)
                print(f"📦 子视频重命名: {os.path.basename(final_path)}")

        # 将子视频文件夹打包成 ZIP（供下载）
        zip_path = ensure_unique_path(os.path.join(app.config['RESULT_FOLDER'], f"{task_id}_sub_videos.zip"))
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for root, _, files in os.walk(task_sub_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    zf.write(full_path, arcname=file)
        print(f"📦 子视频已打包: {zip_path}")

        id_pose_summary = build_id_pose_summary(tracker)
        summary_filename = f"{os.path.splitext(output_filename)[0]}_id_pose_summary.json"
        summary_path = ensure_unique_path(os.path.join(app.config['RESULT_FOLDER'], summary_filename))
        summary_filename = os.path.basename(summary_path)
        with open(summary_path, 'w') as f:
            json.dump(id_pose_summary, f)
            csv_filename = write_swing_summary_csv(id_pose_summary, app.config['RESULT_FOLDER'], os.path.splitext(output_filename)[0])
            events_filename = write_swing_events_csv(tracker, app.config['RESULT_FOLDER'], os.path.splitext(output_filename)[0])
            diag_filename = ''
            if SWING_DIAG:
                diag_filename, _diag_pngs = write_swing_diag(tracker, app.config['RESULT_FOLDER'], os.path.splitext(output_filename)[0])

        # 可选：删除临时子视频目录（释放空间）
        # import shutil
        # shutil.rmtree(task_sub_dir)

        # 通知前端处理完成
        socketio.emit('complete', {
            'task_id': task_id,
            'video_url': f"/media/{output_filename}",
            'log_url': f"/download/{os.path.basename(log_path)}",
            'sub_videos_url': f"/download/{os.path.basename(zip_path)}",
            'summary_url': f"/download/{summary_filename}",
                'swing_csv_url': f"/download/{csv_filename}",
                'swing_events_url': f"/download/{events_filename}",
                'swing_diag_url': f"/download/{diag_filename}" if diag_filename else '',
            'id_pose_summary': id_pose_summary,
        }, room=task_id)
        update_task_state(
            task_id,
            status='completed',
            current_frame=frame_idx,
            total_frames=total_frames,
            video_url=f"/media/{output_filename}",
            log_url=f"/download/{os.path.basename(log_path)}",
            sub_videos_url=f"/download/{os.path.basename(zip_path)}",
            summary_url=f"/download/{summary_filename}",
                swing_csv_url=f"/download/{csv_filename}",
                swing_events_url=f"/download/{events_filename}",
                swing_diag_url=f"/download/{diag_filename}" if diag_filename else '',
            id_pose_summary=id_pose_summary,
        )
        print(f"🎉 任务 {task_id} 处理完成，已通知前端")

    except Exception as e:
        print(f"❌ 任务 {task_id} 发生未捕获异常: {e}")
        import traceback
        traceback.print_exc()
        update_task_state(task_id, status='failed', error_message=str(e))
        socketio.emit('error', {'task_id': task_id, 'message': str(e)}, room=task_id)

@app.after_request
def apply_cors_headers(response):
    """为 HTTP 接口补充 CORS 响应头，允许前端(如 localhost:3000)跨端口调用。"""
    origin = request.headers.get('Origin')
    if origin:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Vary'] = 'Origin'
    else:
        response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Content-Length, X-Requested-With, Authorization'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response


@app.errorhandler(413)
def handle_too_large(error):
    """上传超过 MAX_CONTENT_LENGTH 时返回 JSON 提示（默认是 HTML）。"""
    return jsonify({'error': '文件过大，单个视频不能超过 500MB'}), 413


# ==================== HTTP 路由 ====================
@app.route('/upload', methods=['POST'])
def upload_video():
    """接收上传的视频，启动处理任务"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    file = request.files['video']
    try:
        filename = validate_upload_file(file)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    try:
        video_path = safe_upload_path(filename)
        file.save(video_path)
    except OSError as exc:
        return jsonify({'error': f'Failed to save file: {exc}'}), 500

    print(f"📁 文件已保存: {video_path}")

    # 生成唯一任务 ID
    task_id = generate_task_id()
    with task_lock:
        processing_tasks[task_id] = {
            'status': 'processing',
            'video_path': video_path,
            'filename': filename,
            'current_frame': 0,
            'total_frames': 0,
            'active_tracks': 0,
            'latest_frame': '',
            'tracks': [],
            'video_url': '',
            'log_url': '',
            'sub_videos_url': '',
            'summary_url': '',
            'swing_csv_url': '',
            'swing_events_url': '',
            'swing_diag_url': '',
            'id_pose_summary': [],
            'error_message': '',
            'control': 'running',
        }
    print(f"🆔 生成任务ID: {task_id}")

    # 启动后台线程处理视频
    thread = threading.Thread(target=process_video_task, args=(task_id, video_path))
    thread.daemon = True
    thread.start()
    print(f"🚀 已启动后台线程处理任务 {task_id}")

    return jsonify({'task_id': task_id}), 202

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device) if device else 'uninitialized',
        'infer_imgsz': INFER_IMGSZ,
        'uptime_seconds': round(time.time() - APP_START_TIME, 2),
    })


@app.route('/tasks')
def list_disk_tasks():
    """列出磁盘上的历史任务产物，供前端管理与清理磁盘。"""
    tasks = _disk_task_summary()
    return jsonify({
        'tasks': tasks,
        'count': len(tasks),
        'total_mb': round(sum(t['bytes'] for t in tasks) / 1024 / 1024, 1),
        'upload_total_mb': round(sum(t['upload_bytes'] for t in tasks) / 1024 / 1024, 1),
    })


@app.route('/task/<task_id>', methods=['DELETE'])
def delete_disk_task(task_id):
    """删除某个任务的全部结果文件；with_upload=1 时连同上传的原视频一起删。"""
    if not TASK_ID_RE.fullmatch(task_id or ''):
        return jsonify({'error': '任务号格式不正确'}), 400
    task = next((t for t in _disk_task_summary() if t['task_id'] == task_id), None)
    if task is None:
        return jsonify({'error': '任务不存在'}), 404
    if task.get('running'):
        return jsonify({'error': '该任务正在处理，无法删除'}), 409
    with_upload = str(request.args.get('with_upload', '0')).lower() in ('1', 'true', 'yes')
    removed, freed = _delete_task_files(task_id, with_upload=with_upload)
    return jsonify({'task_id': task_id, 'removed': removed,
                    'freed_mb': round(freed / 1024 / 1024, 1), 'with_upload': with_upload})


@app.route('/tasks/clear', methods=['POST'])
def clear_disk_tasks():
    """批量清理：keep_latest 表示保留最近 N 个任务，其余删除。"""
    data = request.get_json(silent=True) or {}
    with_upload = bool(data.get('with_upload'))
    try:
        keep_latest = max(0, int(data.get('keep_latest', 0)))
    except (TypeError, ValueError):
        keep_latest = 0
    targets = [t for t in _disk_task_summary() if not t.get('running')][keep_latest:]
    removed, freed = 0, 0
    for t in targets:
        r, f = _delete_task_files(t['task_id'], with_upload=with_upload)
        removed += r
        freed += f
    return jsonify({'tasks_deleted': len(targets), 'removed': removed,
                    'freed_mb': round(freed / 1024 / 1024, 1)})


@app.route('/download/<filename>')
def download_file(filename):
    """提供结果文件下载（完整视频、日志、子视频 ZIP）"""
    try:
        file_path = resolve_safe_path(app.config['RESULT_FOLDER'], filename)
    except ValueError:
        return "Invalid file name", 400
    if not file_path.exists():
        return "File not found", 404
    return send_file(str(file_path), as_attachment=True)


@app.route('/media/<filename>')
def media_file(filename):
    """提供可在线播放媒体文件（主要用于前端 video 标签）。"""
    try:
        file_path = resolve_safe_path(app.config['RESULT_FOLDER'], filename)
    except ValueError:
        return "Invalid file name", 400
    if not file_path.exists():
        return "File not found", 404

    ext = os.path.splitext(filename)[1].lower()
    mimetype = 'application/octet-stream'
    if ext == '.mp4':
        mimetype = 'video/mp4'
    elif ext == '.json':
        mimetype = 'application/json'
    elif ext == '.zip':
        mimetype = 'application/zip'

    return send_file(str(file_path), mimetype=mimetype, as_attachment=False, conditional=True)

@app.route('/task/<task_id>/sub_videos')
def list_sub_videos(task_id):
    """列出任务的所有子视频"""
    safe_id = secure_filename(task_id)
    if not safe_id:
        return jsonify({'videos': []})
    try:
        task_sub_dir = resolve_safe_path(app.config['SUB_VIDEO_FOLDER'], safe_id)
    except ValueError:
        return jsonify({'videos': []})
    if not task_sub_dir.exists():
        return jsonify({'videos': []})
    videos = []
    for f in sorted(os.listdir(task_sub_dir)):
        if f.endswith('.mp4'):
            videos.append({
                'filename': f,
                'url': f'/sub_video/{safe_id}/{f}'
            })
    return jsonify({'videos': videos})


@app.route('/task/<task_id>/status')
def task_status(task_id):
    """返回任务状态，用于前端刷新后恢复处理进度"""
    state = get_task_state(str(task_id))
    if state is None:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(state)


@app.route('/task/<task_id>/pause', methods=['POST'])
def pause_task(task_id):
    task_id = str(task_id)
    state = get_task_state(task_id)
    if state is None:
        return jsonify({'error': 'Task not found'}), 404
    update_task_state(task_id, control='paused', status='paused')
    return jsonify({'ok': True, 'task_id': task_id, 'status': 'paused'})


@app.route('/task/<task_id>/resume', methods=['POST'])
def resume_task(task_id):
    task_id = str(task_id)
    state = get_task_state(task_id)
    if state is None:
        return jsonify({'error': 'Task not found'}), 404
    update_task_state(task_id, control='running', status='processing')
    return jsonify({'ok': True, 'task_id': task_id, 'status': 'processing'})


@app.route('/task/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    task_id = str(task_id)
    state = get_task_state(task_id)
    if state is None:
        return jsonify({'error': 'Task not found'}), 404
    update_task_state(task_id, control='cancelled', status='cancelled', error_message='任务已取消')
    return jsonify({'ok': True, 'task_id': task_id, 'status': 'cancelled'})

@app.route('/sub_video/<task_id>/<filename>')
def serve_sub_video(task_id, filename):
    """提供单个子视频文件"""
    safe_id = secure_filename(task_id)
    safe_name = secure_filename(filename)
    if not safe_id or not safe_name:
        return "Invalid file name", 400
    try:
        file_path = resolve_safe_path(os.path.join(app.config['SUB_VIDEO_FOLDER'], safe_id), safe_name)
    except ValueError:
        return "Invalid file name", 400
    if not file_path.exists():
        return "File not found", 404
    return send_file(str(file_path), mimetype='video/mp4', as_attachment=False, conditional=True)

# ==================== WebSocket 事件 ====================
@socketio.on('connect')
def handle_connect():
    print('✅ 客户端已连接')

@socketio.on('join')
def handle_join(data):
    """客户端加入任务房间，用于接收该任务的帧数据"""
    task_id = data.get('task_id')
    if task_id is not None:
        task_id = str(task_id)
        join_room(task_id)
        print(f'🏠 客户端加入房间: {task_id}')

# ==================== 启动服务 ====================
if __name__ == '__main__':
    # 加载模型（仅一次）
    model_path = str((BASE_DIR / 'models' / 'best.pt').resolve())
    if not os.path.exists(model_path):
        print(f"⚠️ 警告：模型文件 {model_path} 不存在，请先放置模型！")
    else:
        model, device = init_model(model_path)
        print(f"✅ 模型加载完成，设备: {device}")

    # 启动 Flask-SocketIO 服务器（可用环境变量 HOST / PORT 覆盖）
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '5000'))
    print(f"服务地址: http://127.0.0.1:{port}   健康检查: /health")
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)