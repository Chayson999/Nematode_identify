"""
整合线虫跟踪与姿态分析模块
- 使用增强跟踪器（卡尔曼滤波+外观特征+运动方向+匈牙利匹配）
- 实时提取中心线（11个关键点）并计算正弦周期数
- 为每个虫子生成子视频（裁剪区域，以ID和周期数命名）
- 提供 process_frame 函数供 WebSocket 逐帧调用
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import time
from scipy.optimize import linear_sum_assignment
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize

# ==================== 配置参数（可根据需要调整） ====================
CONFIDENCE_THRESHOLD = 0.2
IOU_THRESHOLD = 0.4
MAX_LOST_FRAMES = 15          # 兼容旧配置：丢失多少秒后回收（TRACK_PERSISTENCE_SEC 生效后仅作兜底）
MIN_CONFIDENCE = float(os.environ.get('MIN_CONFIDENCE', '0.3'))   # 低于该置信度的检测被丢弃（可用环境变量调整）
MIN_AREA_RATIO = 0.000015     # 最小面积比例（相对于图像面积）
KEYPOINT_UPDATE_INTERVAL = 2   # 每隔多少帧重新提取关键点（越小摆动采样越密，2 = 15Hz）
# 推理参数：线虫是小目标，640 下采样容易漏检。实测 300 帧：
#   640 -> 轨迹丢失 20 次；768 -> 13 次；960 -> 3 次；但速度约为 6.6 / 4 / 2.2 FPS。
# 想更快可用环境变量覆盖，例如启动前执行  set INFER_IMGSZ=640  （识别会更易丢帧）
INFER_IMGSZ = int(os.environ.get('INFER_IMGSZ', '960'))
INFER_CONF = float(os.environ.get('INFER_CONF', '0.25'))   # 送入跟踪器的低分候选阈值

# ==================== 摆动计数参数（可按需调整） ====================
SWING_COUNT_MODE = 'swing'     # swing = 头部每完成一次单方向摆动计 1 次（默认）；
                               # cycle = 头部完成一次左右往返（一个完整周期）计 1 次。
SWING_SIGNAL = 'bend'          # bend = 头部相对体轴弯曲角（抗整体转向漂移，默认）；
                               # absolute = 头部绝对方向角（旧行为）。
SWING_MIN_AMP_DEG = float(os.environ.get('SWING_MIN_AMP', '3.0'))   # 角度信号模式的幅度下限（度）
SWING_MIN_AMP_RATIO = float(os.environ.get('SWING_MIN_AMP_RATIO', '0.25'))  # 自适应下限：近期幅度的比例
SWING_MAX_AMP_DEG = 65.0       # 自适应幅度阈值上限（度），避免大幅度摆动被当成噪声
SWING_MIN_GAP_SEC = 0.10       # 两次摆动计数之间的最小时间间隔（秒）
SWING_BASELINE_SEC = 2.5       # 扣除整体转向漂移的滚动基线窗口（秒）
SWING_LIVE_WINDOW_SEC = 3.0    # 实时摆动频率统计窗口（秒）
SWING_LIVE_MIN_SEC = 1.0       # 实时频率至少累计多少秒数据才输出（避免开头数值离谱）
SWING_DIAG = str(os.environ.get('SWING_DIAG', '0')).strip().lower() not in ('0', 'false', 'no', '')
SWING_FUSION = str(os.environ.get('SWING_FUSION', '1')).strip().lower() not in ('0', 'false', 'no')
SWING_FUSE_WINDOW_SEC = float(os.environ.get('SWING_FUSE_WINDOW', '0.3'))   # 多信号融合的时间窗（秒）
SWING_REF_MAX_HZ = 5.0         # 归一化摆动活性分的参考最大摆动频率（Hz）

# ==================== 体态（C 形 / S 形）摆动计数参数 ====================
# 计数单位：一次"体态"计 1 次 —— 身体形成 C 形或 S 形且幅度足够时算一次。
# 判据全部用体长百分比（尺度无关），不再用"头部角度"（大小虫差异极大，实测幅度中位数 6%~85% 体长）。
SWING_POSTURE_MODE = str(os.environ.get('SWING_POSTURE_MODE', '1')).strip().lower() not in ('0', 'false', 'no')
SWING_POSTURE_MIN_AMP_PCT = float(os.environ.get('SWING_POSTURE_MIN_AMP_PCT', '10.0'))   # 有效弯曲：体长 10%
SWING_POSTURE_RELEASE_PCT = float(os.environ.get('SWING_POSTURE_RELEASE_PCT', '5.0'))    # 迟滞释放线：回落到该值以下才算姿态结束
SWING_POSTURE_HOLD_FRAMES = int(os.environ.get('SWING_POSTURE_HOLD_FRAMES', '2'))        # 姿态最短持续观测帧数
SWING_POSTURE_MIN_LOBE_PCT = float(os.environ.get('SWING_POSTURE_MIN_LOBE_PCT', '6.0'))  # 判 S 形时两个弯各自的幅度下限
SWING_POSTURE_UNIT = str(os.environ.get('SWING_POSTURE_UNIT', 'each')).strip().lower()   # each=每个 C/S 各计 1 次；pair=只有方向翻转才计

# ==================== 视觉残留（识别中断时保留序号）参数 ====================
TRACK_PERSISTENCE_SEC = 600.0   # 目标丢失后 ID 保留时长（秒），期间不会被回收，也不会被新目标抢占
GHOST_OVERLAY_SEC = float(os.environ.get('GHOST_OVERLAY_SEC', '1.5'))  # 丢失后仍绘制残留框 + ID 的时长（秒）
SUB_VIDEO_MIN_HITS = int(os.environ.get('SUB_VIDEO_MIN_HITS', '15'))   # 命中多少帧才为它建子视频（避免噪声 ID 变成"多出来的虫子"）
SUMMARY_MIN_HITS = int(os.environ.get('SUMMARY_MIN_HITS', '15'))       # 汇总只保留命中帧数达标的 ID
STALE_UNOBSERVED_SEC = float(os.environ.get('STALE_UNOBSERVED_SEC', '3.0'))   # 超过该秒数没有新观测即视为数据过期
# ==================== ROI 找回（针对停止运动后长期漏检的目标） ====================
ROI_RECOVER_ENABLED = str(os.environ.get('ROI_RECOVER', '1')).strip().lower() not in ('0', 'false', 'no', '')
ROI_RECOVER_CONF = float(os.environ.get('ROI_RECOVER_CONF', '0.15'))   # 找回用的低阈值
ROI_RECOVER_INTERVAL = int(os.environ.get('ROI_RECOVER_INTERVAL', '3'))  # 每 N 帧做一次找回，控制耗时
ROI_RECOVER_IMGSZ = int(os.environ.get('ROI_RECOVER_IMGSZ', '320'))
ROI_RECOVER_MIN_LOST = 2          # 丢失多少帧后开始找回
ROI_RECOVER_MAX_SEC = 120.0       # 超过该时长不再找回
ROI_RECOVER_PAD = 0.5             # ROI 相对目标外扩比例
ROI_RECOVER_MIN_SIZE = 160        # ROI 最小边长（像素）
ROI_RECOVER_MAX_ITEMS = 4         # 单次找回最多处理几条轨迹   # 超过该秒数没有新观测即视为数据过期
GHOST_REID_MAX_LOST = 90       # 丢失超过多少帧后，重新出现时放宽距离约束做找回匹配

# ==================== 画面叠加模式 ====================
# debug   : 实时预览（网页“实时处理区”逐帧推送的画面）——视觉残留（ghost）目标完全不上屏，
#           画面里只出现真实检测，避免虚线残留框干扰观察；
# release : 结果视频（处理完成后下载的主视频）——残留目标与真实检测一样画实线框，
#           并保留“残留 丢Nf”标签说明该框是预测保持的。
# 两种模式共用同一次推理与跟踪结果（见 process_frame_variants），不会重复推理。
OVERLAY_MODE_DEBUG = 'debug'
OVERLAY_MODE_RELEASE = 'release'
DEFAULT_OVERLAY_MODE = OVERLAY_MODE_RELEASE

# ==================== 重复识别抑制 / 贴合处理参数 ====================
# 去重：同一只虫被模型给出两个框/两块掩码时只保留置信度最高的那个。
# 关键：两条虫贴在一起时"框"也会高度重叠（实测框 IoU 0.2~0.7），
# 所以判据必须以「掩码重叠」为主，不能只看框 —— 旧实现只看框，实测会把贴合的另一条虫误删。
DEDUP_MASK_IOU = float(os.environ.get('DEDUP_MASK_IOU', '0.30'))            # 掩码 IoU 超过该值 = 同一只虫
DEDUP_BOX_STRONG_IOU = float(os.environ.get('DEDUP_BOX_STRONG_IOU', '0.75'))    # 框几乎重合…
DEDUP_BOX_STRONG_MASK_IOU = float(os.environ.get('DEDUP_BOX_STRONG_MASK_IOU', '0.15'))  # …且掩码有交集
# 出生门控：防止同一只虫被重复建 ID、防止瞬时误检占用 ID
BIRTH_BLOCK_MASK_IOU = float(os.environ.get('BIRTH_BLOCK_MASK_IOU', '0.30'))    # 与已匹配检测掩码重叠大 -> 已被解释
BIRTH_FRAGMENT_CONTAIN = float(os.environ.get('BIRTH_FRAGMENT_CONTAIN', '0.80'))  # 框被在跟踪目标的框包含比例
BIRTH_FRAGMENT_AREA = float(os.environ.get('BIRTH_FRAGMENT_AREA', '0.40'))      # 面积小于该比例 -> 判为片段
BIRTH_COOLDOWN_FRAMES = int(os.environ.get('BIRTH_COOLDOWN_FRAMES', '12'))      # 同一位置出生冷却帧数
BIRTH_COOLDOWN_OVERLAP = float(os.environ.get('BIRTH_COOLDOWN_OVERLAP', '0.5'))  # 冷却窗口内的掩码重叠判据
# 贴合共享：两条虫贴在一起、模型只给了一条检测时，让两条轨迹都活着（共享同一份检测）
SHARED_IOU = float(os.environ.get('SHARED_IOU', '0.25'))                        # 预测框与在跟踪框的重叠阈值
SHARED_MAX_LOST_FRAMES = int(os.environ.get('SHARED_MAX_LOST_FRAMES', '0'))     # 只对"上一帧还在"的轨迹生效
SHARED_PER_DET_MAX = int(os.environ.get('SHARED_PER_DET_MAX', '1'))             # 一条检测最多再共享给几条轨迹
# 残留框压在实时框上（同一只虫的重复 ID / 被挡住的虫）时不再画出来，避免看起来像重复识别
GHOST_SUPPRESS_IOU = float(os.environ.get('GHOST_SUPPRESS_IOU', '0.25'))


def create_compatible_video_writer(path, fps, frame_size):
    codecs = ['avc1', 'H264', 'X264', 'mp4v']
    last_error = None
    for codec in codecs:
        try:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, frame_size)
            if writer is not None and writer.isOpened():
                return writer
            if writer is not None:
                writer.release()
        except Exception as e:
            last_error = e
    raise RuntimeError(f"无法创建子视频写入器: {last_error}")

# ==================== 中心线提取与周期计数相关函数（从 yundong.py 移植） ====================

def keep_largest_component(mask_binary):
    """只保留最大的连通区域"""
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_binary
    largest = max(contours, key=cv2.contourArea)
    result = np.zeros_like(mask_binary)
    cv2.drawContours(result, [largest], -1, 1, thickness=-1)
    return result

def prune_skeleton(skel, min_branch_length=15):
    """骨架剪枝：去除短分支"""
    sk = skel.copy().astype(np.uint8)
    if np.sum(sk) == 0:
        return sk
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(min_branch_length):
        neighbor_count = cv2.filter2D(sk, -1, kernel)
        endpoints = (sk == 1) & (neighbor_count == 2)
        if not np.any(endpoints):
            break
        sk[endpoints] = 0
    return sk

def skeleton_from_mask(mask):
    """从二值掩码生成骨架"""
    bw = (mask > 0)
    skel = skeletonize(bw)
    return skel.astype(np.uint8)

def find_any_skeleton_pixel(skel):
    """找到骨架上的任意一个像素点"""
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return None
    return (ys[0], xs[0])

def bfs_on_skeleton(skel, start_rc):
    """BFS在骨架上找到最远点"""
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    prev = {}
    dist = {}
    q = deque()
    q.append(start_rc)
    visited[start_rc] = True
    dist[start_rc] = 0
    farthest = start_rc
    farthest_d = 0
    neigh = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        if d > farthest_d:
            farthest = (r, c)
            farthest_d = d
        for dr, dc in neigh:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and skel[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                prev[(nr, nc)] = (r, c)
                dist[(nr, nc)] = d + 1
                q.append((nr, nc))
    return farthest, farthest_d, prev, dist

def reconstruct_path(prev, start_rc, end_rc):
    """从prev字典重构路径，返回(x,y)格式的点列表"""
    path_rc = []
    cur = end_rc
    path_rc.append(cur)
    while cur != start_rc:
        cur = prev.get(cur)
        if cur is None:
            return None
        path_rc.append(cur)
    path_rc.reverse()
    path_xy = np.array([[c, r] for (r, c) in path_rc], dtype=float)
    return path_xy

def smooth_centerline(points, smoothness=3, num_points=200):
    """使用B样条平滑中心线"""
    if points is None or points.shape[0] == 0:
        return None
    if points.shape[0] < 3:
        xs = np.linspace(points[0, 0], points[-1, 0], num_points)
        ys = np.linspace(points[0, 1], points[-1, 1], num_points)
        return np.vstack([xs, ys]).T
    x = points[:, 0]
    y = points[:, 1]
    try:
        tck, u = interpolate.splprep([x, y], s=smoothness, k=3)
        u_fine = np.linspace(0, 1, num_points)
        x_f, y_f = interpolate.splev(u_fine, tck)
        return np.vstack([x_f, y_f]).T
    except Exception:
        t = np.linspace(0, 1, len(points))
        fx = interpolate.interp1d(t, x)
        fy = interpolate.interp1d(t, y)
        u = np.linspace(0, 1, num_points)
        return np.vstack([fx(u), fy(u)]).T

def sample_evenly_along_curve(points, n_samples):
    """沿曲线均匀采样n_samples个点"""
    if points is None or points.shape[0] == 0:
        return None
    diffs = np.diff(points, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = cumlen[-1]
    if total == 0:
        return np.tile(points[0], (n_samples, 1))
    target_lengths = np.linspace(0.0, total, n_samples)
    xs = points[:, 0]
    ys = points[:, 1]
    x_sampled = np.interp(target_lengths, cumlen, xs)
    y_sampled = np.interp(target_lengths, cumlen, ys)
    return np.vstack([x_sampled, y_sampled]).T

def extract_centerline_from_mask(mask, n_keypoints=11):
    """
    从分割掩码中提取中心线和关键点
    返回: (centerline_smooth, keypoints) 或 (None, None)
    """
    try:
        mask_binary = (mask > 0).astype(np.uint8)
        if np.sum(mask_binary) < 10:
            return None, None

        # 只保留最大连通区域
        mask_binary = keep_largest_component(mask_binary)

        # 形态学开运算
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

        mask_binary = keep_largest_component(mask_binary)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        if np.sum(mask_binary) < 10:
            return None, None

        skel = skeleton_from_mask(mask_binary)
        if np.sum(skel) < 3:
            return None, None

        skel = prune_skeleton(skel, min_branch_length=10)
        if np.sum(skel) < 3:
            return None, None

        s_rc = find_any_skeleton_pixel(skel)
        if s_rc is None:
            return None, None

        a_rc, a_d, prev_s, dist_s = bfs_on_skeleton(skel, s_rc)
        b_rc, b_d, prev_a, dist_a = bfs_on_skeleton(skel, a_rc)

        path_ab_xy = reconstruct_path(prev_a, a_rc, b_rc)
        if path_ab_xy is None or len(path_ab_xy) < 3:
            return None, None

        center_smooth = smooth_centerline(path_ab_xy, smoothness=2, num_points=200)
        if center_smooth is None:
            return None, None

        keypoints = sample_evenly_along_curve(center_smooth, n_keypoints)
        return center_smooth, keypoints
    except Exception:
        return None, None

def compute_head_angle(keypoints):
    """计算头部方向角度（度）"""
    if keypoints is None or len(keypoints) < 3:
        return 0.0
    head_pts = keypoints[:3]
    dx = head_pts[-1][0] - head_pts[0][0]
    dy = head_pts[-1][1] - head_pts[0][1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def compute_head_bend_angle(keypoints):
    """头部相对体轴的弯曲角（度）。

    用 kp[0]->kp[2]（头颈方向）与 kp[0]->kp[-1]（体轴方向）的夹角表示头部弯曲，
    整体平移/转向会被自然抵消，因此比绝对头部方向角更适合作为摆动信号。
    """
    if keypoints is None or len(keypoints) < 4:
        return None
    kp = np.asarray(keypoints, dtype=np.float64)
    head_vec = kp[2] - kp[0]
    axis_vec = kp[-1] - kp[0]
    if np.hypot(head_vec[0], head_vec[1]) < 1e-3 or np.hypot(axis_vec[0], axis_vec[1]) < 1e-3:
        return None
    cross = float(head_vec[0] * axis_vec[1] - head_vec[1] * axis_vec[0])
    dot = float(head_vec[0] * axis_vec[0] + head_vec[1] * axis_vec[1])
    return float(np.degrees(np.arctan2(cross, dot)))


def swing_signal_from_keypoints(keypoints, mode=SWING_SIGNAL):
    """按配置选择摆动信号：默认相对体轴头部弯曲角，可选绝对头部方向角。"""
    if keypoints is None or len(keypoints) < 3:
        return None
    if str(mode).lower().startswith('abs'):
        return float(compute_head_angle(keypoints))
    bend = compute_head_bend_angle(keypoints)
    if bend is None:
        return float(compute_head_angle(keypoints))
    return bend


def compute_mid_lateral_percent(keypoints, index=5):
    """中点相对头尾连线的横向偏移（体长百分比，带符号）。

    反映身体摆动时中段的左右位移，对整体正弦摆动比头颈夹角更敏感。
    """
    if keypoints is None or len(keypoints) < 6:
        return None
    kp = np.asarray(keypoints, dtype=np.float64)
    v = kp[-1] - kp[0]
    p = kp[min(index, len(kp) - 2)] - kp[0]
    length = float(np.hypot(v[0], v[1]))
    if length < 1e-3:
        return None
    cross = float(v[0] * p[1] - v[1] * p[0])
    return float(cross / (length * length) * 100.0)


def compute_body_curvature_deg(keypoints, i0=0, i1=5, i2=-1):
    """头-中-尾三点夹角（度，带符号），反映身体弯曲程度的变化。"""
    if keypoints is None or len(keypoints) < 6:
        return None
    kp = np.asarray(keypoints, dtype=np.float64)
    v1 = kp[i1] - kp[i0]
    v2 = kp[i2] - kp[i1]
    if np.hypot(v1[0], v1[1]) < 1e-3 or np.hypot(v2[0], v2[1]) < 1e-3:
        return None
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    dot = float(v1[0] * v2[0] + v1[1] * v2[1])
    return float(np.degrees(np.arctan2(cross, dot)))


def swing_signals_from_keypoints(keypoints):
    """一次算出三路摆动信号，供多信号融合计数使用。"""
    return {
        'bend': swing_signal_from_keypoints(keypoints),
        'mid': compute_mid_lateral_percent(keypoints),
        'curv': compute_body_curvature_deg(keypoints),
    }


def compute_posture_metrics(keypoints, min_lobe_pct=SWING_POSTURE_MIN_LOBE_PCT):
    """由中心线关键点算「体态」指标（尺度无关，单位：体长百分比）。

    取体长 25% / 50% / 75% 三点相对头尾连线的横向偏移（带符号）：
      s25/s50/s75 正值/负值代表弯向哪一侧
      a_pct       = max(|s25|,|s50|,|s75|)，即弯曲幅度（体长 %）
      shape       = 'C'（三点同号）/ 'S'（首尾异号，且两个弯都够大）/ 'mixed'
      direction   = 主弯曲方向 +1 / -1（S 形取头侧那一个弯的方向）

    返回 None 表示关键点不足或退化（无法判断姿态）。
    """
    if keypoints is None or len(keypoints) < 9:
        return None
    kp = np.asarray(keypoints, dtype=np.float64)
    chord = kp[-1] - kp[0]
    l2 = float(chord[0] * chord[0] + chord[1] * chord[1])
    if l2 < 1e-9:
        return None
    n = len(kp)

    def lateral(i):
        p = kp[i] - kp[0]
        return float(chord[0] * p[1] - chord[1] * p[0]) / l2 * 100.0

    i25 = int(round(0.25 * (n - 1)))
    i50 = int(round(0.50 * (n - 1)))
    i75 = int(round(0.75 * (n - 1)))
    s25, s50, s75 = lateral(i25), lateral(i50), lateral(i75)
    a_pct = max(abs(s25), abs(s50), abs(s75))

    if s25 * s75 < 0 and abs(s25) >= min_lobe_pct and abs(s75) >= min_lobe_pct \
            and abs(s50) < max(abs(s25), abs(s75)):
        shape = 'S'          # 两个相反的弯、中段接近直线 = 典型 S 形
    elif s25 * s50 >= 0 and s50 * s75 >= 0:
        shape = 'C'
    else:
        shape = 'mixed'
    dom = s50 if abs(s50) >= max(abs(s25), abs(s75)) else (s25 if abs(s25) >= abs(s75) else s75)
    return {
        's25': s25, 's50': s50, 's75': s75,
        'a_pct': a_pct, 'shape': shape,
        'direction': 1 if dom >= 0 else -1,
    }

def orient_pose_pair(centerline, keypoints, ref_head=None):
    """根据上一帧头部锚点把中心线/关键点整体定向，保证帧间“头尾”一致。

    骨架提取的起点是随机 BFS 像素，同一只线虫在不同帧可能整体反转；
    若尾部端点比头部端点更靠近上一帧头部，则整体翻转。头尾一致是
    摆动计数可靠性的前提。返回 (centerline, keypoints)。
    """
    if keypoints is None or len(keypoints) < 2:
        return centerline, keypoints
    kp = np.asarray(keypoints, dtype=np.float32)
    if ref_head is not None:
        head_d = float(np.hypot(kp[0, 0] - ref_head[0], kp[0, 1] - ref_head[1]))
        tail_d = float(np.hypot(kp[-1, 0] - ref_head[0], kp[-1, 1] - ref_head[1]))
        if tail_d < head_d:
            kp = kp[::-1].copy()
            if centerline is not None:
                centerline = np.asarray(centerline, dtype=np.float32)[::-1].copy()
    return centerline, kp

def curvature_of_centerline(centerline):
    """计算单帧中心线的 signed curvature 平均值"""
    if centerline is None or len(centerline) < 3:
        return 0.0
    pts = np.array(centerline)
    x = pts[:, 0]
    y = pts[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5 + 1e-9
    k = (dx * ddy - dy * ddx) / denom
    if len(k) > 4:
        return float(np.nanmean(k[2:-2]))
    return float(np.nanmean(k))

# ==================== 运动分类器 ====================

class MotionClassifier:
    """线虫运动分类器 - 识别具体运动类型"""
    
    MOTION_TYPES = {
        'static': 0,        # 静止
        'forward': 1,       # 前进
        'backward': 2,      # 后退
        'sinusoidal': 3,    # 正弦摆动
        'turning': 4,       # 转身
    }
    
    def __init__(self, speed_threshold=2.0, curvature_threshold=0.15):
        self.speed_threshold = speed_threshold
        self.curvature_threshold = curvature_threshold
    
    @staticmethod
    def classify_motion(keypoints, velocity, curvature):
        """
        分类线虫运动类型
        
        参数:
            keypoints: (N, 2) 关键点数组
            velocity: (2,) 速度向量
            curvature: float 曲率
        
        返回:
            (motion_type: str, confidence: float)
        """
        speed = np.linalg.norm(velocity)
        speed_threshold = 2.0
        curvature_threshold = 0.15
        
        # 判断静止
        if speed < speed_threshold:
            return 'static', 0.0
        
        # 判断高曲率 -> 正弦摆动
        curvature_magnitude = abs(curvature)
        if curvature_magnitude > curvature_threshold:
            return 'sinusoidal', curvature_magnitude
        
        # 根据头部方向和速度方向判断前进/后退/转身
        if keypoints is not None and len(keypoints) >= 3:
            head_dir = keypoints[-1] - keypoints[0]
            head_angle = np.arctan2(head_dir[1], head_dir[0])
            velocity_angle = np.arctan2(velocity[1], velocity[0])
            angle_diff = abs(head_angle - velocity_angle)
            
            # 归一化角度差到 [0, pi]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            angle_diff = abs(angle_diff)
            
            angle_threshold = np.pi / 6  # 30 度
            
            if angle_diff < angle_threshold:
                # 方向一致 -> 前进
                return 'forward', speed
            elif angle_diff > 5 * np.pi / 6:  # 150 度
                # 方向相反 -> 后退
                return 'backward', speed
            else:
                # 方向斜角 -> 转身
                return 'turning', angle_diff
        else:
            # 无关键点，按速度判断
            return 'forward', speed
    
    @staticmethod
    def compute_activity_score(motion_history, fps, window_sec=1.0):
        """
        计算目标活性分数 (0-1)
        基于最近 N 秒的运动统计
        
        参数:
            motion_history: list of (motion_type, confidence)
            fps: 帧率
            window_sec: 评估窗口（秒）
        
        返回:
            activity_score: float (0-1)
        """
        motion_history = list(motion_history)   # 兼容 deque 等不支持切片的容器
        interval = max(1, int(KEYPOINT_UPDATE_INTERVAL))
        window_samples = max(1, int(round(fps * window_sec / interval)))
        
        if len(motion_history) > window_samples:
            recent = motion_history[-window_samples:]
        else:
            recent = motion_history
        
        if not recent:
            return 0.0
        
        # 计算非静止的帧数
        active_frames = sum(1 for motion_type, _ in recent if motion_type != 'static')
        
        # 活性分数 = 活跃帧数 / 总帧数
        # 分母用窗口应有帧数：未被观测到的帧按不活跃计，漏检期间活性随时间下降
        activity = min(1.0, active_frames / float(window_samples))
        
        return activity

class SwingCounter:
    """逐帧增量式头部摆动计数器（默认：一次单方向摆动 = 1 次）。

    输入：每观测帧的头部摆动信号（默认头部相对体轴弯曲角）。
    处理：
      - 信号 unwrap 连续化，避免 ±180 度缠绕导致的假过零；
      - 滚动中位数基线扣除整体转向/游走漂移，得到残差 r(t)；
      - 自适应幅度阈值 A：A = clip(0.45 * 近期 |r| 的 90 分位, min_amp, max_amp)，
        既能抓住小幅摆动，又能抑制关键点抖动造成的重复计数；
      - 极值反转检测：残差从最近极值回撤超过 A 才确认一个极值（峰/谷），
        因此摆到一侧再摆回不会被重复计数。
    计数定义：
      - count_mode='swing'（默认）：每确认一个极值（峰或谷）= 头部单方向摆动 1 次，
        一次完整左右往返 = 2 次；
      - count_mode='cycle'：只在峰值事件计数，即一次完整左右往返 = 1 次。
    两种计数始终同时维护（swing_count / half_swing_count / cycle_count）。
    """

    def __init__(self, fps, min_amp_deg=SWING_MIN_AMP_DEG,
                 min_gap_sec=SWING_MIN_GAP_SEC, baseline_sec=SWING_BASELINE_SEC,
                 live_window_sec=SWING_LIVE_WINDOW_SEC, ref_max_hz=SWING_REF_MAX_HZ,
                 live_min_sec=SWING_LIVE_MIN_SEC, count_mode=SWING_COUNT_MODE,
                 max_amp_deg=SWING_MAX_AMP_DEG, smooth_alpha=0.45):
        self.fps = max(float(fps), 1e-6)
        self.min_amp = max(0.5, float(min_amp_deg))
        self.max_amp = max(self.min_amp, float(max_amp_deg))
        self.min_gap_frames = max(1, int(round(float(min_gap_sec) * self.fps)))
        self.baseline_len = max(3, int(round(float(baseline_sec) * self.fps)))
        self.live_len = max(3, int(round(float(live_window_sec) * self.fps)))
        self.live_min_frames = max(1, int(round(float(live_min_sec) * self.fps)))
        self.ref_max_hz = max(0.1, float(ref_max_hz))
        self.count_mode = 'cycle' if str(count_mode).lower().startswith('cyc') else 'swing'
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.05, 1.0))

        self._angles = deque(maxlen=self.baseline_len)   # unwrap 后信号（滚动基线用）
        self._amp_window = deque(maxlen=self.live_len)      # |残差|（自适应阈值用）
        self._residual_window = deque(maxlen=self.live_len)  # 残差（噪声估计用）
        self.event_log = []                                  # 全部摆动事件 (frame_no, kind)，供导出核对
        self._video_frame = None                             # 最近一次喂入的视频帧号（实时窗口按视频时间推进）
        self._observed_stamps = deque()                      # 实时窗口内实际观测到的视频帧号
        self._events = deque()                           # (frame_no, kind) kind: peak/valley
        self._last_unwrapped = None
        self._smoothed = None
        self._peak_val = -1e9
        self._peak_frame = 0
        self._valley_val = 1e9
        self._valley_frame = 0
        self._direction = 1            # +1 上升段 / -1 下降段
        self._last_event_frame = -(10 ** 9)

        self.swing_count = 0           # 主计数（按 count_mode）
        self.half_swing_count = 0      # 单方向摆动次数（峰 + 谷）
        self.cycle_count = 0           # 完整往返周期数（峰）
        self.observed_frames = 0
        self._frame_no = 0
        self.threshold = self.min_amp
        self.last_residual = 0.0

    def _unwrap_next(self, raw_deg):
        if self._last_unwrapped is None:
            self._last_unwrapped = float(raw_deg)
        else:
            delta = float(raw_deg) - self._last_unwrapped
            delta = (delta + 180.0) % 360.0 - 180.0
            self._last_unwrapped += delta
        return self._last_unwrapped

    def _noise_level(self):
        """估计信号的高频噪声（度）：先去趋势，再取绝对偏差的稳健中位数。

        注意不能直接用一阶差分：周期摆动本身每帧变化就很大，会把信号当成噪声，
        导致阈值被抬到几乎无法计数。
        """
        if len(self._residual_window) < 12:
            return 0.0
        v = np.fromiter(self._residual_window, dtype=float)
        trend = np.convolve(v, np.ones(3) / 3.0, mode='valid')   # 3 点滑动平均
        high_freq = v[1:-1] - trend                              # 高频成分
        return float(np.median(np.abs(high_freq)) * 1.4826)

    def _adaptive_threshold(self):
        """自适应幅度阈值（度）。

        下限不再固定为 8 度：改为 max(绝对下限, 近期幅度的比例)，
        这样小幅度摆动也能被计入；上限仍受噪声的 2.5 倍约束，避免把抖动当摆动。
        """
        if len(self._amp_window) < 5:
            return self.min_amp
        amp = float(np.percentile(np.fromiter(self._amp_window, dtype=float), 90))
        floor = max(self.min_amp, SWING_MIN_AMP_RATIO * amp)
        thr = max(0.45 * amp, 3.5 * self._noise_level())
        return float(np.clip(thr, floor, self.max_amp))

    def _register_event(self, kind, frame_no, amp):
        """确认一次极值事件；返回本次新增计数（0 或 1）。"""
        if frame_no - self._last_event_frame < self.min_gap_frames:
            return 0
        extreme = self._peak_val if kind == 'peak' else self._valley_val
        if abs(extreme) < amp:
            return 0            # 极值幅度不足，视为漂移/噪声，不计一次摆动
        self._last_event_frame = frame_no
        self._events.append((frame_no, kind))
        self.event_log.append((frame_no, kind))
        self.half_swing_count += 1
        if kind == 'peak':
            self.cycle_count += 1
        if self.count_mode == 'swing':
            self.swing_count = self.half_swing_count
        else:
            self.swing_count = self.cycle_count
        return 1

    def feed(self, signal_deg, frame_no=None):
        """喂入一帧信号，返回本次新累计的摆动数（0 或 1）。

        frame_no 为视频真实帧号（可选）；传入后事件日志可与视频逐帧对齐。
        """
        self._frame_no += 1
        self._video_frame = self._frame_no if frame_no is None else int(frame_no)
        self.observed_frames += 1
        if signal_deg is None or not np.isfinite(signal_deg):
            return 0
        unwrapped = self._unwrap_next(signal_deg)
        self._angles.append(unwrapped)
        if len(self._angles) < 2:
            return 0

        vals = list(self._angles)
        baseline = float(np.median(vals[:-1]))    # 只用历史样本估计基线，避免自影响
        residual = unwrapped - baseline
        self.last_residual = residual
        self._amp_window.append(abs(residual))
        self._residual_window.append(residual)

        if self._smoothed is None:
            self._smoothed = residual
        else:
            a = self.smooth_alpha
            self._smoothed = (1.0 - a) * self._smoothed + a * residual
        r = self._smoothed

        amp = self._adaptive_threshold()
        self.threshold = amp

        new_swings = 0
        if self._direction > 0:
            if r > self._peak_val:
                self._peak_val = r
                self._peak_frame = self._video_frame
            if self._peak_val - r >= amp:
                new_swings += self._register_event('peak', self._peak_frame, amp)
                self._direction = -1
                self._valley_val = r
                self._valley_frame = self._frame_no
        else:
            if r < self._valley_val:
                self._valley_val = r
                self._valley_frame = self._video_frame
            if r - self._valley_val >= amp:
                new_swings += self._register_event('valley', self._valley_frame, amp)
                self._direction = 1
                self._peak_val = r
                self._peak_frame = self._frame_no

        self._observed_stamps.append(self._video_frame)
        while self._events and self._video_frame - self._events[0][0] > self.live_len:
            self._events.popleft()
        while self._observed_stamps and self._video_frame - self._observed_stamps[0] > self.live_len:
            self._observed_stamps.popleft()
        return new_swings

    # ---------- 统计 ----------
    def observed_seconds(self):
        return self.observed_frames / self.fps

    def _mode_event_frames(self):
        return [f for f, kind in self._events if self.count_mode == 'swing' or kind == 'peak']

    def avg_freq(self):
        secs = self.observed_seconds()
        return self.swing_count / secs if secs > 0 else 0.0

    def avg_cycle_freq(self):
        secs = self.observed_seconds()
        return self.cycle_count / secs if secs > 0 else 0.0

    def live_freq(self):
        """实时摆动频率：按**视频时间**窗口统计。

        事件会随视频时间退出窗口；若目标停止运动并被漏检，窗口内不再有新事件，
        频率会自然衰减到 0，而不会冻结在旧的高值上。
        """
        if self._video_frame is None or len(self._observed_stamps) < self.live_min_frames:
            return 0.0
        events = [f for f in self._mode_event_frames() if self._video_frame - f <= self.live_len]
        observed_sec = len(self._observed_stamps) / self.fps
        if observed_sec <= 0:
            return 0.0
        return len(events) / observed_sec

    def inst_freq(self, max_events=5):
        """用最近若干次事件间隔的中位数估计瞬时频率（Hz）。"""
        frames = self._mode_event_frames()
        if len(frames) < 2:
            return 0.0
        frames = frames[-(max_events + 1):]
        deltas = np.diff(np.asarray(frames, dtype=float)) / self.fps
        deltas = deltas[deltas > 0]
        if len(deltas) == 0:
            return 0.0
        return float(1.0 / np.median(deltas))

    def activity(self):
        return min(1.0, self.avg_freq() / self.ref_max_hz)


class PostureSwingCounter(SwingCounter):
    """按「体态」计数的摆动计数器：一次 C 形或 S 形计 1 次（默认口径）。

    与 SwingCounter 对外接口一致（统计层可直接替换），但计数单位是"一次姿态"，
    而不是头部弯曲角信号的极值 —— 后者会在身体保持同一个 C 形时被头颈抖动反复触发
    （实测相邻两次计数方向相同的比例最高到 7/7）。

    判据（全部尺度无关，单位：体长 %）：
      - 幅度门  ：a_pct >= min_amp_pct（默认体长 10%）才算一次有效弯曲 → 微小摆动不计；
      - 持续门  ：同一体态需连续保持 hold_frames 个观测帧 → 排除单帧抖动；
      - 迟滞门  ：幅度回落到 release_pct 以下才算姿态结束；结束之后再出现同一体态也算新的一次；
      - 单位    ：unit='each'（默认）每个不同的体态（形状或方向变化）各计 1 次；
                  unit='pair' 只有弯曲方向翻转才计 1 次（同一方向的 C/S 视为同一次）；
      - 间隔门  ：沿用 SWING_MIN_GAP_SEC。

    计数维护：half_swing_count / swing_count = 体态次数；
    cycle_count = 完整往返次数 = 相邻已计数体态的方向翻转折算 (flips + 1) // 2。
    """

    def __init__(self, fps, min_amp_pct=SWING_POSTURE_MIN_AMP_PCT,
                 release_pct=SWING_POSTURE_RELEASE_PCT, hold_frames=SWING_POSTURE_HOLD_FRAMES,
                 unit=SWING_POSTURE_UNIT, min_lobe_pct=SWING_POSTURE_MIN_LOBE_PCT,
                 count_mode=SWING_COUNT_MODE):
        super().__init__(fps=fps, count_mode=count_mode)
        self.min_amp_pct = float(min_amp_pct)
        self.release_pct = min(float(release_pct), self.min_amp_pct)
        self.hold_frames = max(1, int(hold_frames))
        self.unit = 'pair' if str(unit).lower().startswith('pair') else 'each'
        self.min_lobe_pct = float(min_lobe_pct)
        self.last_shape = ''
        self.last_amp_pct = 0.0
        self._episode_label = None      # 正在持续的体态 (shape, direction)
        self._episode_len = 0
        self._counted_label = None      # 上一次已计数的体态
        self._counted_direction = 0
        self._flips = 0
        self._flip_frames = deque()     # 方向翻转发生的视频帧（count_mode='cycle' 的频率用）
        self._released = True           # 幅度已回落到释放线以下（允许同一体态再计一次）
        self._last_kp = None            # 上一次喂入的关键点（识别重复观测）
        self.threshold = self.min_amp_pct

    def feed_keypoints(self, keypoints, frame_no=None):
        """喂入一帧关键点，返回本次新增的体态计数（0 或 1）。"""
        self._frame_no += 1
        self._video_frame = int(frame_no) if frame_no is not None else self._frame_no

        self.observed_frames += 1
        self._observed_stamps.append(self._video_frame)
        while self._observed_stamps and self._video_frame - self._observed_stamps[0] > self.live_len:
            self._observed_stamps.popleft()
        while self._events and self._video_frame - self._events[0][0] > self.live_len:
            self._events.popleft()
        while self._flip_frames and self._video_frame - self._flip_frames[0] > self.live_len:
            self._flip_frames.popleft()
        self.threshold = self.min_amp_pct

        metrics = compute_posture_metrics(keypoints, min_lobe_pct=self.min_lobe_pct)
        if metrics is None:
            return 0
        self.last_shape = metrics['shape']
        self.last_amp_pct = metrics['a_pct']

        # 迟滞：幅度回落到释放线以下 -> 本姿态结束（之后同样的姿态再出现算新的一次）
        if metrics['a_pct'] < self.release_pct:
            self._episode_label = None
            self._episode_len = 0
            self._released = True
            self._last_kp = None
            return 0

        # 姿态不明确（既不是清楚的 C 也不是清楚的 S）：不计数，也不释放
        if metrics['shape'] not in ('C', 'S'):
            self._episode_label = None
            self._episode_len = 0
            return 0

        label = (metrics['shape'], metrics['direction'])
        # 关键点每 KEYPOINT_UPDATE_INTERVAL 帧才重新提取一次，期间喂进来的是同一份观测：
        # 重复观测只维持姿态、不累计持续时间，否则单次观测就能满足 hold 条件。
        kp_arr = np.asarray(keypoints, dtype=np.float64)
        fresh = self._last_kp is None or not np.array_equal(self._last_kp, kp_arr)
        self._last_kp = kp_arr.copy()
        if fresh:
            if label == self._episode_label:
                self._episode_len += 1
            else:
                self._episode_label, self._episode_len = label, 1
        elif label == self._episode_label:
            return 0
        else:
            self._episode_label, self._episode_len = label, 1

        if metrics['a_pct'] < self.min_amp_pct:
            return 0                        # 微小摆动：幅度不够
        if self._episode_len < self.hold_frames:
            return 0                        # 持续时间不够（单帧抖动）
        if self.unit == 'pair':
            is_new = metrics['direction'] != self._counted_direction
        else:
            is_new = self._released or label != self._counted_label
        if not is_new:
            return 0                        # 同一个体态还在持续，不重复计数
        if self._video_frame - self._last_event_frame < self.min_gap_frames:
            return 0
        return self._register_posture(label)

    def _register_posture(self, label):
        direction = label[1]
        if self._counted_label is not None and direction != self._counted_direction:
            self._flips += 1
            self._flip_frames.append(self._video_frame)
        self._counted_label = label
        self._counted_direction = direction
        self._released = False
        self._last_event_frame = self._video_frame
        kind = 'peak' if direction > 0 else 'valley'
        self._events.append((self._video_frame, kind))
        self.event_log.append((self._video_frame, kind))
        self.half_swing_count += 1
        self.cycle_count = (self._flips + 1) // 2
        self.swing_count = self.half_swing_count if self.count_mode == 'swing' else self.cycle_count
        return 1

    def _mode_event_frames(self):
        """实时/瞬时频率使用的事件帧：swing 模式=全部体态；cycle 模式=方向翻转。"""
        if self.count_mode == 'cycle':
            return list(self._flip_frames)
        return [f for f, _kind in self._events]


class SwingFusion:
    """多信号融合摆动计数器（A2）。

    三路信号：
      bend = 头部弯曲角（相对体轴）
      mid  = 中点相对头尾连线的横向偏移（体长百分比）
      curv = 头-中-尾三点夹角（身体弯曲程度）
    融合规则：同一时间窗（SWING_FUSE_WINDOW_SEC，默认 0.3 秒）内至少 2 路信号各自
    检测到极值，才计为一次摆动。单路的抖动/伪影会被另外两路否决；某一路因幅度小
    没触发时，另外两路仍可确认，从而减少漏计。

    对外接口与 SwingCounter 一致，便于统计层直接替换。
    """

    def __init__(self, fps, count_mode=SWING_COUNT_MODE, fuse_window_sec=SWING_FUSE_WINDOW_SEC):
        self.fps = max(float(fps), 1e-6)
        self.count_mode = 'cycle' if str(count_mode).lower().startswith('cyc') else 'swing'
        self.window_frames = max(1, int(round(float(fuse_window_sec) * self.fps)))
        self.counters = {name: SwingCounter(fps=fps) for name in ('bend', 'mid', 'curv')}
        for sc in self.counters.values():
            sc.count_mode = 'swing'        # 各路统一按极值记录，输出口径由融合层决定
            sc.swing_count = sc.half_swing_count
        self.ref_max_hz = self.counters['bend'].ref_max_hz
        self._pending = []                 # [(frame_no, kind, source)]
        self.fused_events = []             # 融合后的 (frame_no, kind)
        self.half_swing_count = 0
        self.cycle_count = 0
        self.swing_count = 0
        self._last_fused_frame = -(10 ** 9)

    # ---------- 属性代理 ----------
    @property
    def event_log(self):
        return self.fused_events

    @property
    def observed_frames(self):
        return self.counters['bend'].observed_frames

    @property
    def threshold(self):
        return self.counters['bend'].threshold

    def observed_seconds(self):
        return self.counters['bend'].observed_seconds()

    # ---------- 主流程 ----------
    def feed(self, signals, frame_no=None):
        """signals 为 {'bend': v, 'mid': v, 'curv': v}；缺路按 None 处理（该路本帧不产生事件）。"""
        if not isinstance(signals, dict):
            signals = {'bend': signals}
        for name, sc in self.counters.items():
            before = len(sc.event_log)
            sc.feed(signals.get(name), frame_no=frame_no)
            for ev in sc.event_log[before:]:
                self._pending.append((ev[0], ev[1], name))
        if not self._pending:
            return 0
        self._pending.sort(key=lambda x: x[0])
        latest = self._pending[-1][0]
        stable = [e for e in self._pending if e[0] <= latest - self.window_frames]
        if not stable:
            return 0
        self._pending = [e for e in self._pending if e[0] > latest - self.window_frames]
        new_count = 0
        i = 0
        while i < len(stable):
            t0 = stable[i][0]
            j = i + 1
            while j < len(stable) and stable[j][0] - t0 <= self.window_frames:
                j += 1
            cluster = stable[i:j]
            sources = {e[2] for e in cluster}
            if len(sources) >= 2:
                kinds = [e[1] for e in cluster]
                kind = 'peak' if kinds.count('peak') >= kinds.count('valley') else 'valley'
                frame = int(np.median([e[0] for e in cluster]))
                new_count += self._register_fused(frame, kind)
            i = j
        return new_count

    def _register_fused(self, frame_no, kind):
        min_gap = max(1, self.counters['bend'].min_gap_frames)
        if frame_no - self._last_fused_frame < min_gap:
            return 0
        self._last_fused_frame = frame_no
        self.fused_events.append((frame_no, kind))
        self.half_swing_count += 1
        if kind == 'peak':
            self.cycle_count += 1
        self.swing_count = self.half_swing_count if self.count_mode == 'swing' else self.cycle_count
        return 1

    # ---------- 统计 ----------
    def _mode_event_frames(self):
        return [f for f, kind in self.fused_events if self.count_mode == 'swing' or kind == 'peak']

    def avg_freq(self):
        secs = self.observed_seconds()
        return self.swing_count / secs if secs > 0 else 0.0

    def avg_cycle_freq(self):
        secs = self.observed_seconds()
        return self.cycle_count / secs if secs > 0 else 0.0

    def live_freq(self):
        base = self.counters['bend']
        if base._video_frame is None or len(base._observed_stamps) < base.live_min_frames:
            return 0.0
        events = [f for f in self._mode_event_frames() if base._video_frame - f <= base.live_len]
        observed_sec = len(base._observed_stamps) / self.fps
        if observed_sec <= 0:
            return 0.0
        return len(events) / observed_sec

    def inst_freq(self, max_events=5):
        frames = self._mode_event_frames()
        if len(frames) < 2:
            return 0.0
        frames = frames[-(max_events + 1):]
        deltas = np.diff(np.asarray(frames, dtype=float)) / self.fps
        deltas = deltas[deltas > 0]
        if len(deltas) == 0:
            return 0.0
        return float(1.0 / np.median(deltas))


class TrackHistory:
    """存储每只线虫的历史数据（中心线、头角度、运动信息）用于正弦运动计数与活性评估"""
    def __init__(self, max_len_frames, fps=30):
        self.centerlines = deque(maxlen=max_len_frames)
        self.head_angles = deque(maxlen=max_len_frames)
        self.swing_signals = deque(maxlen=max_len_frames)   # 摆动计数所用信号（弯曲角/绝对角）
        self.velocities = deque(maxlen=max_len_frames)
        self.curvatures = deque(maxlen=max_len_frames)
        self.motion_types = deque(maxlen=max_len_frames)  # list of (motion_type, confidence)
        self.keypoints_list = deque(maxlen=max_len_frames)
        
        self.fps = fps
        self.max_len_frames = max_len_frames
        self.total_cycles = 0
        self.first_frame = None      # 首次被观测到的视频帧号（用于计算存续时间）
        self.last_peak_frame = -1
        self.frame_counter = 0
        self.current_window_cycles = 0
        self.forward_streak = 0
        self.turning_streak = 0
        
        # 运动分类器实例
        self.motion_classifier = MotionClassifier()

        # 诊断模式：保存逐帧三路信号（SWING_DIAG=1 时开启，默认关闭以省内存）
        if SWING_DIAG:
            self.diag_frames = []
            self.diag_signals = {'bend': [], 'mid': [], 'curv': []}

        # 摆动计数器：
        #   默认按体态计数（一次 C 形或 S 形 = 1 次，幅度用体长百分比，微小摆动不计）；
        #   SWING_POSTURE_MODE=0 退回旧口径（SWING_FUSION=1 三路信号融合 / 0 单信号）。
        if SWING_POSTURE_MODE:
            self.swing_counter = PostureSwingCounter(fps=fps)
        elif SWING_FUSION:
            self.swing_counter = SwingFusion(fps=fps)
        else:
            self.swing_counter = SwingCounter(fps=fps)

    def append(self, centerline_11pts, head_angle_deg, velocity=None, curvature=None, keypoints=None):
        """
        添加一帧的数据
        
        参数:
            centerline_11pts: (11, 2) 中心线关键点
            head_angle_deg: float 头部角度（度）
            velocity: (2,) 速度向量，默认 None
            curvature: float 曲率，默认 None
            keypoints: (N, 2) 关键点，默认 None
        """
        self.centerlines.append(centerline_11pts)
        self.head_angles.append(head_angle_deg)
        
        if velocity is None:
            velocity = np.zeros(2, dtype=np.float32)
        if curvature is None:
            curvature = 0.0
        
        self.velocities.append(velocity)
        self.curvatures.append(curvature)
        self.keypoints_list.append(keypoints)
        
        # 分类运动类型
        motion_type, confidence = self.motion_classifier.classify_motion(
            keypoints, velocity, curvature
        )
        self.motion_types.append((motion_type, confidence))
        self.frame_counter += 1

    def diag_append(self, frame_no, signals):
        """诊断模式：记录该帧的三路信号值（缺失记为 NaN）。"""
        if not SWING_DIAG or not hasattr(self, 'diag_frames'):
            return
        self.diag_frames.append(int(frame_no) if frame_no is not None else self.frame_counter)
        values = signals if isinstance(signals, dict) else {'bend': signals}
        for key in ('bend', 'mid', 'curv'):
            value = values.get(key)
            self.diag_signals[key].append(float(value) if value is not None else float('nan'))

    def get_centerlines(self):
        return list(self.centerlines)

    def get_head_angles(self):
        return list(self.head_angles)
    
    def get_velocities(self):
        return list(self.velocities)
    
    def get_curvatures(self):
        return list(self.curvatures)
    
    def get_motion_types(self):
        """获取运动类型列表"""
        return list(self.motion_types)

    def get_activity_score(self, window_sec=1.0):
        """
        获取当前活性分数 (0-1)
        
        参数:
            window_sec: 评估窗口（秒）
        
        返回:
            float (0-1)
        """
        return self.motion_classifier.compute_activity_score(
            list(self.motion_types), self.fps, window_sec
        )
    
    def is_active(self, min_activity=0.3, window_sec=1.0):
        """
        判断目标是否足够活跃

        除运动分类活性外，只要目标确实累计到头部摆动（原地摆动、缓慢爬行时
        整体平移速度很小，会被分类为 static），也判定为活跃目标，避免真实
        线虫因为平移速度低而被活性过滤掉。

        参数:
            min_activity: 最小活性阈值 (0-1)
            window_sec: 评估窗口（秒）

        返回:
            bool
        """
        if self.get_activity_score(window_sec) >= min_activity:
            return True
        if self.swing_counter.swing_count >= 2:
            return True
        return self.swing_counter.live_freq() > 0.0

    def get_dominant_motion_type(self, window_frames=30):
        """
        获取最近 N 帧内主导的运动类型
        
        参数:
            window_frames: 窗口大小
        
        返回:
            str 运动类型
        """
        # 注意：motion_types 是 deque，不支持切片，必须先转 list
        items = list(self.motion_types)
        recent = items if len(items) < window_frames else items[-window_frames:]
        
        if not recent:
            return 'static'
        
        # 统计各运动类型
        type_counts = {}
        for motion_type, _ in recent:
            type_counts[motion_type] = type_counts.get(motion_type, 0) + 1
        
        # 返回频次最高的类型
        return max(type_counts.items(), key=lambda x: x[1])[0]

# ==================== 增强跟踪器（基于 genzong102.py） ====================

class KalmanBoxTracker:
    """卡尔曼滤波器，状态: [cx, cy, w, h, vx, vy, vw, vh]"""
    def __init__(self, bbox, feature=None, dt=1.0, track_id=None, sub_video_writer=None):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], np.float32)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 100.0

        self.history = deque(maxlen=30)
        self.history.append((cx, cy, w, h))
        self.keypoints_history = deque(maxlen=10)
        self.centerline_history = deque(maxlen=10)
        self.motion_vector = np.zeros(2, dtype=np.float32)
        self.last_keypoints = None
        self.last_observed_keypoints = None   # 最近一次真实提取的关键点（摆动计数用）
        self.last_centerline = None
        self.last_keypoint_update_frame = -1
        self.last_predicted_bbox = np.array(bbox, dtype=np.float32)
        self.last_observed_bbox = np.array(bbox, dtype=np.float32)   # 最后一次真实检测框
        self.crop_center = np.array([cx, cy], dtype=np.float32)
        self.crop_size = float(max(w, h) * 1.8)
        self.time_since_update = 0
        self.hits = 1
        self.lost_frames = 0
        self.confirmed = False
        self.track_id = track_id

        self.feature = feature if feature is not None else np.zeros(11)
        self.feature_history = deque(maxlen=20)
        self.feature_history.append(self.feature)

        # 子视频写入器
        self.sub_video_writer = sub_video_writer

        # 头部方向一致性锚点（摆动计数帧间定向用）
        self.orient_head = None

    def predict(self):
        pred = self.kf.predict()
        cx = pred[0, 0].item()
        cy = pred[1, 0].item()
        w = pred[2, 0].item()
        h = pred[3, 0].item()
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        self.last_predicted_bbox = bbox
        self.crop_center = 0.9 * self.crop_center + 0.1 * np.array([cx, cy], dtype=np.float32)
        return bbox

    def update(self, bbox, feature=None):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        # 【修复】测量向量形状 (4,1)，与 Kalman 滤波器输入一致
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(measurement)
        self.history.append((cx, cy, w, h))
        target_center = np.array([cx, cy], dtype=np.float32)
        target_size = float(max(w, h) * 1.8)
        self.crop_center = 0.8 * self.crop_center + 0.2 * target_center
        self.crop_size = 0.85 * self.crop_size + 0.15 * target_size
        self.time_since_update = 0
        self.lost_frames = 0
        self.last_observed_bbox = np.asarray(bbox, dtype=np.float32).copy()
        self.hits += 1
        if feature is not None:
            self.feature = feature.copy()
            self.feature_history.append(feature)

    def get_stable_crop_window(self, frame_w, frame_h, scale=1.35, min_size=72):
        size = int(max(min_size, round(self.crop_size * scale)))
        size = min(size, frame_w, frame_h)
        half = size // 2

        cx = int(round(float(self.crop_center[0])))
        cy = int(round(float(self.crop_center[1])))

        x1 = cx - half
        y1 = cy - half
        x2 = x1 + size
        y2 = y1 + size

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > frame_w:
            shift = x2 - frame_w
            x1 = max(0, x1 - shift)
            x2 = frame_w
        if y2 > frame_h:
            shift = y2 - frame_h
            y1 = max(0, y1 - shift)
            y2 = frame_h

        return int(x1), int(y1), int(x2), int(y2)

    def update_keypoints(self, keypoints, centerline=None):
        if keypoints is None or len(keypoints) == 0:
            return
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if self.last_keypoints is not None and len(self.last_keypoints) == len(keypoints):
            delta = keypoints - self.last_keypoints
            self.motion_vector = 0.7 * self.motion_vector + 0.3 * np.mean(delta, axis=0)
        elif len(self.history) >= 2:
            p1 = np.array(self.history[-2][:2], dtype=np.float32)
            p2 = np.array(self.history[-1][:2], dtype=np.float32)
            self.motion_vector = 0.7 * self.motion_vector + 0.3 * (p2 - p1)
        self.last_keypoints = keypoints
        self.orient_head = keypoints[0].copy()
        if centerline is not None:
            self.last_centerline = np.asarray(centerline, dtype=np.float32)
            self.centerline_history.append(self.last_centerline)
        self.keypoints_history.append(keypoints)

    def freeze(self):
        """进入残留状态：清零卡尔曼速度项与运动向量，框与裁剪窗停在最后位置。

        识别中断常常是线虫停在原地或短暂被遮挡；若继续按速度外推，
        残留框会一路飘走，反而更难与目标对上。
        """
        self.kf.statePost[4:8, 0] = 0.0   # statePost 形状为 (8, 1)
        self.motion_vector[:] = 0.0

    def get_motion_vector(self):
        if np.linalg.norm(self.motion_vector) > 1e-6:
            return self.motion_vector
        if len(self.history) >= 2:
            p1 = np.array(self.history[-2][:2], dtype=np.float32)
            p2 = np.array(self.history[-1][:2], dtype=np.float32)
            return p2 - p1
        velocity = np.asarray(self.get_velocity(), dtype=np.float32)
        if velocity.shape[0] >= 2:
            return velocity[:2]
        return np.zeros(2, dtype=np.float32)

    def predict_keypoints(self):
        if self.last_keypoints is None:
            return None
        return self.last_keypoints + self.get_motion_vector()

    def get_state(self):
        state = self.kf.statePost
        cx = state[0, 0].item()
        cy = state[1, 0].item()
        w = state[2, 0].item()
        h = state[3, 0].item()
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return np.array([x1, y1, x2, y2])

    def get_velocity(self):
        vx = self.kf.statePost[4, 0].item()
        vy = self.kf.statePost[5, 0].item()
        return np.array([vx, vy])

    def get_predicted_center(self):
        state = self.kf.statePost
        return state[0, 0].item(), state[1, 0].item()

    def get_feature(self):
        if len(self.feature_history) > 0:
            return np.mean(self.feature_history, axis=0)
        return self.feature


def _normalize_bbox(bbox):
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.size < 4:
        return np.zeros(4, dtype=np.float32)
    return bbox[:4]


def _mask_to_binary(mask):
    """把 supervision/YOLO 的掩码转成 2D 布尔图（兼容 (H,W)、(1,H,W) 与浮点）。"""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=0)
    return arr > 0


def _mask_iou_in_boxes(mask1, mask2, bbox1, bbox2, area1=None, area2=None):
    """两块掩码的 IoU：只在两框交集内求交，并集按各自总面积算，避免整图运算。

    返回 (iou, intersection_area)；任一掩码为空/形状不一致时返回 (0.0, 0)。
    """
    if mask1 is None or mask2 is None:
        return 0.0, 0
    a = _mask_to_binary(mask1)
    b = _mask_to_binary(mask2)
    if a.shape != b.shape:
        return 0.0, 0
    x1 = max(int(bbox1[0]), int(bbox2[0]))
    y1 = max(int(bbox1[1]), int(bbox2[1]))
    x2 = min(int(bbox1[2]), int(bbox2[2]))
    y2 = min(int(bbox1[3]), int(bbox2[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0
    inter = int(np.count_nonzero(a[y1:y2, x1:x2] & b[y1:y2, x1:x2]))
    if inter == 0:
        return 0.0, 0
    if area1 is None:
        area1 = int(np.count_nonzero(a))
    if area2 is None:
        area2 = int(np.count_nonzero(b))
    union = area1 + area2 - inter
    return (inter / union if union > 0 else 0.0), inter


def _contain_ratio(inner, outer):
    """inner 框被 outer 框覆盖的面积比例（0~1）。"""
    x1 = max(float(inner[0]), float(outer[0]))
    y1 = max(float(inner[1]), float(outer[1]))
    x2 = min(float(inner[2]), float(outer[2]))
    y2 = min(float(inner[3]), float(outer[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = (float(inner[2]) - float(inner[0])) * (float(inner[3]) - float(inner[1]))
    if area <= 0:
        return 0.0
    return float((x2 - x1) * (y2 - y1) / area)


class EnhancedWormTracker:
    """
    增强线虫跟踪器：集成卡尔曼滤波、外观特征、运动方向、匈牙利匹配，
    并整合中心线提取与周期计数，支持子视频生成
    """
    def __init__(self, fps=25, frame_size=(640, 480), sub_video_dir=None):
        self.fps = fps
        self.width, self.height = frame_size
        self.sub_video_dir = sub_video_dir   # 子视频保存目录

        # 跟踪参数
        # 视觉残留：目标丢失后 ID 保留时长（期间不会被回收，也不会被新目标抢占），
        # 以及仍绘制残留框 + 序号的时长；超过保留时长才真正回收轨迹。
        self.persistence_frames = int(round(TRACK_PERSISTENCE_SEC * fps))
        self.ghost_overlay_frames = int(round(GHOST_OVERLAY_SEC * fps))
        self.max_lost_frames = max(int(fps * MAX_LOST_FRAMES), self.persistence_frames)
        self.confirm_hits = 5
        self.sub_video_min_hits = max(int(self.confirm_hits), int(SUB_VIDEO_MIN_HITS))
        # 本帧"真正在跟踪"的目标数（已确认且本帧有观测/共享），用于界面"活跃目标"显示。
        # 注意不能用 len(self.tracks)：残留在后台保留 600 秒，会让数量看起来远超实际虫子数。
        self.live_track_count = 0
        self.iou_match_thresh = 0.5
        self.iou_secondary = 0.3
        self.appearance_weight = 0.3
        self.direction_weight = 0.2
        self.max_speed = 50
        self.cost_threshold = 2.5
        self.keypoint_weight = 0.9
        self.prediction_confidence = 0.18

        # 过滤参数
        self.MIN_CONFIDENCE = MIN_CONFIDENCE
        self.MIN_AREA = (self.width * self.height) * MIN_AREA_RATIO

        # 轨迹管理
        self.tracks = []          # 存储 KalmanBoxTracker 对象
        self.next_id = 1          # 只增不减：被回收的 ID 也不会被复用
        self.current_frame_idx = 0   # 当前处理到的视频帧号（统计存续时间用）
        self.ghost_ids = set()    # 当前处于视觉残留（丢失但保留 ID）状态的 track_id
        self.confirmed_ids = set()  # 已确认（连续命中）的 track_id，未确认的瞬时误检不显示
        self._empty_mask = None   # 复用的空掩码（残留目标用），避免每帧重复分配整幅数组
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.id_last_seen = {}

        # 重叠处理参数
        self.overlap_threshold = 0.3   # 没有掩码时才用的框 NMS 阈值（有掩码时以掩码重叠为主）

        # 重复识别抑制 / 贴合处理参数
        self.dedup_mask_iou = DEDUP_MASK_IOU
        self.dedup_box_strong_iou = DEDUP_BOX_STRONG_IOU
        self.dedup_box_strong_mask_iou = DEDUP_BOX_STRONG_MASK_IOU
        self.birth_block_mask_iou = BIRTH_BLOCK_MASK_IOU
        self.birth_fragment_contain = BIRTH_FRAGMENT_CONTAIN
        self.birth_fragment_area = BIRTH_FRAGMENT_AREA
        self.birth_cooldown_frames = BIRTH_COOLDOWN_FRAMES
        self.birth_cooldown_overlap = BIRTH_COOLDOWN_OVERLAP
        self.shared_iou = SHARED_IOU
        self.shared_max_lost_frames = SHARED_MAX_LOST_FRAMES
        self.shared_per_det_max = SHARED_PER_DET_MAX
        self.ghost_suppress_iou = GHOST_SUPPRESS_IOU
        self.shared_ids = set()       # 本帧"贴合共享"的 track_id（保持存活，不进入残留）
        self.shared_live_ids = set()  # 本帧把检测共享出去、需要暂停摆动计数的 track_id
        self.shadowed_ids = set()     # 本帧被实时框压住的残留 ID：数据保留但不绘制第二个框
        self._recent_births = deque(maxlen=64)   # 最近新建轨迹 (frame_idx, bbox)，用于出生冷却

        # 姿态分析相关
        self.track_histories = {}      # track_id -> TrackHistory
        self.cycle_counts = {}         # track_id -> 周期信息

    def extract_stable_roi(self, frame, track, output_size=(200, 200)):
        if frame is None:
            return None
        x1, y1, x2, y2 = track.get_stable_crop_window(self.width, self.height)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return cv2.resize(roi, output_size)

    def _extract_centerline_in_window(self, mask, x1, y1, x2, y2):
        """在指定窗口内提取中心线并返回整帧坐标（补回窗口偏移）。

        相比在全图 mask 上做 skeletonize/BFS，仅在小窗口内计算，
        可显著降低逐帧中心线提取的开销。
        """
        if mask is None or x2 <= x1 or y2 <= y1:
            return None, None
        if mask.ndim >= 3:
            roi = np.any(mask[y1:y2, x1:x2] > 0, axis=0)
        else:
            roi = mask[y1:y2, x1:x2]
        if roi.size == 0 or int(np.sum(roi > 0)) < 10:
            return None, None
        centerline, keypoints = extract_centerline_from_mask(roi, n_keypoints=11)
        if keypoints is None:
            return None, None
        offset = np.array([x1, y1], dtype=np.float32)
        if centerline is not None:
            centerline = np.asarray(centerline, dtype=np.float32) + offset
        keypoints = np.asarray(keypoints, dtype=np.float32) + offset
        return centerline, keypoints

    def extract_features(self, mask, bbox):
        """提取11维外观特征（仅在 bbox 邻域内找轮廓，避免全图计算）"""
        if mask is None or mask.size == 0:
            return np.zeros(11)
        if mask.ndim >= 3:
            mask = np.any(mask > 0, axis=0)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        aspect_ratio = w / (h + 1e-6)
        roi_x1 = max(0, int(np.floor(x1)) - 2)
        roi_y1 = max(0, int(np.floor(y1)) - 2)
        roi_x2 = min(mask.shape[1], int(np.ceil(x2)) + 2)
        roi_y2 = min(mask.shape[0], int(np.ceil(y2)) + 2)
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return np.zeros(11)
        mask = mask[roi_y1:roi_y2, roi_x1:roi_x2]
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        area = np.sum(mask > 0)
        if area < 50:
            return np.zeros(11)
        log_area = np.log1p(area)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            log_perimeter = 0
            circularity = 0
            hu = np.zeros(7)
        else:
            cnt = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(cnt, True)
            log_perimeter = np.log1p(perimeter)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            moments = cv2.moments(cnt)
            hu = cv2.HuMoments(moments).flatten()
            hu = np.sign(hu) * np.log1p(np.abs(hu))
        return np.concatenate(([log_area, log_perimeter, circularity, aspect_ratio], hu))

    def compute_appearance_distance(self, feat1, feat2):
        return np.linalg.norm(feat1 - feat2)

    def compute_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def compute_center_distance(self, bbox1, bbox2):
        cx1 = (bbox1[0] + bbox1[2]) / 2.0
        cy1 = (bbox1[1] + bbox1[3]) / 2.0
        cx2 = (bbox2[0] + bbox2[2]) / 2.0
        cy2 = (bbox2[1] + bbox2[3]) / 2.0
        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def compute_direction_cost(self, track, detection_center):
        v = track.get_motion_vector()
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return 0.0
        v_dir = v / norm
        pred_center = np.array(track.get_predicted_center())
        delta = np.array(detection_center) - pred_center
        norm_delta = np.linalg.norm(delta)
        if norm_delta < 1e-6:
            return 0.0
        delta_dir = delta / norm_delta
        cos_sim = np.dot(v_dir, delta_dir)
        cost = (1 - cos_sim) / 2.0
        return cost

    def compute_keypoint_cost(self, track, detection_keypoints):
        if detection_keypoints is None or len(detection_keypoints) == 0:
            return 0.0
        predicted_keypoints = track.predict_keypoints()
        if predicted_keypoints is None or len(predicted_keypoints) == 0:
            if track.last_keypoints is None or len(track.last_keypoints) == 0:
                return 0.0
            predicted_keypoints = track.last_keypoints

        predicted_keypoints = np.asarray(predicted_keypoints, dtype=np.float32)
        detection_keypoints = np.asarray(detection_keypoints, dtype=np.float32)
        if predicted_keypoints.shape != detection_keypoints.shape:
            return 0.0

        scale = np.linalg.norm(predicted_keypoints[-1] - predicted_keypoints[0])
        if scale < 1e-6:
            scale = np.linalg.norm(track.get_velocity()) + 1.0

        direct_cost = np.mean(np.linalg.norm(predicted_keypoints - detection_keypoints, axis=1)) / scale
        reversed_cost = np.mean(np.linalg.norm(predicted_keypoints - detection_keypoints[::-1], axis=1)) / scale
        return min(direct_cost, reversed_cost)

    def build_prediction_detections(self, predicted_tracks, detections):
        if len(predicted_tracks) == 0:
            return detections

        pred_boxes = []
        pred_confidences = []
        pred_tracker_ids = []
        pred_class_ids = []
        pred_masks = []

        actual_mask = getattr(detections, 'mask', None)
        mask_shape = None
        if actual_mask is not None and isinstance(actual_mask, np.ndarray) and actual_mask.ndim >= 3:
            mask_shape = actual_mask.shape[1:]

        for track in predicted_tracks:
            bbox = track.last_predicted_bbox if track.last_predicted_bbox is not None else track.get_state()
            pred_boxes.append(_normalize_bbox(bbox))
            pred_confidences.append(self.prediction_confidence)
            pred_tracker_ids.append(int(track.track_id))
            pred_class_ids.append(0)
            if mask_shape is not None:
                if self._empty_mask is None or self._empty_mask.shape != mask_shape:
                    self._empty_mask = np.zeros(mask_shape, dtype=bool)
                pred_masks.append(self._empty_mask)   # 复用同一份，避免每帧新建大数组

        pred_boxes = np.vstack(pred_boxes).astype(np.float32) if len(pred_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        pred_confidences = np.asarray(pred_confidences, dtype=np.float32)
        pred_tracker_ids = np.asarray(pred_tracker_ids, dtype=int)
        pred_class_ids = np.asarray(pred_class_ids, dtype=int)

        xyxy = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4) if len(detections) > 0 else np.zeros((0, 4), dtype=np.float32)
        confidence = detections.confidence if detections.confidence is not None else np.ones(len(detections), dtype=np.float32)
        tracker_id = detections.tracker_id if hasattr(detections, 'tracker_id') and detections.tracker_id is not None else np.full(len(detections), -1, dtype=int)
        class_id = detections.class_id if hasattr(detections, 'class_id') and detections.class_id is not None else np.zeros(len(detections), dtype=int)

        merged_xyxy = np.concatenate([xyxy, pred_boxes], axis=0) if len(xyxy) > 0 else pred_boxes
        merged_confidence = np.concatenate([confidence, pred_confidences], axis=0) if len(confidence) > 0 else pred_confidences
        merged_tracker_id = np.concatenate([tracker_id, pred_tracker_ids], axis=0) if len(tracker_id) > 0 else pred_tracker_ids
        merged_class_id = np.concatenate([class_id, pred_class_ids], axis=0) if len(class_id) > 0 else pred_class_ids

        merged_mask = None
        if actual_mask is not None and isinstance(actual_mask, np.ndarray) and actual_mask.ndim >= 3 and len(pred_masks) > 0:
            merged_mask = np.concatenate([actual_mask, np.asarray(pred_masks)], axis=0)
        elif actual_mask is not None:
            merged_mask = actual_mask
        elif len(pred_masks) > 0:
            merged_mask = np.asarray(pred_masks)

        return sv.Detections(
            xyxy=merged_xyxy,
            mask=merged_mask,
            confidence=merged_confidence,
            class_id=merged_class_id,
            tracker_id=merged_tracker_id,
        )

    def apply_nms(self, detections, masks=None):
        """去重：同一只虫的重复检测只保留置信度最高的那个。

        有掩码时的判据：
          1) 掩码 IoU > `dedup_mask_iou` —— 同一只虫（整只 + 局部片段、两套重复预测）；
          2) 框 IoU > `dedup_box_strong_iou` 且掩码有交集 —— 同一只虫；
          3) 一个框几乎完全包住另一个框、且掩码严重重叠 —— 同一只虫的片段。
        **不再仅凭框 IoU 抑制**：两条虫贴在一起时框会高度重叠（实测 0.2~0.7），
        只看框会把贴合的另一条虫直接删掉。没有掩码时才退回原来的"只比框"行为。
        """
        n = len(detections)
        if n <= 1:
            return detections
        boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
        scores = detections.confidence if detections.confidence is not None else np.ones(n)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        mask_areas = None
        if masks is not None and len(masks) == n:
            mask_areas = [int(np.count_nonzero(_mask_to_binary(masks[i]))) for i in range(n)]

        order = list(np.argsort(-scores))
        keep = []
        while order:
            i = int(order[0])
            keep.append(i)
            if len(order) == 1:
                break
            rest = [int(r) for r in order[1:]]
            rest_arr = np.asarray(rest, dtype=int)
            xx1 = np.maximum(boxes[i, 0], boxes[rest_arr, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest_arr, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest_arr, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest_arr, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[i] + areas[rest_arr] - inter
            box_ious = np.divide(inter, union, out=np.zeros_like(union), where=union > 0)

            survivors = []
            for k, r in enumerate(rest):
                if mask_areas is None:
                    suppressed = bool(box_ious[k] > self.overlap_threshold)
                else:
                    miou, _inter = _mask_iou_in_boxes(masks[i], masks[r], boxes[i], boxes[r],
                                                      mask_areas[i], mask_areas[r])
                    suppressed = miou > self.dedup_mask_iou
                    if not suppressed and box_ious[k] > self.dedup_box_strong_iou \
                            and miou > self.dedup_box_strong_mask_iou:
                        suppressed = True
                    if not suppressed:
                        contain = max(_contain_ratio(boxes[r], boxes[i]),
                                      _contain_ratio(boxes[i], boxes[r]))
                        if contain >= 0.95 and miou > 0.5:
                            suppressed = True
                if not suppressed:
                    survivors.append(r)
            order = survivors
        return detections[keep]

    def compute_motion_consistency_cost(self, track, detection_bbox, detection_size=None):
        """
        计算运动一致性成本
        包括：速度一致性、尺寸稳定性
        
        参数:
            track: KalmanBoxTracker
            detection_bbox: (x1, y1, x2, y2)
            detection_size: float 检测目标大小，可选
        
        返回:
            float 成本值 (0-1)
        """
        if len(track.history) < 2:
            return 0.0
        
        # 历史位置
        cx_prev, cy_prev, w_prev, h_prev = track.history[-1]
        
        # 检测位置
        cx_det = (detection_bbox[0] + detection_bbox[2]) / 2.0
        cy_det = (detection_bbox[1] + detection_bbox[3]) / 2.0
        w_det = detection_bbox[2] - detection_bbox[0]
        h_det = detection_bbox[3] - detection_bbox[1]
        
        if detection_size is None:
            detection_size = w_det + h_det
        
        # 预测位置
        cx_pred, cy_pred = track.get_predicted_center()
        
        # 1. 速度一致性：检测位置应接近卡尔曼预测位置
        velocity_diff = np.sqrt((cx_det - cx_pred)**2 + (cy_det - cy_pred)**2)
        # 正规化到 [0, 1]
        velocity_cost = min(velocity_diff / (self.max_speed + 1e-6), 1.0)
        
        # 2. 尺寸稳定性：同一虫体尺寸变化应平缓
        size_prev = w_prev + h_prev
        size_curr = w_det + h_det
        size_ratio = max(size_curr, size_prev) / (min(size_curr, size_prev) + 1e-6)
        # 正规化：理想比值为1，差异越大成本越高
        size_cost = min(abs(size_ratio - 1.0) / 0.5, 1.0)
        
        # 综合成本：速度和尺寸各占50%
        motion_cost = 0.5 * velocity_cost + 0.5 * size_cost
        
        return motion_cost

    def _match_gate(self, track):
        """关联搜索半径：目标丢失越久，允许的位置偏差越大（配合残留 ID 找回）。"""
        # 丢失越久半径越大，但增长在 GHOST_REID_MAX_LOST 帧后封顶
        grown = min(float(track.lost_frames), float(GHOST_REID_MAX_LOST))
        base = self.max_speed * (1.0 + grown / 20.0)
        limit = max(0.18 * max(self.width, self.height), self.max_speed * 1.5)
        return float(min(base, limit))

    def associate_detections_to_tracks(self, detections, masks):
        """
        改进的关联算法：添加运动一致性权重
        
        返回:
            (matches, unmatched_dets, unmatched_tracks)
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks))), [None] * len(detections)

        pred_bboxes = [track.predict() for track in self.tracks]
        det_centers = []
        det_features = []
        det_sizes = []
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            det_centers.append(((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2))
            det_sizes.append((bbox[2] - bbox[0]) + (bbox[3] - bbox[1]))
            mask = masks[i] if masks is not None and i < len(masks) else None
            feat = self.extract_features(mask, bbox)
            det_features.append(feat)

        num_tracks = len(self.tracks)
        num_dets = len(detections)
        confidences = detections.confidence if detections.confidence is not None else np.ones(num_dets)
        track_features = [track.get_feature() for track in self.tracks]

        cost_matrix = np.zeros((num_tracks, num_dets))
        
        # 新的权重配置（提高运动一致性权重）
        weight_iou = 0.25
        weight_motion = 0.45  # 运动一致性权重提高
        weight_direction = 0.15
        weight_appearance = 0.15
        
        for t in range(num_tracks):
            track = self.tracks[t]
            pred_bbox = pred_bboxes[t]
            
            for d in range(num_dets):
                det_bbox = detections.xyxy[d]
                
                # 距离检测：太远就不匹配（丢失越久搜索半径越大）
                center_dist = self.compute_center_distance(pred_bbox, det_bbox)
                if center_dist > self._match_gate(track):
                    cost_matrix[t, d] = 1e6
                    continue
                
                # 1. IoU 成本
                iou = self.compute_iou(pred_bbox, det_bbox)
                iou_cost = 1 - iou
                
                # 2. 运动一致性成本（新增）
                motion_cost = self.compute_motion_consistency_cost(
                    track, det_bbox, det_sizes[d]
                )
                
                # 3. 方向成本
                dir_cost = self.compute_direction_cost(track, det_centers[d])
                
                # 4. 外观成本
                app_dist = self.compute_appearance_distance(
                    track_features[t], det_features[d]
                )
                app_cost = min(app_dist / 5.0, 1.0)
                
                # 5. 置信度惩罚
                conf_penalty = (1 - confidences[d]) * 0.1
                
                # 综合成本
                combined_cost = (
                    weight_iou * iou_cost +
                    weight_motion * motion_cost +
                    weight_direction * dir_cost +
                    weight_appearance * app_cost +
                    conf_penalty
                )
                
                # 低 IoU 额外惩罚（残留找回时不加，否则长时间丢失后无法接回原 ID）
                if iou < self.iou_match_thresh and track.lost_frames == 0:
                    combined_cost += 0.5
                
                cost_matrix[t, d] = combined_cost

        # 匈牙利匹配
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        matches = []
        used_tracks = set()
        used_dets = set()
        
        for t, d in zip(row_idx, col_idx):
            if cost_matrix[t, d] < self.cost_threshold:
                matches.append((t, d))
                used_tracks.add(t)
                used_dets.add(d)

        unmatched_tracks = [t for t in range(num_tracks) if t not in used_tracks]
        unmatched_dets = [d for d in range(num_dets) if d not in used_dets]

        # 二次匹配（对于高 IoU 但未首次匹配的目标）
        for t in unmatched_tracks[:]:
            best_d = -1
            best_cost = 1e6
            
            for d in unmatched_dets:
                iou = self.compute_iou(pred_bboxes[t], detections.xyxy[d])
                if iou < self.iou_secondary:
                    continue
                
                center_dist = self.compute_center_distance(
                    pred_bboxes[t], detections.xyxy[d]
                )
                if center_dist > self._match_gate(self.tracks[t]):
                    continue
                
                motion_cost = self.compute_motion_consistency_cost(
                    self.tracks[t], detections.xyxy[d], det_sizes[d]
                )
                app_dist = self.compute_appearance_distance(
                    track_features[t], det_features[d]
                )
                app_cost = min(app_dist / 5.0, 1.0)
                
                # 二次匹配使用较宽松的成本
                cost = 0.3 * (1 - iou) + 0.4 * motion_cost + 0.3 * app_cost
                
                if cost < best_cost:
                    best_cost = cost
                    best_d = d
            
            if best_d != -1 and best_cost < 1.2:  # 二次匹配阈值稍高
                matches.append((t, best_d))
                used_tracks.add(t)
                used_dets.add(best_d)

        # ---- 第三阶段：视觉残留找回匹配 ----
        # 目标被遮挡/漏检后重新出现时，位置可能已偏离预测很远（IoU=0），
        # 用位置 + 尺寸 + 外观的宽松成本再匹配一次，尽量沿用原序号而不是新建 ID。
        unmatched_tracks = [t for t in range(num_tracks) if t not in used_tracks]
        unmatched_dets = [d for d in range(num_dets) if d not in used_dets]
        recovery_tracks = sorted(
            [t for t in unmatched_tracks if self.tracks[t].lost_frames >= 1],
            key=lambda t: -self.tracks[t].lost_frames
        )
        for t in recovery_tracks:
            if t not in unmatched_tracks:
                continue
            track = self.tracks[t]
            pred_bbox = pred_bboxes[t]
            gate = self._match_gate(track)
            size_prev = (pred_bbox[2] - pred_bbox[0]) + (pred_bbox[3] - pred_bbox[1])
            best_d, best_cost = -1, 1e6
            for d in list(unmatched_dets):
                det_bbox = detections.xyxy[d]
                dist = self.compute_center_distance(pred_bbox, det_bbox)
                if dist > gate:
                    continue
                size_det = det_sizes[d]
                ratio = max(size_prev, size_det) / (min(size_prev, size_det) + 1e-6)
                if ratio > 2.5:
                    continue
                app_dist = self.compute_appearance_distance(track_features[t], det_features[d])
                if app_dist > 8.0:
                    continue
                cost = (0.55 * min(dist / gate, 1.0) +
                        0.25 * min(abs(ratio - 1.0), 1.0) +
                        0.20 * min(app_dist / 5.0, 1.0))
                if cost < best_cost:
                    best_cost, best_d = cost, d
            if best_d >= 0 and best_cost < 1.0:
                matches.append((t, best_d))
                used_tracks.add(t)
                used_dets.add(best_d)
                unmatched_tracks.remove(t)
                unmatched_dets.remove(best_d)

        return matches, unmatched_dets, unmatched_tracks, det_features

    def update_sine_history(self, track_id, keypoints, velocity=None, curvature=None, track=None):
        """
        更新正弦运动历史与运动分类
        
        参数:
            track_id: 目标ID
            keypoints: (N, 2) 关键点数组
            velocity: (2,) 速度向量
            curvature: float 曲率
            track: KalmanBoxTracker 对象（用于提取速度信息）
        """
        if track_id not in self.track_histories:
            self.track_histories[track_id] = TrackHistory(min(int(self.max_lost_frames), 900), fps=self.fps)
            self.cycle_counts[track_id] = {
                'cycles': 0,
                'posture_count': 0,
                'swing_count': 0,
                'swings': 0,
                'observed_frames': 0,
                'observed_seconds': 0.0,
                'swing_freq_hz': 0.0,
                'live_swing_freq': 0.0,
                'inst_freq': 0.0,
                'cycle_freq': 0.0,
                'cycle_count': 0,
                'half_swing_count': 0,
                'count_mode': SWING_COUNT_MODE,
                'is_ghost': False,
                'lost_frames': 0,
                'bends_per_min': 0.0,
                'swing_activity': 0.0,
                'freq': 0.0,
                'avg_freq': 0.0,
                'average_frequency': 0.0,
                'live_freq': 0.0,
                'realtime_freq': 0.0,
                'swing_speed': 0.0,
                'head_swing_frequency': 0.0,
                'motion_type': 'static',
                'activity_score': 0.0
            }
        
        history = self.track_histories[track_id]
        
        # 计算速度（如果没有提供）
        if velocity is None and track is not None:
            velocity = track.get_motion_vector()
        if velocity is None:
            velocity = np.zeros(2, dtype=np.float32)
        
        # 计算曲率（如果没有提供）
        if curvature is None and keypoints is not None:
            curvature = curvature_of_centerline(keypoints)
        if curvature is None:
            curvature = 0.0
        
        # 计算头部角度
        head_angle = compute_head_angle(keypoints)
        
        # 添加到历史
        history.append(keypoints, head_angle, velocity=velocity, 
                      curvature=curvature, keypoints=keypoints)
        
        # 每个观测帧的摆动累计见 feed_swing_sample；这里在每获得一段
        # 连续中心线样本后全量刷新统计（含运动类型/活性分），避免旧实现
        # “整段视频计数恒为 0 / 用窗口频率乘全程时长”的问题。
        if len(history.centerlines) >= 6:
            self._sync_cycle_counts(track_id, history, full=True)


    def feed_swing_sample(self, track_id, keypoints, frame_no=None):
        """每个观测帧喂入一次摆动信号，累计摆动次数并轻量刷新统计。

        只使用最近一次真实提取的关键点（非预测外推点），避免预测漂移造成假摆动；
        但时间基准仍按每个观测帧累加，因此频率基于有效观测时间。
        """
        signals = swing_signals_from_keypoints(keypoints)
        if not signals or all(v is None for v in signals.values()):
            return
        if track_id not in self.track_histories:
            self.track_histories[track_id] = TrackHistory(min(int(self.max_lost_frames), 900), fps=self.fps)
        history = self.track_histories[track_id]
        sc = history.swing_counter
        if history.first_frame is None and frame_no is not None:
            history.first_frame = int(frame_no)
        if isinstance(sc, PostureSwingCounter):
            # 体态计数：直接用关键点算 C/S 形（尺度无关），不用头部角度信号
            new_swings = sc.feed_keypoints(keypoints, frame_no=frame_no)
        elif isinstance(sc, SwingFusion):
            new_swings = sc.feed(signals, frame_no=frame_no)
        else:
            new_swings = sc.feed(signals.get('bend'), frame_no=frame_no)
        history.swing_signals.append(signals.get('bend'))
        history.diag_append(frame_no, signals)
        # 画面叠加需要接近实时的计数/频率，无需每帧全量重算运动分类
        if new_swings > 0 or sc.observed_frames % 5 == 0:
            self._sync_cycle_counts(track_id, history, full=False)

    def _sync_cycle_counts(self, track_id, history, full=True):
        """把摆动计数/频率/活性写回 cycle_counts[track_id]（兼容旧字段）。

        频率口径：
          - freq / avg_freq：摆动次数 / 轨迹存续秒数（含丢失与未观测时间）。
            目标停止运动后被漏检时时间仍在推进，数值会自然下降，不会冻结在旧的高值。
          - avg_freq_observed：摆动次数 / 有效观测秒数（只算真正看到目标的帧），供对照。
        数据过期：超过 STALE_UNOBSERVED_SEC 秒没有新观测时，实时频率与活性按 0 输出，
        并置 is_stale，界面据此显示未观测时长。
        """
        sc = history.swing_counter
        observed_seconds = sc.observed_seconds()
        avg_freq_observed = sc.avg_freq() if observed_seconds > 0 else 0.0
        swing_count = sc.swing_count

        elapsed_seconds = 0.0
        first_frame = getattr(history, 'first_frame', None)
        if first_frame is not None and self.current_frame_idx:
            elapsed_seconds = max(0.0, (self.current_frame_idx - int(first_frame)) / max(self.fps, 1e-6))
        unobserved_seconds = max(0.0, elapsed_seconds - observed_seconds)
        base_seconds = elapsed_seconds if elapsed_seconds > 0.5 else observed_seconds
        avg_freq = (swing_count / base_seconds) if base_seconds > 0 else 0.0
        cycle_freq = (sc.cycle_count / base_seconds) if base_seconds > 0 else 0.0
        stale = unobserved_seconds > STALE_UNOBSERVED_SEC

        live_freq = 0.0 if stale else sc.live_freq()
        inst_freq = 0.0 if stale else sc.inst_freq()

        info = {
            'cycles': swing_count,
            'posture_count': swing_count,
            'swing_count': swing_count,
            'swings': swing_count,
            'half_swing_count': sc.half_swing_count,
            'cycle_count': sc.cycle_count,
            'swing_cycle_count': sc.cycle_count,
            'count_mode': sc.count_mode,
            'count_unit': getattr(sc, 'unit', ''),
            'posture_shape': getattr(sc, 'last_shape', ''),
            'posture_amp_pct': round(float(getattr(sc, 'last_amp_pct', 0.0)), 2),
            'observed_frames': sc.observed_frames,
            'observed_seconds': round(observed_seconds, 3),
            'elapsed_seconds': round(elapsed_seconds, 3),
            'unobserved_seconds': round(unobserved_seconds, 3),
            'is_stale': bool(stale),
            'swing_freq_hz': round(avg_freq, 3),
            'avg_freq_observed': round(avg_freq_observed, 3),
            'live_swing_freq': round(live_freq, 3),
            'cycle_freq': round(cycle_freq, 3),
            'inst_freq': round(inst_freq, 3),
            'bends_per_min': round(avg_freq * 60.0, 1),
            'swing_activity': round(min(1.0, avg_freq / sc.ref_max_hz), 3),
            'freq': round(avg_freq, 3),
            'avg_freq': round(avg_freq, 3),
            'average_frequency': round(avg_freq, 3),
            'live_freq': round(live_freq, 3),
            'realtime_freq': round(live_freq, 3),
            'swing_speed': round(avg_freq, 3),
            'head_swing_frequency': round(avg_freq, 3),
        }
        prev = self.cycle_counts.get(track_id, {})
        if full:
            info['motion_type'] = history.get_dominant_motion_type(window_frames=30)
            info['activity_score'] = 0.0 if stale else history.get_activity_score(window_sec=1.0)
        else:
            info['motion_type'] = prev.get('motion_type', 'static')
            info['activity_score'] = 0.0 if stale else prev.get('activity_score', 0.0)
        info['is_ghost'] = bool(prev.get('is_ghost', False))
        info['lost_frames'] = int(prev.get('lost_frames', 0))
        self.cycle_counts[track_id] = info

    def _ensure_sub_writer(self, track):
        """确认后才创建子视频写入器。

        未确认的瞬时误检不再产生垃圾子视频文件（旧实现一出生就建 worm_N.mp4，
        幽灵 ID 也会留下空文件）。
        """
        if self.sub_video_dir is None or track.sub_video_writer is not None:
            return
        sub_path = os.path.join(self.sub_video_dir, f"worm_{track.track_id}.mp4")
        try:
            track.sub_video_writer = create_compatible_video_writer(sub_path, self.fps, (200, 200))
        except Exception as e:
            print(f"子视频写入器警告: {e}")
            track.sub_video_writer = None

    def _should_block_birth(self, d_idx, detections, masks, matches, frame_idx):
        """出生门控：这个未匹配检测值不值得新建一条轨迹。

        1) 与某条已匹配检测的掩码重叠明显      -> 同一只虫的重复检测（去重没拦住的兜底）；
        2) 框被某条在跟踪目标的框包住且面积很小 -> 是一只虫身上的片段；
        3) 同一位置刚刚出生过（冷却窗口内）    -> 抑制检测抖动反复造 ID。
        """
        bbox = detections.xyxy[d_idx]
        mask = masks[d_idx] if masks is not None and d_idx < len(masks) else None
        area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        # 1) 掩码重复（同一只虫）
        if mask is not None and self.birth_block_mask_iou > 0:
            for _t_idx, d_match in matches:
                if d_match == d_idx:
                    continue
                other = masks[d_match] if d_match < len(masks) else None
                miou, _inter = _mask_iou_in_boxes(mask, other, bbox, detections.xyxy[d_match])
                if miou > self.birth_block_mask_iou:
                    return True

        # 2) 片段（框被在跟踪的框包住、面积明显更小）
        if area > 0:
            for _t_idx, d_match in matches:
                other_box = detections.xyxy[d_match]
                other_area = float((other_box[2] - other_box[0]) * (other_box[3] - other_box[1]))
                if other_area <= 0:
                    continue
                if area < self.birth_fragment_area * other_area and \
                        _contain_ratio(bbox, other_box) >= self.birth_fragment_contain:
                    return True

        # 3) 出生冷却（同一位置刚刚建过轨迹，且掩码明显重叠 = 同一只虫）
        return self._recent_birth_hit(bbox, mask, frame_idx)

    @staticmethod
    def _crop_mask_to_box(mask, bbox):
        """把掩码裁到框内（存入出生记录用，避免长期持有整幅大数组）。"""
        if mask is None:
            return None, (0, 0)
        m = _mask_to_binary(mask)
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(m.shape[1], int(bbox[2]))
        y2 = min(m.shape[0], int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None, (0, 0)
        return m[y1:y2, x1:x2].copy(), (x1, y1)

    def _recent_birth_hit(self, bbox, mask, frame_idx):
        """出生冷却：刚在同一位置出生过、且掩码明显重叠的，不再出生（抑制抖动反复造 ID）。

        只看掩码：两条贴在一起的虫"框"本来就重叠（实测 0.2~0.7），按框判会把另一条虫挡掉，
        那就又变成"两条虫合成一条"了。
        """
        if self.birth_cooldown_frames <= 0 or mask is None:
            return False
        mask_bin = _mask_to_binary(mask)
        area = int(np.count_nonzero(mask_bin))
        if area <= 0:
            return False
        for prev_frame, prev_box, prev_crop, prev_origin, prev_area in self._recent_births:
            if frame_idx - prev_frame > self.birth_cooldown_frames:
                continue
            if prev_crop is None or prev_area <= 0:
                continue
            ix1 = max(int(bbox[0]), int(prev_box[0]))
            iy1 = max(int(bbox[1]), int(prev_box[1]))
            ix2 = min(int(bbox[2]), int(prev_box[2]))
            iy2 = min(int(bbox[3]), int(prev_box[3]))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            ox, oy = prev_origin
            new_patch = mask_bin[iy1:iy2, ix1:ix2]
            prev_patch = prev_crop[max(0, iy1 - oy):max(0, iy2 - oy),
                                   max(0, ix1 - ox):max(0, ix2 - ox)]
            h = min(new_patch.shape[0], prev_patch.shape[0])
            w = min(new_patch.shape[1], prev_patch.shape[1])
            if h <= 0 or w <= 0:
                continue
            inter = int(np.count_nonzero(new_patch[:h, :w] & prev_patch[:h, :w]))
            if inter > 0 and inter / float(min(area, prev_area)) > self.birth_cooldown_overlap:
                return True
        return False

    def _resolve_shared_detections(self, matches, unmatched_tracks, detections):
        """贴合共享：两条虫贴在一起、模型只给了一条检测时，别把另一条判成丢失。

        条件：已确认、上一帧还在（lost_frames <= SHARED_MAX_LOST_FRAMES）、
        预测框与某条"已匹配检测"的框重叠 >= shared_iou。
        这类轨迹保持存活（不进入残留、不丢 ID、继续写子视频），
        但本帧不喂摆动计数 —— 两虫共用一条检测，关键点/幅度都不可信。
        """
        self.shared_ids = set()
        self.shared_live_ids = set()
        shared_t_idx = []
        if self.shared_iou <= 0 or not matches or not unmatched_tracks:
            return self.shared_ids, self.shared_live_ids, shared_t_idx

        shared_per_det = {}
        for t_idx in sorted(unmatched_tracks, key=lambda i: -self.tracks[i].lost_frames):
            track = self.tracks[t_idx]
            if track.track_id not in self.confirmed_ids:
                continue
            if track.lost_frames > self.shared_max_lost_frames:
                continue
            pred_bbox = track.last_predicted_bbox if track.last_predicted_bbox is not None else track.get_state()
            best_d, best_iou = -1, 0.0
            for _m_t_idx, m_d_idx in matches:
                if shared_per_det.get(m_d_idx, 0) >= self.shared_per_det_max:
                    continue
                iou = self.compute_iou(pred_bbox, detections.xyxy[m_d_idx])
                if iou > best_iou:
                    best_iou, best_d = iou, m_d_idx
            if best_d >= 0 and best_iou >= self.shared_iou:
                shared_per_det[best_d] = shared_per_det.get(best_d, 0) + 1
                self.shared_ids.add(int(track.track_id))
                for m_t_idx, m_d_idx in matches:
                    if m_d_idx == best_d:
                        self.shared_live_ids.add(int(self.tracks[m_t_idx].track_id))
                        break
                shared_t_idx.append(t_idx)
        return self.shared_ids, self.shared_live_ids, shared_t_idx

    def _apply_shared_tracks(self, shared_t_idx, frame, frame_idx):
        """贴合共享轨迹的收尾：保持存活（不进入残留）、框停在原地、继续写子视频与轨迹点。"""
        for t_idx in shared_t_idx:
            track = self.tracks[t_idx]
            tid = int(track.track_id)
            track.lost_frames = 0
            track.freeze()                     # 框停在最后一次位置，不按速度外推
            self.ghost_ids.discard(tid)
            self.id_last_seen[tid] = frame_idx
            info = self.cycle_counts.get(tid)
            if info is not None:
                info['is_ghost'] = False
                info['lost_frames'] = 0
                info['is_shared'] = True
            pred_bbox = track.last_predicted_bbox if track.last_predicted_bbox is not None else track.get_state()
            if pred_bbox is not None:
                self.track_history[tid].append(((pred_bbox[0] + pred_bbox[2]) / 2,
                                                (pred_bbox[1] + pred_bbox[3]) / 2))
            if frame is not None and track.sub_video_writer is not None:
                roi_resized = self.extract_stable_roi(frame, track, output_size=(200, 200))
                if roi_resized is not None:
                    track.sub_video_writer.write(roi_resized)

    def _handle_lost_tracks(self, track_indices, frame, frame_idx):
        """未匹配/丢失轨迹的统一处理：丢失计数、冻结位置、视觉残留。

        残留窗口内：ID 继续显示，框停在最后位置（freeze），并继续写子视频。
        超出窗口：不再绘制，但 ID 仍保留（TRACK_PERSISTENCE_SEC 内不会被回收）。
        """
        for t_idx in track_indices:
            track = self.tracks[t_idx]
            track.lost_frames += 1
            tid = track.track_id
            info = self.cycle_counts.get(tid)
            if track.lost_frames <= self.ghost_overlay_frames:
                self.ghost_ids.add(tid)
                track.freeze()
                if frame is not None and track.sub_video_writer is not None:
                    roi_resized = self.extract_stable_roi(frame, track, output_size=(200, 200))
                    if roi_resized is not None:
                        track.sub_video_writer.write(roi_resized)
                is_ghost = True
            else:
                self.ghost_ids.discard(tid)
                is_ghost = False
            if info is not None:
                info['is_ghost'] = is_ghost
                info['is_shared'] = False
                info['lost_frames'] = int(track.lost_frames)
            pred_bbox = track.last_observed_bbox if track.last_observed_bbox is not None else track.last_predicted_bbox
            if pred_bbox is not None:
                pred_centroid = ((pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2)
                self.track_history[tid].append(pred_centroid)

    def _prune_lost_tracks(self):
        """回收超过 ID 保留时长仍未找回的轨迹（ID 只增不减，不会被复用）。"""
        remaining_tracks = []
        for t in self.tracks:
            if t.lost_frames > self.max_lost_frames:
                self.ghost_ids.discard(t.track_id)
                self.confirmed_ids.discard(t.track_id)
                if t.sub_video_writer:
                    t.sub_video_writer.release()
            else:
                remaining_tracks.append(t)
        self.tracks = remaining_tracks

    # ---------- B1：ROI 低阈值找回（针对停滞/被遮挡的已确认目标） ----------
    def iter_recover_candidates(self, max_items=4):
        """需要做 ROI 找回的轨迹：已确认、丢失若干帧、且未超过找回时限。"""
        out = []
        for track in self.tracks:
            if track.track_id not in self.confirmed_ids:
                continue
            if track.lost_frames < ROI_RECOVER_MIN_LOST:
                continue
            last = self.id_last_seen.get(track.track_id)
            if last is not None and (self.current_frame_idx - last) / max(self.fps, 1e-6) > ROI_RECOVER_MAX_SEC:
                continue
            out.append(track)
            if len(out) >= max_items:
                break
        return out

    def get_recover_roi(self, track):
        """按最后观测位置取一块外扩 ROI，返回 (x1, y1, x2, y2)。"""
        bbox = np.asarray(track.last_observed_bbox, dtype=float)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        side = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * (1.0 + ROI_RECOVER_PAD)
        side = max(ROI_RECOVER_MIN_SIZE, side * 1.4)
        x1 = int(max(0, cx - side / 2))
        y1 = int(max(0, cy - side / 2))
        x2 = int(min(self.width, x1 + side))
        y2 = int(min(self.height, y1 + side))
        return x1, y1, x2, y2

    def attach_recovered(self, track, bbox, mask, frame_idx, frame=None):
        """ROI 找回命中：把检测并回原轨迹（不走全局关联，保证不换序号）。"""
        feat = self.extract_features(mask, bbox) if mask is not None else np.zeros(11)
        track.update(np.asarray(bbox, dtype=np.float32), feat)
        track.confirmed = True
        self.confirmed_ids.add(track.track_id)
        self.ghost_ids.discard(track.track_id)
        self.id_last_seen[track.track_id] = frame_idx
        centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.track_history[track.track_id].append(centroid)
        info = self.cycle_counts.get(track.track_id)
        if info is not None:
            info['is_ghost'] = False
            info['lost_frames'] = 0
        if frame is not None and track.sub_video_writer is not None:
            roi_resized = self.extract_stable_roi(frame, track, output_size=(200, 200))
            if roi_resized is not None:
                track.sub_video_writer.write(roi_resized)

    def recover_missing_targets(self, frame, model, device, frame_idx):
        """对长时间未匹配的已确认轨迹，在最后位置做低阈值 ROI 推理找回（可关：ROI_RECOVER=0）。"""
        if not ROI_RECOVER_ENABLED or frame is None or model is None:
            return 0
        if frame_idx % max(1, ROI_RECOVER_INTERVAL) != 0:
            return 0
        recovered = 0
        for track in self.iter_recover_candidates(max_items=ROI_RECOVER_MAX_ITEMS):
            x1, y1, x2, y2 = self.get_recover_roi(track)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            roi = frame[y1:y2, x1:x2]
            try:
                res = model(roi, imgsz=ROI_RECOVER_IMGSZ, conf=ROI_RECOVER_CONF,
                            verbose=False, device=device)[0]
            except Exception:
                continue
            dets = sv.Detections.from_ultralytics(res)
            if len(dets) == 0:
                continue
            tb = np.asarray(track.last_observed_bbox, dtype=float)
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            best, best_cost = None, 1e9
            for i in range(len(dets)):
                b = dets.xyxy[i]
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                w, h = b[2] - b[0], b[3] - b[1]
                dist = float(np.hypot(cx - tcx, cy - tcy))
                ratio = max(w, tw) / max(min(w, tw), 1e-6) if tw > 0 else 1.0
                if dist > max(40.0, 1.5 * max(tw, th)) or ratio > 2.0:
                    continue
                cost = dist / max(tw, th, 1e-6) + abs(ratio - 1.0)
                if cost < best_cost:
                    best_cost, best = cost, i
            if best is None:
                continue
            bbox = np.asarray(dets.xyxy[best], dtype=np.float32) + np.array([x1, y1, x1, y1], dtype=np.float32)
            mask = None
            if getattr(dets, 'mask', None) is not None and len(dets.mask) > best:
                m = np.asarray(dets.mask[best])
                m2 = np.any(m > 0, axis=0) if (m.ndim == 3 and m.shape[0] > 1) else (m > 0)
                full = np.zeros((self.height, self.width), dtype=bool)
                h2 = min(self.height - y1, m2.shape[0])
                w2 = min(self.width - x1, m2.shape[1])
                full[y1:y1 + h2, x1:x1 + w2] = m2[:h2, :w2]
                mask = full
            self.attach_recovered(track, bbox, mask, frame_idx, frame)
            recovered += 1
        return recovered

    def update_tracking(self, detections, masks, frame_idx, frame=None):
        self.current_frame_idx = frame_idx
        """
        主更新函数，返回带有tracker_id的detections
        若提供frame参数，则会写入子视频
        """
        # 掩码兜底：调用方只传了带 mask 的 Detections 时，也要走"掩码优先"的去重，
        # 否则会退回只看框的 NMS，把贴合的两条虫误删成一条。
        if masks is None:
            masks = getattr(detections, 'mask', None)
        # 每帧重置一次本帧状态，避免上一帧的贴合/被压住标记残留到这一帧
        self.shared_ids = set()
        self.shared_live_ids = set()
        self.shadowed_ids = set()
        if len(detections) == 0:
            for track in self.tracks:
                track.predict()          # 本帧没有检测：先做一次卡尔曼预测（每帧只预测一次）
            self._handle_lost_tracks(list(range(len(self.tracks))), frame, frame_idx)
            self._prune_lost_tracks()
            self.live_track_count = 0    # 本帧没有任何真实观测
            ghost_tracks = [t for t in self.tracks if t.track_id in self.ghost_ids]
            empty_dets = sv.Detections.empty()
            empty_dets.tracker_id = np.array([], dtype=int)
            return self.build_prediction_detections(ghost_tracks, empty_dets)

        detections = self.apply_nms(detections, masks)
        # 去重后必须按新的检测顺序重取掩码：否则框与掩码会错位（旧实现用旧 masks 下标索引，
        # 一旦 apply_nms 删掉了检测，后面的框就会配上别的虫的掩码，中心线/摆动计数跟着出错）。
        if getattr(detections, 'mask', None) is not None:
            masks = detections.mask
        elif masks is not None and len(masks) != len(detections):
            masks = None
        matches, unmatched_dets, unmatched_tracks, det_features = self.associate_detections_to_tracks(detections, masks)

        # ---- 贴合共享：两条虫贴在一起、模型只给了一条检测 ----
        # 必须在"丢失计数"之前判定：这类轨迹要按存活处理，不能进入残留。
        shared_t_idx = []
        if self.shared_iou > 0:
            _s_ids, _s_live, shared_t_idx = self._resolve_shared_detections(
                matches, unmatched_tracks, detections)
            if shared_t_idx:
                unmatched_tracks = [t for t in unmatched_tracks if t not in shared_t_idx]

        # 更新匹配的轨迹（复用关联阶段已提取的特征，避免重复计算）
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            bbox = detections.xyxy[d_idx]
            mask = masks[d_idx] if masks is not None else None
            feat = det_features[d_idx]
            if feat is None:
                feat = self.extract_features(mask, bbox)
            track.update(bbox, feat)
            if not track.confirmed and track.hits >= self.confirm_hits:
                track.confirmed = True
                self.confirmed_ids.add(track.track_id)
            # 子视频只给"稳定观测"的目标建：命中帧数达到 sub_video_min_hits（默认 15 帧≈0.5s），
            # 避免瞬时噪声 ID 在下载的 ZIP/列表里变成"多出来的虫子"。
            if track.sub_video_writer is None and track.confirmed \
                    and track.hits >= self.sub_video_min_hits:
                self._ensure_sub_writer(track)
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            self.track_history[track.track_id].append(centroid)
            self.id_last_seen[track.track_id] = frame_idx
            # 目标已被找回：结束视觉残留状态
            self.ghost_ids.discard(track.track_id)
            _prev_info = self.cycle_counts.get(track.track_id)
            if _prev_info is not None:
                _prev_info['is_ghost'] = False
                _prev_info['lost_frames'] = 0
                _prev_info['is_shared'] = False

            # 贴合共享帧：这条检测是两条虫的重叠区域，关键点与摆动幅度都不可信 —— 暂停计数
            # （两虫共用一条检测，照常计数会让两条虫算出趋同甚至重复的频率）
            if track.track_id in self.shared_live_ids:
                if track.sub_video_writer is not None and frame is not None:
                    roi_resized = self.extract_stable_roi(frame, track, output_size=(200, 200))
                    if roi_resized is not None:
                        track.sub_video_writer.write(roi_resized)
                continue

            # 姿态分析：提取中心线并更新历史（限定在稳定窗口内，避免全图骨架化）
            if mask is not None and (
                track.last_keypoint_update_frame < 0
                or frame_idx - track.last_keypoint_update_frame >= KEYPOINT_UPDATE_INTERVAL
            ):
                cw = track.get_stable_crop_window(self.width, self.height)
                centerline_raw, keypoints_raw = self._extract_centerline_in_window(mask, *cw)
                if keypoints_raw is not None:
                    # 保证该线虫头部方向与上一帧一致（骨架 BFS 端点可能翻转）
                    centerline_raw, keypoints_raw = orient_pose_pair(
                        centerline_raw, keypoints_raw, track.orient_head
                    )
                    track.update_keypoints(keypoints_raw, centerline_raw)
                    track.last_observed_keypoints = keypoints_raw   # 摆动计数只用真实关键点
                    self.update_sine_history(track.track_id, keypoints_raw)
                    track.last_keypoint_update_frame = frame_idx
            elif track.last_keypoints is not None:
                track.update_keypoints(track.predict_keypoints(), track.last_centerline)

            # 每个观测帧喂入一次摆动信号（沿用最近一次真实关键点），保证时间基准正确
            self.feed_swing_sample(track.track_id, track.last_observed_keypoints, frame_idx)

            if track.sub_video_writer is not None and frame is not None:
                roi_resized = self.extract_stable_roi(frame, track, output_size=(200, 200))
                if roi_resized is not None:
                    track.sub_video_writer.write(roi_resized)

        # 创建新轨迹（带出生门控：同一只虫不重复建 ID，瞬时误检不占号）
        new_tracks = []
        created_pairs = []      # (d_idx, new_track)：给检测回填 tracker_id 用
        for d_idx in unmatched_dets:
            bbox = detections.xyxy[d_idx]
            mask = masks[d_idx] if masks is not None and d_idx < len(masks) else None
            if self._should_block_birth(d_idx, detections, masks, matches, frame_idx):
                continue        # 该检测已被现有轨迹解释 / 是片段 / 刚在同一位置建过
            feat = det_features[d_idx]
            if feat is None:
                feat = self.extract_features(mask, bbox)

            # 子视频写入器不在这里创建：等确认后再建（见 _ensure_sub_writer），
            # 避免从未确认的幽灵 ID 留下垃圾子视频文件。
            new_track = KalmanBoxTracker(bbox, feat, track_id=self.next_id, sub_video_writer=None)
            new_track.track_id = self.next_id
            self.next_id += 1
            new_track.confirmed = False
            new_tracks.append(new_track)
            created_pairs.append((d_idx, new_track))
            _crop, _origin = self._crop_mask_to_box(mask, bbox)
            self._recent_births.append((
                frame_idx,
                np.asarray(bbox, dtype=np.float32).copy(),
                _crop,
                _origin,
                int(np.count_nonzero(_mask_to_binary(mask))) if mask is not None else 0,
            ))
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            self.track_history[new_track.track_id].append(centroid)
            self.id_last_seen[new_track.track_id] = frame_idx

            # 新轨迹也立即提取一次中心线（限定在轨迹窗口内，降低首帧开销）
            if mask is not None:
                cw = new_track.get_stable_crop_window(self.width, self.height)
                centerline_raw, keypoints_raw = self._extract_centerline_in_window(mask, *cw)
                if keypoints_raw is not None:
                    centerline_raw, keypoints_raw = orient_pose_pair(
                        centerline_raw, keypoints_raw, new_track.orient_head
                    )
                    new_track.update_keypoints(keypoints_raw, centerline_raw)
                    new_track.last_observed_keypoints = keypoints_raw
                    self.update_sine_history(new_track.track_id, keypoints_raw)
                    self.feed_swing_sample(new_track.track_id, keypoints_raw, frame_idx)
                    new_track.last_keypoint_update_frame = frame_idx

        self.tracks.extend(new_tracks)

        # 贴合共享轨迹的收尾：保持存活、框停在原地、继续写子视频与轨迹点
        if shared_t_idx:
            self._apply_shared_tracks(shared_t_idx, frame, frame_idx)

        # 增加未匹配轨迹的丢失计数
        self._handle_lost_tracks(unmatched_tracks, frame, frame_idx)

        # 为detections分配tracker_id（未通过出生门控的检测保持 -1：本帧不产生轨迹）
        tracker_ids = [-1] * len(detections)
        for t_idx, d_idx in matches:
            tracker_ids[d_idx] = self.tracks[t_idx].track_id
        for d_idx, new_track in created_pairs:
            tracker_ids[d_idx] = new_track.track_id

        detections.tracker_id = np.array(tracker_ids, dtype=int)

        predicted_tracks = []
        predicted_track_ids = []
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            # 只在残留绘制窗口内把预测框（残留框）交给前端/画面；ID 仍在后台保留
            if track.lost_frames <= self.ghost_overlay_frames:
                predicted_tracks.append(track)
                predicted_track_ids.append(track.track_id)

        # 贴合共享的轨迹同样要画出来（两条虫贴住时各自保留自己的框，避免"合成一条"）
        for t_idx in shared_t_idx:
            track = self.tracks[t_idx]
            predicted_tracks.append(track)
            predicted_track_ids.append(track.track_id)

        if len(predicted_tracks) > 0:
            detections = self.build_prediction_detections(predicted_tracks, detections)
            tracker_ids = list(tracker_ids) + predicted_track_ids
        else:
            tracker_ids = list(tracker_ids)

        detections.tracker_id = np.array(tracker_ids, dtype=int)

        self._prune_lost_tracks()

        # 平滑框（使用卡尔曼状态）
        tracks_by_id = {t.track_id: t for t in self.tracks}
        smoothed_bboxes = []
        for i, tid in enumerate(tracker_ids):
            if tid is not None:
                track = tracks_by_id.get(tid)
                if track is not None:
                    if i < len(detections.xyxy) - len(predicted_track_ids):
                        smoothed_bboxes.append(_normalize_bbox(track.get_state()))
                    else:
                        smoothed_bboxes.append(_normalize_bbox(track.last_predicted_bbox if track.last_predicted_bbox is not None else track.get_state()))
                else:
                    smoothed_bboxes.append(_normalize_bbox(detections.xyxy[i]))
            else:
                smoothed_bboxes.append(_normalize_bbox(detections.xyxy[i]))
        if len(smoothed_bboxes) > 0:
            detections.xyxy = np.vstack(smoothed_bboxes).astype(np.float32)

        # 残留框压在实时框上时不再绘制（标记 shadowed）。
        # 必须放在"平滑框"之后：画面上真正显示的是平滑后的框，
        # 用原始检测框判定会漏掉一批（实测 63 帧里有 43 帧因此没被抑制）。
        self.shadowed_ids = set()
        if self.ghost_suppress_iou > 0 and len(detections) > 0:
            live_boxes = []
            ghost_boxes = {}
            for i, tid in enumerate(detections.tracker_id):
                tid = int(tid)
                if tid <= 0:
                    continue
                if tid in self.ghost_ids:
                    ghost_boxes[tid] = detections.xyxy[i]
                elif tid in self.confirmed_ids:
                    live_boxes.append(detections.xyxy[i])
            if live_boxes:
                for tid, ghost_box in ghost_boxes.items():
                    if any(self.compute_iou(ghost_box, lb) > self.ghost_suppress_iou
                           for lb in live_boxes):
                        self.shadowed_ids.add(tid)

        # 本帧"真正在跟踪"的目标数：已确认 + 本帧有观测（匹配）或处于贴合共享。
        # 界面"活跃目标"用这个数，而不是 len(self.tracks)（后者含后台保留 600 秒的残留轨迹）。
        self.live_track_count = sum(
            1 for t in self.tracks
            if int(t.track_id) in self.confirmed_ids and t.lost_frames == 0
        )

        return detections

# ==================== 初始化模型函数 ====================
def init_model(model_path):
    """加载 YOLO 模型，返回 (model, device)。

    设备选择优先级：
      1) 环境变量 DEVICE（例如 DEVICE=cpu / DEVICE=0 / DEVICE=cuda:0）
      2) 自动判断：PyTorch 支持 CUDA 就用 GPU，否则用 CPU
    """
    device = (os.environ.get('DEVICE') or '').strip()
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wants_gpu = device.lower().startswith('cuda') or device.isdigit()
    if wants_gpu and not torch.cuda.is_available():
        print('[警告] 指定了 GPU，但当前 PyTorch 不支持 CUDA（装的可能是 CPU 版），已回退到 CPU。')
        print('       想用 GPU：运行 4-安装GPU训练依赖.bat，装好后重启本服务。')
        device = 'cpu'
    if device != 'cpu':
        torch.backends.cudnn.benchmark = True   # 固定输入尺寸下可提速
    model = YOLO(model_path)
    model.to(device)
    if device != 'cpu':
        try:
            print('[信息] 推理设备: GPU %s' % torch.cuda.get_device_name(0))
        except Exception:
            print('[信息] 推理设备: GPU')
    else:
        print('[信息] 推理设备: CPU')
    return model, device

# ==================== 画面标注工具（清晰化） ====================
# 标注统一白色 + 深色描边（不用彩色），保证清晰又不与画面内容抢色
_MARK_COLOR = (255, 255, 255)
_MARK_OUTLINE = (18, 18, 18)


def _draw_label_box(frame, text, org, color, scale=0.7, thickness=2, pad=8):
    """写带深色半透明底的标签，白色加粗抗锯齿文字，任何背景下都清晰。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    h, w = frame.shape[:2]
    x1 = int(max(0, min(org[0], w - tw - 2 * pad - 2)))
    y1 = int(max(0, min(org[1], h - th - base - 2 * pad - 2)))
    x2 = x1 + tw + 2 * pad
    y2 = y1 + th + base + pad
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        dark = np.zeros_like(roi)
        dark[:] = (26, 24, 22)
        cv2.addWeighted(dark, 0.74, roi, 0.26, 0, roi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x1 + pad, y1 + pad + th), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return x1, y1, x2, y2


def _draw_contour_overlay(frame, tracked_dets, tracker, overlay_mode=DEFAULT_OVERLAY_MODE):
    """绘制每个线虫的轮廓、矩形框与编号标签（清晰化版）。

    - 框与标签使用按 ID 分配的高对比颜色，并加深色描边，浅色背景下也醒目；
    - 标签优先贴在框正上方，用同色短线连到框；上方空间不足时贴在框内顶部，
      保证序号与线虫紧密对应；
    - 标签为深底白字、加粗抗锯齿，远处也能看清；
    - overlay_mode='debug'（实时预览）时残留目标完全不上屏；
      overlay_mode='release'（结果视频）时残留目标与真实检测一样画实线框。
    """
    if tracked_dets is None or len(tracked_dets) == 0:
        return frame

    tracker_ids = getattr(tracked_dets, 'tracker_id', None)
    masks = getattr(tracked_dets, 'mask', None)
    xyxy = getattr(tracked_dets, 'xyxy', None)
    if tracker_ids is None:
        return frame

    ghost_ids = getattr(tracker, 'ghost_ids', set())
    confirmed_ids = getattr(tracker, 'confirmed_ids', set())
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2

    for idx, tid in enumerate(tracker_ids):
        if tid is None:
            continue
        tid_int = int(tid)
        if tid_int not in confirmed_ids:
            continue    # 未确认的瞬时误检不绘制
        if tid_int in getattr(tracker, 'shadowed_ids', set()):
            continue    # 被实时框压住的残留 ID：数据保留，但不画第二个框（避免看着像重复识别）
        cycle_info = tracker.cycle_counts.get(tid_int, {})
        color = _MARK_COLOR
        is_ghost = tid_int in ghost_ids

        # 实时预览（debug）不把残留目标画到画面上：框、标签一概不画，画面里只保留真实检测
        if is_ghost and overlay_mode == OVERLAY_MODE_DEBUG:
            continue

        # 1) 轮廓：深色描边 + ID 颜色
        if masks is not None and len(masks) > idx:
            mask = masks[idx]
            if mask is not None:
                mask_array = np.asarray(mask)
                if mask_array.ndim == 3 and mask_array.shape[0] > 1:
                    binary = np.any(mask_array > 0, axis=0)
                else:
                    binary = mask_array > 0
                binary = binary.astype(np.uint8)
                if binary.any():
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(frame, contours, -1, _MARK_OUTLINE, 5, cv2.LINE_AA)
                        cv2.drawContours(frame, contours, -1, _MARK_COLOR, 3, cv2.LINE_AA)

        if xyxy is None or len(xyxy) <= idx:
            continue
        x1, y1, x2, y2 = [int(v) for v in xyxy[idx]]

        # 2) 矩形框：残留目标（预测保持）同样画实线框，不再画虚线，避免结果视频里出现断线框
        cv2.rectangle(frame, (x1, y1), (x2, y2), _MARK_OUTLINE, 6, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _MARK_COLOR, 3, cv2.LINE_AA)

        # 3) 标签：残留目标在结果视频里保留“残留 丢Nf”说明
        live_freq = cycle_info.get('live_freq', cycle_info.get('realtime_freq', cycle_info.get('freq', 0.0)))
        swing_count = cycle_info.get('swing_count', cycle_info.get('cycles', 0))
        if is_ghost:
            text = "ID %d  残留 丢%df" % (tid_int, int(cycle_info.get('lost_frames', 0)))
        else:
            text = "ID %d  %d次  %.2fHz" % (tid_int, int(swing_count), float(live_freq))

        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        box_h = th + base + 16
        if y1 - box_h - 10 >= 0:
            lx, ly = x1, y1 - box_h - 8
            _draw_label_box(frame, text, (lx, ly), color, scale=scale, thickness=thickness)
            cv2.line(frame, (x1 + 8, ly + box_h), (x1 + 8, y1), color, 2, cv2.LINE_AA)
        else:
            _draw_label_box(frame, text, (x1 + 2, y1 + 2), color, scale=scale, thickness=thickness)

    return frame


# ==================== 逐帧处理函数（供 WebSocket 调用） ====================
def _detect_and_track(frame, model, device, tracker, frame_idx):
    """推理 + 过滤 + 跟踪 + ROI 找回，返回本帧的跟踪结果（不做任何绘制）。

    说明：ultralytics 8.4 起 half 参数已废弃（提示改用 quantize），此处不传；
    GPU 推理如需 FP16 提速，可在下面 model(...) 调用加 half=True（会有 deprecated 提示）。
    """
    results = model(frame, imgsz=INFER_IMGSZ, conf=INFER_CONF,
                    verbose=False, device=device)[0]
    detections = sv.Detections.from_ultralytics(results)

    # 过滤小目标和低置信度目标
    if len(detections) > 0:
        x1, y1, x2, y2 = detections.xyxy[:, 0], detections.xyxy[:, 1], detections.xyxy[:, 2], detections.xyxy[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        conf = detections.confidence if detections.confidence is not None else np.ones(len(detections))
        keep = (areas >= tracker.MIN_AREA) & (conf >= tracker.MIN_CONFIDENCE)
        detections = detections[keep]

    masks = detections.mask if hasattr(detections, 'mask') else None
    tracked_dets = tracker.update_tracking(detections, masks, frame_idx, frame=frame)

    # B1：对停止运动后被长期漏检的已确认轨迹，在最后位置做低阈值 ROI 找回（ROI_RECOVER=0 可关闭）
    try:
        tracker.recover_missing_targets(frame, model, device, frame_idx)
    except Exception as exc:
        print(f"ROI 找回警告: {exc}")

    return tracked_dets


def _active_track_state(tracker, min_activity=0.25):
    """一次性算出本帧要绘制/统计的目标集合，避免在多个循环里重复调用 is_active。"""
    # 视觉残留目标不受活性阈值过滤，保证识别中断时序号不中断
    ghost_ids = set(getattr(tracker, 'ghost_ids', set()))
    active_ids_all = set(ghost_ids)
    confirmed_ids = set(getattr(tracker, 'confirmed_ids', set()))
    for _tid, _hist in tracker.track_histories.items():
        # 已确认的目标即使静止（活性低）也保留显示，避免看起来像识别中断
        if _tid in confirmed_ids or _hist.is_active(min_activity=min_activity):
            active_ids_all.add(_tid)
    return ghost_ids, active_ids_all, confirmed_ids


def render_overlay(frame, tracked_dets, tracker, frame_idx, state, overlay_mode=DEFAULT_OVERLAY_MODE):
    """按叠加模式渲染标注帧。

    - debug（实时预览）：残留目标完全不上屏，画面里只出现真实检测；
    - release（结果视频）：残留目标与真实检测一样画实线框，并保留“残留 丢Nf”标签。
    """
    ghost_ids, active_ids_all, _ = state
    shadowed_ids = set(getattr(tracker, 'shadowed_ids', set()))

    # 可视化
    annotated_frame = frame.copy()
    annotated_frame = _draw_contour_overlay(annotated_frame, tracked_dets, tracker,
                                            overlay_mode=overlay_mode)
    # 轮廓、边框与计数/频率标签统一由 _draw_contour_overlay 绘制，
    # 不再重复调用 sv 的 Mask/Box/Label 标注器（每帧省一次全图绘制与对象创建）。

    # 绘制轨迹线（可选，仅活跃目标）
    for tid, history in tracker.track_history.items():
        if tid not in active_ids_all:
            continue  # 跳过低活性目标
        if tid in ghost_ids and overlay_mode == OVERLAY_MODE_DEBUG:
            continue  # 实时预览不显示残留目标的轨迹残留点
        if tid in shadowed_ids:
            continue  # 被实时框压住的残留 ID 也不画轨迹点

        history_list = list(history)
        if len(history_list) > 1:
            points = []
            for point in history_list:
                if isinstance(point, (tuple, list)) and len(point) == 2:
                    points.append([int(point[0]), int(point[1])])
            
            if len(points) > 1:
                points = np.array(points, dtype=np.int32)
                for j in range(1, len(points)):
                    color_intensity = int(255 * (j / len(points)))
                    cv2.line(annotated_frame, tuple(points[j-1]), tuple(points[j]), (0, color_intensity, 0), 2)

    # 在帧上显示每个ID的详细信息
    for tid, cycle_info in tracker.cycle_counts.items():
        if tid not in active_ids_all:
            continue  # 不显示低活性目标
        if tid in ghost_ids and overlay_mode == OVERLAY_MODE_DEBUG:
            continue  # 实时预览不显示残留目标的详细信息标签
        if tid in shadowed_ids:
            continue  # 被实时框压住的残留 ID 也不画详细信息标签

        if hasattr(tracked_dets, 'tracker_id') and tracked_dets.tracker_id is not None:
            idx = np.where(tracked_dets.tracker_id == tid)[0]
            if len(idx) > 0:
                bbox = tracked_dets.xyxy[idx[0]]
                x1, y1, x2, y2 = map(int, bbox)
                
                # 显示累计摆动次数、频率、次/分钟、运动类型、活性
                motion_type = cycle_info.get('motion_type', 'unknown')
                activity = cycle_info.get('activity_score', 0.0)
                cycles = cycle_info.get('cycles', 0)
                freq = cycle_info.get('live_freq', cycle_info.get('realtime_freq', cycle_info.get('freq', 0.0)))
                bpm = cycle_info.get('bends_per_min', 0.0)

                text = f"#{cycles} {freq:.2f}Hz {bpm:.0f}/min {motion_type} {activity:.0%}"
                _draw_label_box(annotated_frame, text, (x1, min(y2 + 8, annotated_frame.shape[0] - 34)),
                                _MARK_COLOR, scale=0.55, thickness=1, pad=6)

    # 添加帧信息和统计
    active_count = sum(1 for tid in tracker.cycle_counts if tid in active_ids_all)
    
    _draw_label_box(annotated_frame, f"Frame {frame_idx}", (10, 8),
                _MARK_COLOR, scale=0.65, thickness=2, pad=7)
    _draw_label_box(annotated_frame, f"Active IDs {active_count}/{len(tracker.tracks)}", (10, 46),
                _MARK_COLOR, scale=0.65, thickness=2, pad=7)

    return annotated_frame


def process_frame(frame, model, device, tracker, frame_idx, min_activity=0.25,
                  overlay_mode=DEFAULT_OVERLAY_MODE):
    """处理单帧，返回标注后的帧和当前跟踪数据（单模式，沿用既有调用方式）。

    返回:
        (annotated_frame, track_data)
    """
    tracked_dets = _detect_and_track(frame, model, device, tracker, frame_idx)
    state = _active_track_state(tracker, min_activity)
    annotated_frame = render_overlay(frame, tracked_dets, tracker, frame_idx, state, overlay_mode)
    track_data = _build_track_data(tracker, tracked_dets, state, frame_idx)
    return annotated_frame, track_data


def process_frame_variants(frame, model, device, tracker, frame_idx, min_activity=0.25,
                           overlay_modes=(OVERLAY_MODE_DEBUG, OVERLAY_MODE_RELEASE)):
    """一次推理与跟踪，同时渲染多种叠加模式的帧。

    实时预览用 debug（不画残留框），结果视频用 release（残留框画实线），
    两者共用同一份跟踪结果，避免同一帧推理两次。

    返回:
        (frames, track_data): frames = {模式名: 标注帧}
    """
    tracked_dets = _detect_and_track(frame, model, device, tracker, frame_idx)
    state = _active_track_state(tracker, min_activity)
    frames = {
        mode: render_overlay(frame, tracked_dets, tracker, frame_idx, state, mode)
        for mode in overlay_modes
    }
    track_data = _build_track_data(tracker, tracked_dets, state, frame_idx)
    return frames, track_data


def _build_track_data(tracker, tracked_dets, state, frame_idx):
    """汇总本帧输出的跟踪数据（仅包含活跃/已确认目标）。"""
    ghost_ids, active_ids_all, confirmed_ids = state
    shared_ids = set(getattr(tracker, 'shared_ids', set()))
    shadowed_ids = set(getattr(tracker, 'shadowed_ids', set()))

    # 收集跟踪数据（仅包含活跃目标）
    track_data = []
    active_ids = []

    if hasattr(tracked_dets, 'tracker_id') and tracked_dets.tracker_id is not None:
        all_ids = {int(tid) for tid in tracked_dets.tracker_id
                   if tid is not None and int(tid) > 0 and int(tid) in confirmed_ids}
        active_ids = sorted(tid for tid in all_ids if tid in active_ids_all)

    for tid in active_ids:
        cycle_info = tracker.cycle_counts.get(tid, {})
        avg_freq = cycle_info.get('avg_freq', cycle_info.get('average_frequency', cycle_info.get('freq', 0.0)))
        live_freq = cycle_info.get('live_freq', cycle_info.get('realtime_freq', avg_freq))
        track_data.append({
            'track_id': tid,
            'posture_count': cycle_info.get('cycles', 0),
            'cycles': cycle_info.get('cycles', 0),
            'swing_count': cycle_info.get('swing_count', 0),
            'swings': cycle_info.get('swings', 0),
            'observed_seconds': cycle_info.get('observed_seconds', 0.0),
            'swing_freq_hz': cycle_info.get('swing_freq_hz', avg_freq),
            'live_swing_freq': cycle_info.get('live_swing_freq', live_freq),
            'bends_per_min': cycle_info.get('bends_per_min', 0.0),
            'swing_activity': cycle_info.get('swing_activity', 0.0),
            'freq': avg_freq,
            'avg_freq': avg_freq,
            'average_frequency': avg_freq,
            'live_freq': live_freq,
            'realtime_freq': live_freq,
            'swing_speed': avg_freq,
            'head_swing_frequency': avg_freq,
            'motion_type': cycle_info.get('motion_type', 'unknown'),
            'activity_score': cycle_info.get('activity_score', 0.0),
            # 视觉残留相关 + 完整周期信息
            'is_ghost': bool(cycle_info.get('is_ghost', tid in ghost_ids)),
            'is_shared': bool(tid in shared_ids),
            'is_shadowed': bool(tid in shadowed_ids),
            'lost_frames': int(cycle_info.get('lost_frames', 0)),
            'cycle_count': int(cycle_info.get('cycle_count', 0)),
            'cycle_freq': float(cycle_info.get('cycle_freq', 0.0)),
            'inst_freq': float(cycle_info.get('inst_freq', 0.0)),
            'half_swing_count': int(cycle_info.get('half_swing_count', 0)),
            'count_mode': cycle_info.get('count_mode', SWING_COUNT_MODE),
            'count_unit': cycle_info.get('count_unit', ''),
            'posture_shape': cycle_info.get('posture_shape', ''),
            'posture_amp_pct': float(cycle_info.get('posture_amp_pct', 0.0)),
            'is_stale': bool(cycle_info.get('is_stale', False)),
            'unobserved_seconds': float(cycle_info.get('unobserved_seconds', 0.0)),
            'elapsed_seconds': float(cycle_info.get('elapsed_seconds', 0.0)),
            'avg_freq_observed': float(cycle_info.get('avg_freq_observed', 0.0)),
        })

    # A4：已确认但本帧没有检测到的轨迹（停止运动被漏检）仍然列出，并标注未观测时长；
    # 数据过期时实时频率与活性按 0 显示，避免显示被冻结的旧高值。
    present_ids = set(active_ids)
    fps_val = max(float(getattr(tracker, 'fps', 30.0)), 1e-6)
    for tid in sorted(getattr(tracker, 'confirmed_ids', set())):
        if tid in present_ids:
            continue
        cycle_info = tracker.cycle_counts.get(tid)
        if not cycle_info:
            continue
        last_seen = tracker.id_last_seen.get(tid)
        unobserved = ((frame_idx - last_seen) / fps_val) if last_seen is not None else float(cycle_info.get('unobserved_seconds', 0.0))
        stale = unobserved > STALE_UNOBSERVED_SEC
        avg_freq = cycle_info.get('avg_freq', cycle_info.get('average_frequency', cycle_info.get('freq', 0.0)))
        track_data.append({
            'track_id': tid,
            'posture_count': cycle_info.get('cycles', 0),
            'cycles': cycle_info.get('cycles', 0),
            'swing_count': cycle_info.get('swing_count', 0),
            'swings': cycle_info.get('swings', 0),
            'observed_seconds': cycle_info.get('observed_seconds', 0.0),
            'elapsed_seconds': cycle_info.get('elapsed_seconds', 0.0),
            'unobserved_seconds': round(unobserved, 2),
            'is_stale': bool(stale),
            'avg_freq_observed': cycle_info.get('avg_freq_observed', 0.0),
            'swing_freq_hz': avg_freq,
            'live_swing_freq': 0.0,
            'bends_per_min': cycle_info.get('bends_per_min', 0.0),
            'swing_activity': cycle_info.get('swing_activity', 0.0),
            'freq': avg_freq,
            'avg_freq': avg_freq,
            'average_frequency': avg_freq,
            'live_freq': 0.0,
            'realtime_freq': 0.0,
            'swing_speed': avg_freq,
            'head_swing_frequency': avg_freq,
            'motion_type': cycle_info.get('motion_type', 'unknown'),
            'activity_score': 0.0,
            'is_ghost': True,
            'is_shared': False,
            'is_shadowed': bool(tid in shadowed_ids),
            'lost_frames': int(cycle_info.get('lost_frames', 0)),
            'cycle_count': int(cycle_info.get('cycle_count', 0)),
            'cycle_freq': float(cycle_info.get('cycle_freq', 0.0)),
            'inst_freq': 0.0,
            'half_swing_count': int(cycle_info.get('half_swing_count', 0)),
            'count_mode': cycle_info.get('count_mode', SWING_COUNT_MODE),
            'count_unit': cycle_info.get('count_unit', ''),
            'posture_shape': cycle_info.get('posture_shape', ''),
            'posture_amp_pct': float(cycle_info.get('posture_amp_pct', 0.0)),
        })

    return track_data