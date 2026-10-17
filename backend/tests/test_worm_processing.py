import sys
from pathlib import Path
import os
import tempfile
import unittest

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worm_processing import (
    EnhancedWormTracker,
    KalmanBoxTracker,
    PostureSwingCounter,
    SwingCounter,
    TrackHistory,
    compute_head_bend_angle,
    compute_posture_metrics,
    orient_pose_pair,
    render_overlay,
    _active_track_state,
    _contain_ratio,
    _draw_contour_overlay,
    _mask_iou_in_boxes,
    _mask_to_binary,
    OVERLAY_MODE_DEBUG,
    OVERLAY_MODE_RELEASE,
)


def make_detection(bbox, confidence=0.9):
    return sv.Detections(
        xyxy=np.asarray([bbox], dtype=np.float32),
        confidence=np.asarray([confidence], dtype=np.float32),
        class_id=np.asarray([0], dtype=int),
    )


def empty_detections():
    return sv.Detections.empty()


class HeadBendSignalTests(unittest.TestCase):
    def test_straight_worm_has_zero_bend(self):
        kp = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
        self.assertAlmostEqual(compute_head_bend_angle(kp), 0.0, places=5)

    def test_bend_is_relative_to_body_axis(self):
        # 头颈方向相对体轴向上偏 45 度
        kp = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
        self.assertAlmostEqual(abs(compute_head_bend_angle(kp)), 45.0, places=3)

    def test_returns_none_for_short_keypoints(self):
        self.assertIsNone(compute_head_bend_angle(np.zeros((3, 2), dtype=np.float32)))


class TrackHistoryTests(unittest.TestCase):
    def test_dominant_motion_type_works_after_many_frames(self):
        # 回归：motion_types 是 deque，旧实现用 deque[-N:] 切片会抛
        # TypeError: sequence index must be integer, not 'slice'，导致识别中断。
        history = TrackHistory(max_len_frames=500, fps=10)
        for _ in range(80):
            history.append(np.zeros((11, 2), dtype=np.float32), 0.0)
        self.assertEqual(history.get_dominant_motion_type(window_frames=30), 'static')


class KalmanTrackerTests(unittest.TestCase):
    def test_predict_after_update_returns_valid_bbox(self):
        tracker = KalmanBoxTracker([0.0, 0.0, 10.0, 10.0], feature=np.zeros(11, dtype=np.float32))
        tracker.update([1.0, 1.0, 11.0, 11.0], feature=np.zeros(11, dtype=np.float32))

        bbox = tracker.predict()

        self.assertEqual(len(bbox), 4)
        self.assertTrue(np.all(np.isfinite(bbox)))


class SwingCounterTests(unittest.TestCase):
    """摆动计数器回归测试（默认：一次单方向摆动 = 1 次）。"""

    @staticmethod
    def _sine(freq_hz, amp_deg, duration_sec, fps=30.0, phase=0.0):
        t = np.arange(int(fps * duration_sec)) / fps
        return amp_deg * np.sin(2 * np.pi * freq_hz * t + phase)

    @staticmethod
    def _feed(angles, fps=30.0, **kwargs):
        sc = SwingCounter(fps=fps, **kwargs)
        for a in angles:
            sc.feed(a)
        return sc

    def test_counts_each_directional_swing_by_default(self):
        sc = self._feed(self._sine(2.0, 30.0, 6.0))
        # 2Hz 完整周期 12 个 -> 单方向摆动约 24 次
        self.assertTrue(20 <= sc.swing_count <= 26, f'swing_count={sc.swing_count}')
        self.assertTrue(9 <= sc.cycle_count <= 13, f'cycle_count={sc.cycle_count}')
        self.assertAlmostEqual(sc.avg_freq(), 4.0, delta=0.6)

    def test_cycle_mode_counts_full_cycles(self):
        sc = self._feed(self._sine(2.0, 30.0, 6.0), count_mode='cycle')
        self.assertTrue(9 <= sc.swing_count <= 13, f'swing_count={sc.swing_count}')
        self.assertAlmostEqual(sc.avg_freq(), 2.0, delta=0.5)

    def test_ignores_low_amplitude_noise(self):
        rng = np.random.default_rng(7)
        angles = rng.uniform(-5.0, 5.0, size=180)
        sc = self._feed(angles, fps=30.0)
        # 均匀白噪声的极值偶尔接近真实小幅摆动，允许 6 秒内最多 1 次误计
        self.assertLessEqual(sc.swing_count, 1, f'swing_count={sc.swing_count}')

    def test_ignores_bends_below_amplitude_threshold(self):
        # 极小幅度（2 度）低于绝对下限，且被噪声保护挡住，不计入
        sc = self._feed(self._sine(2.0, 2.0, 6.0), fps=30.0)
        self.assertEqual(sc.swing_count, 0)

    def test_counts_small_amplitude_swings(self):
        # A1：阈值下限自适应后，小幅度（6 度）摆动也应被计入
        sc = self._feed(self._sine(2.0, 6.0, 6.0), fps=30.0)
        self.assertGreaterEqual(sc.swing_count, 8, f'swing_count={sc.swing_count}')

    def test_static_angles_produce_zero(self):
        sc = self._feed([45.0] * 120, fps=30.0)
        self.assertEqual(sc.swing_count, 0)

    def test_orientation_flip_does_not_break_count(self):
        fps = 30.0
        n = int(fps * 6.0)
        sig = self._sine(2.0, 30.0, 6.0, fps)
        mid = n // 2
        flipped = np.concatenate([sig[:mid], -sig[mid:]])

        sc_ref = self._feed(sig, fps)
        sc_flip = self._feed(flipped, fps)
        self.assertLessEqual(abs(sc_flip.swing_count - sc_ref.swing_count), 4)

    def test_live_frequency_is_not_inflated_at_start(self):
        sc = SwingCounter(fps=30.0)
        for a in [0.0, 30.0, 0.0]:
            sc.feed(a)
        self.assertEqual(sc.live_freq(), 0.0)

    def test_slow_drift_does_not_create_many_swings(self):
        # 单向缓慢漂移（机身整体转向）不应被当成连续摆动
        angles = np.linspace(0.0, 120.0, 150)
        sc = self._feed(angles, fps=30.0)
        self.assertLessEqual(sc.swing_count, 2)


class PersistenceTests(unittest.TestCase):
    """视觉残留 / 序号保留回归测试。"""

    def test_keeps_id_and_ghost_during_interruption(self):
        tracker = EnhancedWormTracker(fps=10, frame_size=(640, 480))
        for i in range(1, 6):
            tracker.update_tracking(make_detection([100, 100, 140, 140]), None, i)

        self.assertEqual(len(tracker.tracks), 1)
        tid = tracker.tracks[0].track_id

        # 丢失 5 帧（仍在残留窗口内）：残留框与序号继续输出
        for i in range(6, 11):
            tracked = tracker.update_tracking(empty_detections(), None, i)
        self.assertIn(tid, tracker.ghost_ids)
        self.assertIn(tid, [int(t) for t in tracked.tracker_id])

        # 丢失超出残留绘制窗口：不再绘制，但 ID 仍然保留，不会被回收
        for i in range(11, 80):
            tracked = tracker.update_tracking(empty_detections(), None, i)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertNotIn(tid, tracker.ghost_ids)
        self.assertEqual(tracker.tracks[0].track_id, tid)

        # 目标重新出现：沿用原序号（而不是新建 ID）
        tracked = tracker.update_tracking(make_detection([150, 110, 190, 150]), None, 81)
        self.assertEqual([int(t) for t in tracked.tracker_id], [tid])
        self.assertNotIn(tid, tracker.ghost_ids)

    def test_pruned_id_is_not_reused(self):
        tracker = EnhancedWormTracker(fps=10, frame_size=(640, 480))
        for i in range(1, 4):
            tracker.update_tracking(make_detection([100, 100, 140, 140]), None, i)
        first_id = tracker.tracks[0].track_id

        for i in range(4, tracker.max_lost_frames + 10):
            tracker.update_tracking(empty_detections(), None, i)
        self.assertEqual(len(tracker.tracks), 0)

        tracked = tracker.update_tracking(make_detection([100, 100, 140, 140]), None, 99999)
        self.assertNotEqual(int(tracked.tracker_id[0]), first_id)


class GhostOverlayModeTests(unittest.TestCase):
    """残留框渲染模式：实时预览(debug)不落笔，结果视频(release)画实线框。"""

    BBOX = [300, 300, 360, 360]

    def _ghost_state(self):
        """造一个处于视觉残留（ghost）状态的已确认目标。"""
        tracker = EnhancedWormTracker(fps=10, frame_size=(640, 480))
        for i in range(1, 6):
            tracker.update_tracking(make_detection(self.BBOX), None, i)
        tracked = None
        for i in range(6, 11):
            tracked = tracker.update_tracking(empty_detections(), None, i)

        tid = tracker.tracks[0].track_id
        self.assertIn(tid, tracker.confirmed_ids)
        self.assertIn(tid, tracker.ghost_ids)
        self.assertIn(tid, [int(t) for t in tracked.tracker_id])
        return tracker, tracked

    @staticmethod
    def _frame():
        # 灰色背景：避免与白色标注（_MARK_COLOR）同色，方便用像素差断言
        return np.full((480, 480, 3), 90, dtype=np.uint8)

    def test_debug_preview_has_no_ghost_artifacts(self):
        tracker, tracked = self._ghost_state()
        frame = self._frame()

        out = _draw_contour_overlay(frame.copy(), tracked, tracker, overlay_mode=OVERLAY_MODE_DEBUG)

        self.assertTrue(np.array_equal(out, frame), '实时预览不应画出任何残留框或标签')

    def test_release_video_draws_solid_ghost_box(self):
        tracker, tracked = self._ghost_state()
        frame = self._frame()

        out = _draw_contour_overlay(frame.copy(), tracked, tracker, overlay_mode=OVERLAY_MODE_RELEASE)

        x1, y1, x2, y2 = [int(v) for v in tracked.xyxy[0]]
        bg = frame[y1, x1:x2 + 1]
        row = out[y1, x1:x2 + 1]
        self.assertFalse(np.array_equal(row, bg), '结果视频里应画出残留框')
        # 实线：上边整条都被画上；虚线会留出成片的背景色间断
        self.assertTrue(np.all(row != bg), '结果视频里的残留框应为实线，上边不能有间断')

    def test_render_overlay_switches_ghost_by_mode(self):
        tracker, tracked = self._ghost_state()
        state = _active_track_state(tracker, 0.25)
        frame = self._frame()
        x1, y1, x2, y2 = [int(v) for v in tracked.xyxy[0]]

        debug_out = render_overlay(frame, tracked, tracker, 11, state, overlay_mode=OVERLAY_MODE_DEBUG)
        release_out = render_overlay(frame, tracked, tracker, 11, state, overlay_mode=OVERLAY_MODE_RELEASE)

        self.assertTrue(
            np.array_equal(debug_out[y1, x1:x2 + 1], frame[y1, x1:x2 + 1]),
            '实时预览帧在残留框位置不应有任何绘制',
        )
        self.assertTrue(
            np.all(release_out[y1, x1:x2 + 1] != frame[y1, x1:x2 + 1]),
            '结果视频帧在残留框位置应为实线',
        )


class DedupAndTouchingTests(unittest.TestCase):
    """重复识别抑制：检测去重、出生门控、贴合共享、残留框被压住不绘制。"""

    H = W = 480

    @staticmethod
    def _dets(boxes, scores=None, masks=None):
        boxes = np.asarray(boxes, dtype=np.float32)
        n = len(boxes)
        kwargs = {
            'xyxy': boxes,
            'confidence': np.asarray(scores if scores is not None else [0.9] * n, dtype=np.float32),
            'class_id': np.zeros(n, dtype=int),
        }
        if masks is not None:
            kwargs['mask'] = np.asarray(masks)
        return sv.Detections(**kwargs)

    @staticmethod
    def _blank_mask():
        return np.zeros((DedupAndTouchingTests.H, DedupAndTouchingTests.W), dtype=bool)

    def test_crossing_worms_are_not_deduped_by_box_overlap(self):
        """两条虫交叉：框几乎完全重叠，但掩码各走一条对角线 —— 不能当成重复删掉一条。"""
        tracker = EnhancedWormTracker(fps=10, frame_size=(self.W, self.H))
        m1, m2 = self._blank_mask(), self._blank_mask()
        cv2.line(m1, (100, 100), (200, 200), True, 9)
        cv2.line(m2, (100, 200), (200, 100), True, 9)
        box1 = [100, 100, 200, 200]
        box2 = [100, 100, 200, 200]
        det = self._dets([box1, box2], masks=[m1, m2])

        box_iou = tracker.compute_iou(np.asarray(box1, dtype=np.float32), np.asarray(box2, dtype=np.float32))
        mask_iou, _ = _mask_iou_in_boxes(m1, m2, box1, box2)
        self.assertGreater(box_iou, tracker.overlap_threshold, '前提：框重叠超过旧的 NMS 阈值')
        self.assertLess(mask_iou, tracker.dedup_mask_iou, '前提：掩码并不重叠')

        out = tracker.apply_nms(det, det.mask)

        self.assertEqual(len(out), 2, '交叉的两条虫都必须保留（旧实现会按框 IoU 删掉一条）')

    def test_fragment_duplicate_is_deduped_by_mask(self):
        """同一只虫的"整只 + 局部片段"：框 IoU 低于阈值，但掩码重叠明显 —— 必须去重。"""
        tracker = EnhancedWormTracker(fps=10, frame_size=(self.W, self.H))
        m_full, m_frag = self._blank_mask(), self._blank_mask()
        cv2.line(m_full, (100, 100), (220, 220), True, 9)   # 整只
        cv2.line(m_frag, (100, 100), (160, 160), True, 9)   # 只覆盖前半段
        full_box = [96, 96, 225, 225]
        frag_box = [96, 96, 165, 165]
        det = self._dets([full_box, frag_box], scores=[0.9, 0.8], masks=[m_full, m_frag])

        box_iou = tracker.compute_iou(np.asarray(full_box, dtype=np.float32), np.asarray(frag_box, dtype=np.float32))
        mask_iou, _ = _mask_iou_in_boxes(m_full, m_frag, full_box, frag_box)
        self.assertLess(box_iou, tracker.overlap_threshold, '前提：只看框是漏掉这种重复的')
        self.assertGreater(mask_iou, tracker.dedup_mask_iou, '前提：掩码重叠明显')

        out = tracker.apply_nms(det, det.mask)

        self.assertEqual(len(out), 1, '同一只虫的重复检测必须去重')
        self.assertAlmostEqual(float(out.confidence[0]), 0.9, places=5, msg='保留置信度高的那个')

    def test_masks_stay_aligned_with_boxes_after_dedup(self):
        """去重后掩码必须跟框一一对应（旧实现继续用旧下标取掩码，会配错虫的掩码）。"""
        tracker = EnhancedWormTracker(fps=10, frame_size=(self.W, self.H))
        mask_a, mask_b = self._blank_mask(), self._blank_mask()
        mask_a[100:120, 100:150] = True     # 低分、将被抑制
        mask_b[100:115, 120:160] = True     # 高分、保留
        box_a = [100, 100, 150, 120]
        box_b = [120, 100, 160, 115]
        det = self._dets([box_a, box_b], scores=[0.80, 0.95], masks=[mask_a, mask_b])
        self.assertEqual(len(tracker.apply_nms(det, det.mask)), 1, '前提：两者被判为同一只虫')

        tracker.update_tracking(det, det.mask, 1)

        self.assertEqual(len(tracker.tracks), 1)
        aligned = tracker.extract_features(mask_b, np.asarray(box_b, dtype=np.float32))
        misaligned = tracker.extract_features(mask_a, np.asarray(box_b, dtype=np.float32))
        self.assertFalse(np.allclose(aligned, misaligned), '前提：错位的掩码算出的特征确实不同')
        self.assertTrue(np.allclose(tracker.tracks[0].feature, aligned),
                        '去重后必须用幸存检测自己的掩码算特征')

    def test_birth_blocked_for_fragment_of_live_track(self):
        """同一帧里"整只 + 包在里面更小的片段"：片段不能再新建一条 ID。"""
        tracker = EnhancedWormTracker(fps=10, frame_size=(self.W, self.H))
        whole = [100, 100, 180, 120]
        for i in range(1, 6):
            tracker.update_tracking(self._dets([whole]), None, i)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertIn(tracker.tracks[0].track_id, tracker.confirmed_ids)
        next_id_before = tracker.next_id

        fragment = [110, 105, 130, 115]      # 被整只的框完全包住，面积小得多
        tracker.update_tracking(self._dets([whole, fragment], scores=[0.9, 0.85]), None, 6)

        self.assertEqual(len(tracker.tracks), 1, '片段不应新建轨迹')
        self.assertEqual(tracker.next_id, next_id_before, '不应消耗新的 ID')

    def test_birth_cooldown_blocks_repeat_birth_at_same_place(self):
        """冷却只按掩码判：同一只虫反复出生要拦，两条贴在一起的虫（掩码不重叠）不能拦。"""
        tracker = EnhancedWormTracker(fps=10, frame_size=(self.W, self.H))
        prev_mask = self._blank_mask()
        prev_mask[300:310, 300:340] = True
        prev_box = np.asarray([300, 300, 340, 310], dtype=np.float32)
        crop, origin = tracker._crop_mask_to_box(prev_mask, prev_box)
        tracker._recent_births.append((5, prev_box, crop, origin, int(prev_mask.sum())))

        same = self._blank_mask()
        same[301:311, 305:345] = True                       # 同一只虫（掩码几乎重合）
        other = self._blank_mask()
        other[300:310, 360:400] = True                      # 另一条虫（掩码不重叠）

        self.assertTrue(tracker._should_block_birth(
            0, self._dets([[305, 301, 345, 311]], masks=[same]), np.asarray([same]), [],
            5 + tracker.birth_cooldown_frames), '冷却窗口内同一只虫不应重复出生')
        self.assertFalse(tracker._should_block_birth(
            0, self._dets([[360, 300, 400, 310]], masks=[other]), np.asarray([other]), [],
            5 + tracker.birth_cooldown_frames), '掩码不重叠的另一条虫必须允许出生')
        self.assertFalse(tracker._should_block_birth(
            0, self._dets([[305, 301, 345, 311]], masks=[same]), np.asarray([same]), [],
            5 + tracker.birth_cooldown_frames + 1), '冷却结束后允许出生')

    def test_sub_video_only_created_after_stable_observation(self):
        """未确认、或观测帧数太少的轨迹都不建子视频；命中数达到 sub_video_min_hits 才建。

        目的：避免瞬时噪声 ID 在下载的 ZIP/列表里变成"多出来的虫子"。
        """
        with tempfile.TemporaryDirectory() as tmp:
            tracker = EnhancedWormTracker(fps=10, frame_size=(self.W, self.H), sub_video_dir=tmp)
            box = [100, 100, 140, 110]
            tracker.update_tracking(self._dets([box]), None, 1)
            self.assertFalse(os.listdir(tmp), '第 1 帧不应建子视频')

            for i in range(2, 6):
                tracker.update_tracking(self._dets([box]), None, i)
            self.assertTrue(tracker.tracks[0].confirmed, '命中 5 帧后应确认')
            self.assertLess(tracker.tracks[0].hits, tracker.sub_video_min_hits)
            self.assertFalse(os.listdir(tmp), '确认了但观测太短，仍不应建子视频')

            for i in range(6, tracker.sub_video_min_hits + 2):
                tracker.update_tracking(self._dets([box]), None, i)
            self.assertGreaterEqual(tracker.tracks[0].hits, tracker.sub_video_min_hits)
            self.assertTrue(os.listdir(tmp), '达到 sub_video_min_hits 后应建子视频')
            if tracker.tracks[0].sub_video_writer is not None:
                tracker.tracks[0].sub_video_writer.release()

    def test_touching_worms_share_one_detection_and_stay_alive(self):
        """两条虫贴住、模型只给一条检测：两条轨迹都要活着（不能合成一条），且共享帧暂停计数。"""
        tracker = EnhancedWormTracker(fps=10, frame_size=(self.W, self.H))
        box_a, box_b = [100, 100, 140, 120], [120, 100, 160, 120]
        mask_a, mask_b = self._blank_mask(), self._blank_mask()
        mask_a[100:110, 100:140] = True
        mask_b[110:120, 120:160] = True
        for i in range(1, 6):
            d = self._dets([box_a, box_b], masks=[mask_a, mask_b])
            tracker.update_tracking(d, d.mask, i)
        self.assertEqual(len(tracker.tracks), 2, '前提：两条虫各自有轨迹')
        both_ids = sorted(int(t.track_id) for t in tracker.tracks)

        fed = []
        tracker.feed_swing_sample = lambda tid, kp, fn=None: fed.append(tid)   # 计数调用打点

        # 第 6 帧：贴在一起，模型只给了一条检测（另一条的框与它重叠 0.33）
        d6 = self._dets([box_b], masks=[mask_b])
        tracked = tracker.update_tracking(d6, d6.mask, 6)

        self.assertTrue(tracker.shared_ids, '应判为贴合共享')
        self.assertFalse(tracker.ghost_ids, '贴合共享不应进入视觉残留')
        self.assertEqual(len(tracker.shared_live_ids), 1)
        out_ids = {int(t) for t in tracked.tracker_id}
        self.assertTrue(tracker.shared_ids <= out_ids, '共享的那条也要出现在输出里')
        self.assertTrue(tracker.shared_live_ids <= out_ids, '实时那条也要出现在输出里')
        self.assertEqual(fed, [], '共享帧两条虫都暂停摆动计数')
        self.assertEqual(sorted(both_ids), both_ids)

    def test_shadowed_ghost_box_is_not_drawn(self):
        """残留框压在实时框上（同一只虫的重复 ID / 被挡住的虫）时不再绘制第二个框。"""
        tracker = EnhancedWormTracker(fps=10, frame_size=(self.W, self.H))
        box_a, box_b = [100, 100, 140, 120], [120, 100, 160, 120]
        mask_a, mask_b = self._blank_mask(), self._blank_mask()
        mask_a[100:110, 100:140] = True
        mask_b[110:120, 120:160] = True
        for i in range(1, 6):
            d = self._dets([box_a, box_b], masks=[mask_a, mask_b])
            tracker.update_tracking(d, d.mask, i)

        tracker.shared_iou = 0.0     # 关掉共享，单独验证"被压住的残留框不画"
        d6 = self._dets([box_b], masks=[mask_b])
        tracked = tracker.update_tracking(d6, d6.mask, 6)

        self.assertEqual(len(tracker.ghost_ids), 1, '前提：另一条进入残留')
        self.assertTrue(tracker.shadowed_ids, '残留框与实时框重叠 -> 应被标记为不绘制')

        frame = np.full((self.H, self.W, 3), 90, dtype=np.uint8)
        drawn = _draw_contour_overlay(frame.copy(), tracked, tracker, overlay_mode=OVERLAY_MODE_RELEASE)
        saved = set(tracker.shadowed_ids)
        tracker.shadowed_ids = set()
        drawn_all = _draw_contour_overlay(frame.copy(), tracked, tracker, overlay_mode=OVERLAY_MODE_RELEASE)
        tracker.shadowed_ids = saved

        self.assertFalse(np.array_equal(drawn, drawn_all),
                         '被压住的残留框不应出现在画面上（这里验证它确实少画了一个框）')


class PostureSwingCounterTests(unittest.TestCase):
    """体态计数：一次 C 形或 S 形计 1 次；微小摆动不计；同一姿态不重复计数。"""

    @staticmethod
    def _kp(a, b, c, L=100.0):
        """构造 11 点中心线：体长 20%/50%/80% 处横向偏移为 a/b/c（单位：体长比例）。"""
        xs = np.linspace(0, 1, 11)
        ys = np.interp(xs, [0.0, 0.2, 0.5, 0.8, 1.0], [0.0, a, b, c, 0.0])
        return np.stack([xs * L, ys * L], axis=1)

    @staticmethod
    def _feed(counter, keypoints, n):
        """喂 n 帧：每帧是一份"新观测"（带亚像素抖动，判定的姿态不变）。

        真实流水线里关键点每 2 帧才重新提取一次，期间喂的是同一份观测，
        重复观测不累计持续时间 —— 这里必须模拟"新观测"，否则 hold 条件永远不满足。
        """
        for i in range(n):
            k = np.asarray(keypoints, dtype=np.float64).copy()
            k[:, 1] += 0.2 if i % 2 else -0.2
            counter.feed_keypoints(k)

    def test_posture_metrics_classification(self):
        c_shape = compute_posture_metrics(self._kp(0.15, 0.22, 0.15))
        s_shape = compute_posture_metrics(self._kp(0.15, 0.0, -0.15))
        flipped = compute_posture_metrics(self._kp(-0.15, -0.22, -0.15))

        self.assertEqual(c_shape['shape'], 'C')
        self.assertEqual(s_shape['shape'], 'S')
        self.assertEqual(c_shape['direction'], 1)
        self.assertEqual(flipped['direction'], -1)
        self.assertAlmostEqual(c_shape['a_pct'], 22.0, delta=1.0)
        self.assertIsNone(compute_posture_metrics(None))
        self.assertIsNone(compute_posture_metrics(np.zeros((4, 2))))

    def test_sustained_c_shape_counts_once(self):
        """身体保持同一个 C 形（头颈抖动）只算 1 次 —— 旧口径这里会连续计很多次。"""
        counter = PostureSwingCounter(fps=30)
        self._feed(counter, self._kp(0.15, 0.22, 0.15), 12)

        self.assertEqual(counter.swing_count, 1)
        self.assertEqual(counter.last_shape, 'C')

    def test_jitter_within_one_posture_counts_once(self):
        counter = PostureSwingCounter(fps=30)
        for i in range(20):
            j = 0.02 if i % 2 == 0 else -0.02
            self._feed(counter, self._kp(0.15 + j, 0.22 + j, 0.15 + j), 1)

        self.assertEqual(counter.swing_count, 1)

    def test_micro_swing_is_ignored(self):
        """幅度只有体长 5%（低于 10% 门槛）的来回摆动不计入。"""
        counter = PostureSwingCounter(fps=30)
        for _ in range(10):
            self._feed(counter, self._kp(0.03, 0.05, 0.03), 3)
            self._feed(counter, self._kp(-0.03, -0.05, -0.03), 3)

        self.assertEqual(counter.swing_count, 0)

    def test_opposite_c_postures_count_twice(self):
        counter = PostureSwingCounter(fps=30)
        self._feed(counter, self._kp(0.15, 0.22, 0.15), 6)     # C 向右
        self._feed(counter, self._kp(0.0, 0.0, 0.0), 2)        # 回落（释放）
        self._feed(counter, self._kp(-0.15, -0.22, -0.15), 6)  # C 向左

        self.assertEqual(counter.swing_count, 2)
        self.assertEqual(counter.cycle_count, 1, '一个完整往返 = 1 个周期')

    def test_s_shape_counts_as_its_own_posture(self):
        """默认口径：C 和 S 都算独立姿态，各自计 1 次。"""
        counter = PostureSwingCounter(fps=30)
        self._feed(counter, self._kp(0.15, 0.20, 0.15), 6)     # C（头侧向右）
        self._feed(counter, self._kp(0.15, 0.0, -0.15), 6)     # S（头侧仍向右）

        self.assertEqual(counter.swing_count, 2)

    def test_pair_unit_only_counts_direction_flips(self):
        counter = PostureSwingCounter(fps=30, unit='pair')
        self._feed(counter, self._kp(0.15, 0.20, 0.15), 6)     # C 向右
        self._feed(counter, self._kp(0.15, 0.0, -0.15), 6)     # S，方向仍向右
        self.assertEqual(counter.swing_count, 1, '同一方向不应算新的一次')

        self._feed(counter, self._kp(-0.15, -0.20, -0.15), 6)  # C 向左
        self.assertEqual(counter.swing_count, 2)

    def test_repeat_same_posture_after_release_counts_again(self):
        counter = PostureSwingCounter(fps=30)
        self._feed(counter, self._kp(0.15, 0.22, 0.15), 6)
        self._feed(counter, self._kp(0.0, 0.0, 0.0), 3)        # 姿态结束（幅度落到释放线以下）
        self._feed(counter, self._kp(0.15, 0.22, 0.15), 6)     # 又形成同一个 C 形

        self.assertEqual(counter.swing_count, 2)

    def test_single_frame_spike_is_not_counted(self):
        counter = PostureSwingCounter(fps=30, hold_frames=3)
        self._feed(counter, self._kp(0.0, 0.0, 0.0), 2)
        self._feed(counter, self._kp(0.20, 0.30, 0.20), 1)     # 只有 1 帧
        self._feed(counter, self._kp(0.0, 0.0, 0.0), 3)

        self.assertEqual(counter.swing_count, 0)

    def test_frequency_uses_posture_events(self):
        counter = PostureSwingCounter(fps=30)
        self._feed(counter, self._kp(0.15, 0.22, 0.15), 15)
        self._feed(counter, self._kp(-0.15, -0.22, -0.15), 15)

        self.assertEqual(counter.swing_count, 2)
        self.assertGreater(counter.observed_seconds(), 0.9)
        self.assertAlmostEqual(counter.avg_freq(), 2 / counter.observed_seconds(), places=5)
        self.assertGreater(counter.live_freq(), 0.0)


class OrientPosePairTests(unittest.TestCase):
    def test_reverses_when_tail_is_closer_to_previous_head(self):
        keypoints = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        centerline = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        ref_head = keypoints[-1].copy()

        _, out_kp = orient_pose_pair(centerline, keypoints, ref_head)
        self.assertTrue(np.allclose(out_kp[0], ref_head))
        self.assertTrue(np.allclose(keypoints[0], [0.0, 0.0]))

    def test_keeps_order_when_head_is_closer(self):
        keypoints = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        ref_head = keypoints[0].copy()
        _, out_kp = orient_pose_pair(None, keypoints, ref_head)
        self.assertTrue(np.allclose(out_kp[0], ref_head))


if __name__ == "__main__":
    unittest.main()
