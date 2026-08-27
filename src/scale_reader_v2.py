"""
自主刻度定位 V2 — 关键点优先 + CV检测回退 + 角度比例法读数
参考 Nanodet-YOLOv8-Pose 仓库方案
"""
import math
import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ScaleReaderV2:
    """自主刻度定位：关键点→角度比例→读数值，无需OCR"""

    def __init__(self, presets: dict = None, default_range=(0.0, 1.6)):
        """
        presets: 预设量程配置 {'id': {'anchors':[...], 'unit':'MPa'}}
        default_range: 默认量程 (min, max)
        """
        self.presets = presets or {}
        self.default_min, self.default_max = default_range
        self._preset_ranges = {}
        self._build_preset_cache()

    def load_presets(self, presets: dict):
        self.presets = presets
        self._build_preset_cache()

    def _build_preset_cache(self):
        for pid, cfg in self.presets.items():
            anchors = cfg.get("anchors", [])
            if len(anchors) >= 2:
                angles = sorted([a for a, v in anchors])
                self._preset_ranges[pid] = {
                    'min_angle': angles[0],
                    'max_angle': angles[-1],
                    'span': self._angle_span(angles[0], angles[-1]),
                    'min_val': min(v for a, v in anchors),
                    'max_val': max(v for a, v in anchors),
                    'unit': cfg.get('unit', ''),
                }

    # ============================================================
    # 核心：角度比例法 (参考仓库)
    # ============================================================

    def compute_reading(self,
                         center: Tuple[float, float],
                         pointer_tip: Tuple[float, float],
                         scale_start: Tuple[float, float] = None,
                         scale_end: Tuple[float, float] = None,
                         scale_range: Tuple[float, float] = None,
                         unit: str = ""
                         ) -> Optional[Dict]:
        """
        角度比例法计算读数
        reading = min_val + ratio * (max_val - min_val)
        ratio = (pointer_angle_from_start) / scale_span

        参数:
            center: 表盘中心
            pointer_tip: 指针尖端
            scale_start: 刻度起点(可选，无则用estimate)
            scale_end: 刻度终点(可选)
            scale_range: (min_val, max_val) 量程范围(可选，无则用default)
        """
        cx, cy = center

        # 各点角度 (从中心出发，12点顺时针)
        ptr_angle = self._point_to_angle(pointer_tip[0], pointer_tip[1], cx, cy)

        if scale_start and scale_end:
            start_angle = self._point_to_angle(scale_start[0], scale_start[1], cx, cy)
            end_angle = self._point_to_angle(scale_end[0], scale_end[1], cx, cy)
            span = self._angle_span(start_angle, end_angle)
            method = "kp_full"
        elif scale_start or scale_end:
            # 单侧刻度 → 自动检测扫表角度
            detected_pt = scale_start or scale_end
            detected_angle = self._point_to_angle(detected_pt[0], detected_pt[1], cx, cy)
            is_start = scale_start is not None
            start_angle, end_angle, span = self._auto_detect_sweep(
                detected_angle, ptr_angle, is_start, scale_range)
            method = "kp_auto_sweep"
        else:
            return None

        # 指针在扫表范围内的角度
        ptr_from_start = self._angle_span(start_angle, ptr_angle)
        ratio = ptr_from_start / max(span, 1.0)

        # 量程
        if scale_range:
            min_val, max_val = scale_range
        else:
            min_val, max_val = self.default_min, self.default_max

        # 钳制ratio到[0,1]（允许少量外推±10%）
        ratio_clamped = max(-0.1, min(1.1, ratio))
        value = min_val + ratio_clamped * (max_val - min_val)

        # 参考仓库校准补偿
        value = self._calibrate(value)

        # 如果ratio超出范围，标记为外推
        is_extrapolated = ratio < 0 or ratio > 1

        return {
            'value': value,
            'ratio': ratio,
            'ratio_clamped': ratio_clamped,
            'start_angle': start_angle,
            'end_angle': end_angle,
            'ptr_angle': ptr_angle,
            'span': span,
            'method': method,
            'extrapolated': is_extrapolated,
            'unit': unit,
        }

    # ============================================================
    # 从关键点结果完整推理
    # ============================================================

    def from_keypoints(self,
                        center: Tuple[float, float],
                        pivot: Tuple[float, float],
                        tip: Tuple[float, float],
                        scale_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
                        scale_info: dict = None,
                        known_range: Tuple[float, float] = None,
                        unit: str = ""
                        ) -> Optional[Dict]:
        """
        从关键点结果一站式计算读数
        """
        # 先尝试完整刻度
        if scale_points:
            result = self.compute_reading(
                center, tip, scale_points[0], scale_points[1],
                scale_range=known_range, unit=unit)
            if result:
                return result

        # 单侧刻度
        if scale_info:
            partial = scale_info.get('partial', '')
            if 'right_only' in partial and 'right_point' in scale_info:
                return self.compute_reading(
                    center, tip,
                    scale_end=scale_info['right_point'],
                    scale_range=known_range, unit=unit)
            elif 'left_only' in partial and 'left_point' in scale_info:
                return self.compute_reading(
                    center, tip,
                    scale_start=scale_info['left_point'],
                    scale_range=known_range, unit=unit)

        # 无刻度信息 → 用预设匹配
        if self.presets:
            ptr_angle = self._point_to_angle(tip[0], tip[1], center[0], center[1])
            best = self._match_by_pointer(ptr_angle)
            if best:
                pid, cfg = best
                anchors = cfg['anchors']
                # 找最接近的预设锚点
                start_angle = min(a for a, v in anchors)
                end_angle = max(a for a, v in anchors)
                return self.compute_reading(
                    center, tip,
                    scale_start=(center[0] + math.sin(math.radians(start_angle)) * 100,
                                 center[1] - math.cos(math.radians(start_angle)) * 100),
                    scale_end=(center[0] + math.sin(math.radians(end_angle)) * 100,
                               center[1] - math.cos(math.radians(end_angle)) * 100),
                    scale_range=(cfg['min_val'], cfg['max_val']),
                    unit=cfg.get('unit', ''))

        return None

    # ============================================================
    # CV回退：刻度标记检测
    # ============================================================

    def detect_scale_markers_cv(self,
                                 dial_roi: np.ndarray,
                                 center: Tuple[float, float],
                                 radius: float
                                 ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        CV检测刻度起止标记 (当关键点模型失败时的回退)
        策略：在表盘环带区域找最左和最右的非零像素(刻度线密集区)
        """
        h, w = dial_roi.shape[:2]
        gray = cv2.cvtColor(dial_roi, cv2.COLOR_BGR2GRAY)

        # 自适应阈值 → 提取刻度线
        binary = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 5)

        # 环带mask（40%-85%半径）
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(mask, (cx, cy), int(radius * 0.85), 255, -1)
        cv2.circle(mask, (cx, cy), int(radius * 0.40), 0, -1)

        ring = cv2.bitwise_and(binary, binary, mask=mask)

        # 找到所有刻度线像素
        ys, xs = np.where(ring > 0)
        if len(xs) < 50:
            return None

        # 计算每个像素的角度
        angles = []
        for x, y in zip(xs, ys):
            a = self._point_to_angle(x, y, center[0], center[1])
            angles.append(a)

        angles = np.array(angles)

        # 在0-360°上使用直方图找到刻度线的两个端点
        hist, bins = np.histogram(angles, bins=72, range=(0, 360))

        # 找连续的高密度区域（刻度线）
        # 简单策略：取5%分位数和95%分位数作为起止点
        start_angle = np.percentile(angles, 5)
        end_angle = np.percentile(angles, 95)

        # 转换为点坐标
        r = radius * 0.65
        start_pt = (cx + r * math.sin(math.radians(start_angle)),
                    cy - r * math.cos(math.radians(start_angle)))
        end_pt = (cx + r * math.sin(math.radians(end_angle)),
                  cy - r * math.cos(math.radians(end_angle)))

        return (start_pt, end_pt)

    # ============================================================
    # 辅助
    # ============================================================

    @staticmethod
    def _point_to_angle(px, py, cx, cy):
        """像素→12点顺时针角度"""
        dx = px - cx
        dy = py - cy
        return math.degrees(math.atan2(dx, -dy)) % 360

    @staticmethod
    def _angle_span(a1, a2):
        """顺时针从a1到a2的角度跨度"""
        if a2 >= a1:
            return a2 - a1
        return 360 - a1 + a2

    def _auto_detect_sweep(self, detected_angle: float, ptr_angle: float,
                            is_start: bool, known_range=None
                            ) -> Tuple[float, float, float]:
        """
        自动检测扫表角度（替代硬编码270°）
        策略：连续扫描常见扫表角(90°-360°, 步进15°)，综合多个合理性指标打分
        指标：
          1) 指针必须在扫表范围内
          2) 扫表角接近常见值(180/225/270/300/360)
          3) 指针比例不极端(不在0或1边界, 除非确实如此)
        """
        # 连续候选: 90-360, 步进15
        sweep_candidates = list(range(90, 361, 15))
        # 常见扫表角优先
        common = [270, 225, 300, 180, 360, 240, 120]
        best_span = 270
        best_score = float('inf')

        for span in sweep_candidates:
            if is_start:
                end_angle = (detected_angle + span) % 360
                start_angle = detected_angle
            else:
                start_angle = (detected_angle - span) % 360
                end_angle = detected_angle

            # 指标1: 指针是否在范围内
            ptr_in_range = self._is_in_range(ptr_angle, start_angle, end_angle)
            if not ptr_in_range:
                continue  # 指针不在范围内直接淘汰

            # 指针相对起点的比例
            ptr_from_start = self._angle_span(start_angle, ptr_angle)
            ratio = ptr_from_start / max(span, 1.0)

            # 指标2: 接近常见扫表角 (差值越小越好, 每1°算0.3分)
            common_diff = min(abs(span - c) for c in common)
            score = common_diff * 0.3

            # 指标3: 指针比例极端惩罚 (越接近0或1, 越可能是错误扫表角)
            # 但允许指针确实在边界的情况(如读数≈0或满量程)
            if ratio < 0.05 or ratio > 0.95:
                score += 20  # 轻微惩罚, 不直接淘汰

            # 指标4: 匹配预设量程 (若存在)
            for pid, cfg in self._preset_ranges.items():
                span_diff = abs(cfg['span'] - span)
                score += span_diff * 0.1

            if score < best_score:
                best_score = score
                best_span = span

        if is_start:
            return detected_angle, (detected_angle + best_span) % 360, best_span
        else:
            return (detected_angle - best_span) % 360, detected_angle, best_span

    @staticmethod
    def _is_in_range(angle, start, end):
        """判断角度是否在顺时针范围内"""
        if end >= start:
            return start <= angle <= end
        return angle >= start or angle <= end

    def _calibrate(self, value: float) -> float:
        """
        参考仓库校准补偿
        来源: Nanodet-YOLOv8-Pose result_visualizer()
        """
        if value <= 0.50:
            return value + 0.012
        else:
            return value + 0.008

    def _match_by_pointer(self, ptr_angle: float) -> Optional[Tuple[str, Dict]]:
        """根据指针角度匹配最可能的预设量程"""
        best_pid = None
        best_score = float('inf')
        for pid, cfg in self._preset_ranges.items():
            # 指针是否在量程范围内
            if self._angle_span(cfg['min_angle'], ptr_angle) <= cfg['span']:
                # 在范围内 → 匹配
                mid = (cfg['min_angle'] + cfg['span'] / 2) % 360
                diff = min(abs(ptr_angle - mid), 360 - abs(ptr_angle - mid))
                if diff < best_score:
                    best_score = diff
                    best_pid = pid

        if best_pid:
            return best_pid, {
                **self._preset_ranges[best_pid],
                'anchors': self.presets[best_pid].get('anchors', [])
            }
        return None

    def match_preset_by_span(self, start_angle: float, end_angle: float
                              ) -> Optional[Tuple[str, Dict]]:
        """根据角度范围匹配预设"""
        span = self._angle_span(start_angle, end_angle)
        best_pid = None
        best_diff = float('inf')
        for pid, cfg in self._preset_ranges.items():
            diff = abs(cfg['span'] - span)
            if diff < best_diff and diff < 45:
                best_diff = diff
                best_pid = pid
        if best_pid:
            return best_pid, {
                **self._preset_ranges[best_pid],
                'anchors': self.presets[best_pid].get('anchors', [])
            }
        return None
