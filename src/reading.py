"""
读数计算模块
- 角度法：指针角度 → 线性插值 → 实际读数
- 距离法：相邻刻度间距 → 指针位置 → 实际读数
- 支持环绕处理（360°表盘）
- 支持外推（指针超出锚点范围）
"""

import logging
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReadingResult:
    """读数计算结果"""
    value: float  # 最终读数值
    unit: str = ""  # 单位 (MPa, V, A, ℃, kPa, ...)
    method: str = "angle_interpolation"  # 计算方法
    confidence: float = 1.0  # 置信度
    raw_angle: Optional[float] = None  # 原始指针角度
    anchors_used: List[Tuple[float, float]] = None  # 使用的锚点
    interpolation_factor: float = 0.0  # 插值系数 (0-1)
    details: dict = None  # 额外调试信息

    def __post_init__(self):
        if self.anchors_used is None:
            self.anchors_used = []
        if self.details is None:
            self.details = {}

    def __repr__(self):
        return f"ReadingResult({self.value:.2f} {self.unit}, conf={self.confidence:.2f})"


class ReadingCalculator:
    """
    读数计算器
    核心算法：角度线性插值（含环绕处理）
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        read_cfg = cfg.get('reading', {})

        self.method = read_cfg.get('method', 'angle_interpolation')
        self.sweep_angle_default = read_cfg.get('sweep_angle_default', 270)
        self.extrapolation = read_cfg.get('extrapolation', 'linear')

    # ============================================================
    # 主入口：角度法
    # ============================================================

    def compute(self,
                pointer_angle: float,
                anchors: List[Tuple[float, float]],
                sweep_angle: Optional[float] = None,
                unit: str = "",
                gauge_type: str = "circular") -> ReadingResult:
        """
        角度插值法计算读数
        参数:
            pointer_angle: 指针角度 (从12点顺时针, 0-360°)
            anchors: 锚点列表 [(angle1, value1), (angle2, value2), ...]
            sweep_angle: 表盘扫过角度 (如270°)，不传则自动估算
            unit: 单位字符串
            gauge_type: 仪表类型
        返回:
            ReadingResult 对象
        """
        if len(anchors) < 2:
            logger.warning("锚点不足2个，无法计算读数")
            return ReadingResult(
                value=0.0, unit=unit,
                confidence=0.0,
                raw_angle=pointer_angle,
                details={"error": "insufficient_anchors"}
            )

        # --- 将锚点按正确扫表顺序（顺时针）排列 ---
        # 圆形表盘的锚点可能跨360°，简单按角度排序会打乱刻度对应关系
        # 正确做法：找到死区（值跳跃最大的相邻锚点对），以此为界重排
        sorted_by_angle = sorted(anchors, key=lambda x: x[0])
        n = len(sorted_by_angle)

        # 计算相邻锚点间的值跳跃（绝对值差）
        jumps = []
        for i in range(n):
            v1 = sorted_by_angle[i][1]
            v2 = sorted_by_angle[(i + 1) % n][1]
            jumps.append(abs(v2 - v1))

        max_jump = max(jumps)
        split_idx = jumps.index(max_jump)
        sorted_jumps = sorted(jumps, reverse=True)
        second_jump = sorted_jumps[1] if len(sorted_jumps) >= 2 else 0

        # 死区特征：值从量程最大值跳回最小值（或反之），跳跃量最大
        # 仅当最大跳跃明显大于次大跳跃时才认为存在真正的死区
        if max_jump > 1.3 * second_jump and max_jump > 0:
            # 死区在 split_idx → split_idx+1 之间，扫表从 split_idx+1 开始
            start_idx = (split_idx + 1) % n
            anchors = sorted_by_angle[start_idx:] + sorted_by_angle[:start_idx]
        else:
            # 无明显死区（全360°表盘或仅2个锚点），保持角度排序
            anchors = sorted_by_angle
        angles = [a for a, _ in anchors]
        values = [v for _, v in anchors]

        # 自动估算 sweep_angle
        if sweep_angle is None:
            sweep_angle = self._estimate_sweep_angle(angles, gauge_type)

        # 锚点覆盖范围（顺时针从第一个锚点到最后一个锚点）
        min_angle = angles[0]
        max_angle = angles[-1]
        if max_angle >= min_angle:
            angle_range = max_angle - min_angle
        else:
            angle_range = max_angle + 360 - min_angle  # 跨360°环绕

        # 指针是否在锚点覆盖范围内
        if self._is_within_range(pointer_angle, min_angle, max_angle, angle_range):
            # 标准情况：指针在两个锚点之间
            value, interp_factor = self._interpolate(pointer_angle, anchors)
        else:
            # 指针在锚点范围外（死区），需要外推
            if self.extrapolation == 'linear':
                value, interp_factor = self._extrapolate(pointer_angle, anchors, sweep_angle)
            else:
                # nearest: 取最近的锚点值
                nearest = min(anchors, key=lambda a: self._angle_diff(pointer_angle, a[0]))
                value = nearest[1]
                interp_factor = 0.0

        # 计算置信度
        confidence = self._compute_confidence(pointer_angle, anchors, interp_factor)

        return ReadingResult(
            value=float(value),
            unit=unit,
            method=self.method,
            confidence=confidence,
            raw_angle=pointer_angle,
            anchors_used=anchors,
            interpolation_factor=float(interp_factor),
            details={
                "sweep_angle": sweep_angle,
                "angle_range": angle_range,
                "min_angle": min_angle,
                "max_angle": max_angle,
                "anchor_count": len(anchors),
            }
        )

    # ============================================================
    # 内插
    # ============================================================

    def _interpolate(self, pointer_angle: float,
                     anchors: List[Tuple[float, float]]
                     ) -> Tuple[float, float]:
        """
        线性内插：找到指针落在哪两个锚点之间
        anchors 已按扫表顺序排列（顺时针，值单调变化）
        返回: (value, interpolation_factor)
        """
        for i in range(len(anchors) - 1):
            a1, v1 = anchors[i]
            a2, v2 = anchors[i + 1]

            if a1 <= a2:
                # 正常区间（不跨360°）
                if a1 <= pointer_angle <= a2:
                    t = (pointer_angle - a1) / (a2 - a1) if a2 != a1 else 0
                    value = v1 + t * (v2 - v1)
                    return value, t
            else:
                # 跨360°环绕区间（如 315°→0°）
                if pointer_angle >= a1 or pointer_angle <= a2:
                    pw = pointer_angle if pointer_angle >= a1 else pointer_angle + 360
                    a2w = a2 + 360
                    t = (pw - a1) / (a2w - a1) if a2w != a1 else 0
                    value = v1 + t * (v2 - v1)
                    return value, t

        # 死区回退：指针在最后一个锚点到第一个锚点之间（顺时针通过死区）
        a1, v1 = anchors[-1]
        a2, v2 = anchors[0]

        if pointer_angle >= a1:
            dist_to_p = pointer_angle - a1
        else:
            dist_to_p = pointer_angle + 360 - a1

        if a2 >= a1:
            total_dist = a2 - a1
        else:
            total_dist = a2 + 360 - a1

        if total_dist == 0:
            return v1, 0.0

        t = dist_to_p / total_dist
        value = v1 + t * (v2 - v1)
        return value, t

    # ============================================================
    # 外推
    # ============================================================

    def _extrapolate(self, pointer_angle: float,
                     anchors: List[Tuple[float, float]],
                     sweep_angle: float) -> Tuple[float, float]:
        """
        线性外推：指针在死区内，用最近的端点锚点对做线性外推
        改进：限制外推幅度，防止过度外推
        """
        start_angle = anchors[0][0]
        end_angle = anchors[-1][0]

        # 顺时针从 end 到 pointer 的距离
        if pointer_angle >= end_angle:
            dist_from_end = pointer_angle - end_angle
        else:
            dist_from_end = pointer_angle + 360 - end_angle

        # 顺时针从 pointer 到 start 的距离
        if start_angle >= pointer_angle:
            dist_to_start = start_angle - pointer_angle
        else:
            dist_to_start = start_angle + 360 - pointer_angle

        # 计算两端锚点间距（用于限制外推幅度）
        if len(anchors) >= 2:
            a_first = anchors[0][0]
            a_last = anchors[-1][0]
            if a_last >= a_first:
                anchor_span = a_last - a_first
            else:
                anchor_span = a_last + 360 - a_first
        else:
            anchor_span = 270  # 默认值

        # 选择更近的一端做外推
        if dist_from_end <= dist_to_start:
            a1, v1 = anchors[-2]
            a2, v2 = anchors[-1]
            is_extrapolate_forward = True  # 超过终点
            dist_from_anchor = dist_from_end
        else:
            a1, v1 = anchors[0]
            a2, v2 = anchors[1]
            is_extrapolate_forward = False  # 在起点之前
            dist_from_anchor = dist_to_start

        # 展开角度到以 a1 为基准的连续空间
        if a2 >= a1:
            a2_u = a2
        else:
            a2_u = a2 + 360  # 跨360°的情况

        # 指针角度展开：分方向处理
        if is_extrapolate_forward:
            # 指针超过终点：展开到a1之后
            if pointer_angle >= a1:
                p_u = pointer_angle
            else:
                p_u = pointer_angle + 360
        else:
            # 指针在起点之前（顺时针从终点通过死区到达指针，指针还没到起点）
            # 指针应该被理解为在a1之前（较小的角度值），不加360
            # 但如果指针>a1（如 a1=34°, ptr=256° 的情况），需要减360
            if pointer_angle <= a1:
                p_u = pointer_angle  # 正常：指针在a1之前
            else:
                p_u = pointer_angle - 360  # 指针实际在a1之后（跨360°），视为在a1之前

        # 锚点对的跨步
        step_span = a2_u - a1
        if step_span <= 0:
            step_span = anchor_span / max(1, len(anchors) - 1)

        t = (p_u - a1) / step_span

        # 钳制外推幅度
        if t > 1.5:
            t = 1.5
        elif t < -0.5:
            t = -0.5

        value = v1 + t * (v2 - v1)
        return value, t

    # ============================================================
    # 辅助
    # ============================================================

    @staticmethod
    def _is_within_range(angle: float, min_a: float, max_a: float,
                         angle_range: float) -> bool:
        """判断角度是否在锚点覆盖范围内"""
        if angle_range >= 360:
            return True  # 全覆盖
        if min_a <= max_a:
            # 正常情况：扫表不跨360°
            return min_a <= angle <= max_a
        else:
            # 环绕情况：扫表跨360°（如225°→45°），角度在 min_a~360 或 0~max_a
            return angle >= min_a or angle <= max_a

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """两个角度的最小差（0-180°）"""
        diff = abs(a - b) % 360
        return min(diff, 360 - diff)

    @staticmethod
    def _estimate_sweep_angle(angles: List[float],
                              gauge_type: str = "circular") -> float:
        """根据锚点角度估算表盘扫过角度（angles 已按扫表顺序排列）"""
        if len(angles) < 2:
            return 270  # 默认圆形表

        first, last = angles[0], angles[-1]
        if last >= first:
            span = last - first
        else:
            span = last + 360 - first  # 跨360°

        if gauge_type == "circular":
            return max(270, span)
        elif gauge_type == "arc":
            return max(90, span)
        else:
            return span

    @staticmethod
    def _compute_confidence(pointer_angle: float,
                            anchors: List[Tuple[float, float]],
                            interp_factor: float) -> float:
        """估算读数置信度"""
        # 锚点越多，置信度越高
        anchor_score = min(1.0, len(anchors) / 5.0)

        # 插值越在中间（0.2-0.8），置信度越高
        interp_score = 1.0 - 2.0 * abs(interp_factor - 0.5)

        # 综合
        return float(0.5 * anchor_score + 0.5 * interp_score)

    # ============================================================
    # 距离法（备选）
    # ============================================================

    def compute_by_distance(self,
                            pointer_angle: float,
                            scale_angles: List[float],
                            anchors: List[Tuple[float, float]],
                            unit: str = "") -> ReadingResult:
        """
        距离-刻度映射法
        通过相邻刻度间距+指针位置计算读数
        （适用于刻度线均匀分布的仪表）
        """
        if len(anchors) < 2 or len(scale_angles) < 2:
            return self.compute(pointer_angle, anchors, unit=unit)

        # 方法同角度法，但利用全部刻度线做辅助定位
        # 这里简化为角度法 + 所有刻度线做密度验证
        result = self.compute(pointer_angle, anchors, unit=unit)

        # 用刻度线间距验证线性假设
        angles_sorted = sorted(scale_angles)
        gaps = np.diff(angles_sorted)
        if len(gaps) > 3:
            gap_std = np.std(gaps)
            gap_mean = np.mean(gaps)
            if gap_std / (gap_mean + 1e-10) < 0.2:
                # 刻度线均匀，提升置信度
                result.confidence = min(1.0, result.confidence + 0.1)

        result.method = "distance_interpolation"
        return result
