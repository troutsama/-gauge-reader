"""
YOLOv8-Pose关键点读取器 V2
适配3类标注: pointer_rect(指针两端) + left_rect(刻度起点) + right_rect(刻度终点)
每个类都有kpt_shape=[2,3]，其中left_rect/right_rect只用第1个关键点
"""
import math
import logging
from typing import Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# 类别索引
CLS_POINTER = 0  # pointer_rect: 指针矩形, 2个关键点(两端)
CLS_LEFT = 1     # left_rect: 刻度起点, 1个有效关键点
CLS_RIGHT = 2    # right_rect: 刻度终点, 1个有效关键点


class KeypointReaderV2:
    """基于YOLOv8-Pose的指针+刻度读取器 (3类版)"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        import torch
        if torch.cuda.is_available():
            try:
                self.model.model.half()
                self.model.model.fuse()
            except Exception:
                pass
        logger.info(f"关键点模型V2已加载: {self.model_path}")

    def extract(self, roi: np.ndarray, gauge_center: Tuple[float, float] = None) -> Optional[dict]:
        """
        从ROI中提取指针和刻度关键点.
        参数:
            roi: 表盘裁剪区域 (RGB)
            gauge_center: 可选的表盘中心坐标 (来自检测器), 用于更准确的枢轴/针尖判定
        返回: {
            'angle': float,           # 指针角度 (12点顺时针)
            'pivot': (x,y),           # 指针近端点
            'tip': (x,y),             # 指针远端
            'confidence': float,      # 综合置信度 (box+关键点)
            'pointer_conf': float,    # 指针关键点平均置信度
            'box_conf': float,        # 检测框置信度
            'scale_points': [(x,y), (x,y)] | None,  # 刻度起止点
        }
        """
        if self.model is None:
            return None

        h, w = roi.shape[:2]

        # 缩放适配训练分布 (小表盘不缩小, 避免模糊指针细节)
        scale_factor = 1.0
        if max(w, h) > 800:
            scale_factor = 550.0 / max(w, h)
        elif max(w, h) < 300:
            # 放大到模型输入尺寸以上才放大, 否则保持原始分辨率
            if max(w, h) < 200:
                scale_factor = 400.0 / max(w, h)

        if scale_factor != 1.0:
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            roi_resized = roi

        try:
            import torch
            with torch.inference_mode():
                results = self.model(roi_resized, conf=0.15, verbose=False)
        except Exception as e:
            logger.warning(f"关键点推理失败: {e}")
            return None

        r = results[0]
        if r.keypoints is None or len(r.keypoints.data) == 0:
            return None

        # 收集各类预测
        pointer_kps = []   # [(conf, [kp0, kp1])]
        left_kps = []      # [(conf, (x,y))]
        right_kps = []     # [(conf, (x,y))]

        boxes = r.boxes
        kps_data = r.keypoints.data

        if boxes is None or kps_data is None:
            return None

        for i, (box, kp_tensor) in enumerate(zip(boxes, kps_data)):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            kp = kp_tensor.cpu().numpy()  # shape [2, 3]

            if cls == CLS_POINTER:
                pointer_kps.append((conf, kp))
            elif cls == CLS_LEFT:
                if kp[0][2] > 0:
                    left_kps.append((conf, (kp[0][0], kp[0][1])))
            elif cls == CLS_RIGHT:
                if kp[0][2] > 0:
                    right_kps.append((conf, (kp[0][0], kp[0][1])))

        # --- 提取指针角度 ---
        if len(pointer_kps) == 0:
            return None  # 没有指针 → 失败

        # 选置信度最高的指针
        pointer_kps.sort(key=lambda x: -x[0])
        best_box_conf, best_kp = pointer_kps[0]

        # ★ 关键点置信度检查 (修复致命缺陷: 之前只检查box.conf)
        kp_conf_0 = float(best_kp[0][2])  # 第1个关键点置信度
        kp_conf_1 = float(best_kp[1][2])  # 第2个关键点置信度
        kp_conf_mean = (kp_conf_0 + kp_conf_1) / 2.0
        kp_conf_min = min(kp_conf_0, kp_conf_1)

        # 综合置信度: box + 两关键点的均值
        combined_conf = (best_box_conf + kp_conf_mean) / 2.0

        # 任一关键点极低 → 标记为不可靠 (但不直接丢弃, 让调用方决策)
        pointer_lowconf = (kp_conf_min < 0.15)

        # kp[0] 和 kp[1] 是指针两端 (顺序任意)
        p0 = (best_kp[0][0] / scale_factor, best_kp[0][1] / scale_factor)
        p1 = (best_kp[1][0] / scale_factor, best_kp[1][1] / scale_factor)

        # ★ 指针长度检查: 太短的"指针"很可能是刻度线
        ptr_length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        crop_diag = math.hypot(w, h)
        length_ratio = ptr_length / max(crop_diag, 1.0)

        # 指针应至少占裁剪对角线的25% (训练数据中指针bbox占图32-42%)
        # 如果<20%，几乎不可能是真正的指针
        if length_ratio < 0.20:
            pointer_lowconf = True  # 强制标记为不可靠，后续回退

        # 角度: p0→p1, 12点顺时针
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        angle = math.degrees(math.atan2(dx, -dy)) % 360

        # 枢轴/针尖: 使用检测中心 (优先) 或 ROI 几何中心
        if gauge_center is not None:
            ref_center = gauge_center
        else:
            ref_center = (w / 2, h / 2)
        d0 = math.hypot(p0[0] - ref_center[0], p0[1] - ref_center[1])
        d1 = math.hypot(p1[0] - ref_center[0], p1[1] - ref_center[1])
        if d0 < d1:
            pivot, tip = p0, p1
        else:
            pivot, tip = p1, p0

        # CV精修针尖 —— 约束在指针方向扇形区域内，避免检测到刻度线
        refined_tip = None
        rh, rw = roi_resized.shape[:2]
        # 检查关键点预测是否可疑: 针尖到枢轴距离远超刻度环 (刻度线/表盘边缘被当针尖)
        kp_ptr_len = math.hypot(tip[0] - pivot[0], tip[1] - pivot[1])
        scale_radius_ref = None
        if len(left_kps) > 0 and len(right_kps) > 0:
            l_pt = left_kps[0][1]
            r_pt = right_kps[0][1]
            sr1 = math.hypot(l_pt[0] - pivot[0], l_pt[1] - pivot[1])
            sr2 = math.hypot(r_pt[0] - pivot[0], r_pt[1] - pivot[1])
            scale_radius_ref = (sr1 + sr2) / 2

        # 指针长度应约为刻度半径的 0.7~1.3 倍 (针尖到达刻度环附近)
        kp_tip_suspicious = False
        if scale_radius_ref and scale_radius_ref > 10:
            len_ratio_to_scale = kp_ptr_len / scale_radius_ref
            if len_ratio_to_scale > 1.3 or len_ratio_to_scale < 0.7:
                kp_tip_suspicious = True

        if max(rw, rh) <= 800 and not pointer_lowconf:
            kp_ptr_angle = angle  # 已计算的12点顺时针角度
            # 关键点针尖可疑时, 用更宽的扇形(±60°)搜索, 摆脱错误方向
            half_angle = 60 if kp_tip_suspicious else 25
            refined_tip = self.refine_tip_cv(roi_resized, pivot, kp_ptr_angle,
                                             half_angle=half_angle)

        if refined_tip is not None:
            cv_tip = (refined_tip[0] / scale_factor, refined_tip[1] / scale_factor)
            # 检查CV角度与关键点角度是否一致 (差>20°则CV检错了线)
            kp_dx = tip[0] - pivot[0]
            kp_dy = tip[1] - pivot[1]
            cv_dx = cv_tip[0] - pivot[0]
            cv_dy = cv_tip[1] - pivot[1]
            kp_angle = math.degrees(math.atan2(kp_dx, -kp_dy)) % 360
            cv_angle = math.degrees(math.atan2(cv_dx, -cv_dy)) % 360
            angle_diff = min(abs(kp_angle - cv_angle), 360 - abs(kp_angle - cv_angle))

            # 针尖可疑时放宽一致性检查 (关键点本身可能错, 信任CV)
            consistent_threshold = 40 if kp_tip_suspicious else 20
            if angle_diff < consistent_threshold:  # CV与关键点一致 → 采用CV精修的针尖
                tip = cv_tip
                angle = cv_angle

        # --- 智能左右刻度推断 ---
        scale_info = {}
        # 当缺失一侧但另一侧有多个检测 → 可能是误分类(模型把left判成right)
        if len(left_kps) == 0 and len(right_kps) >= 2 and len(pointer_kps) > 0:
            # 用指针方向判断：指针从pivot指向tip，顺时针前方的是right，后方的是left
            dx_tip = tip[0] - pivot[0]
            dy_tip = tip[1] - pivot[1]
            ptr_angle = math.degrees(math.atan2(dx_tip, -dy_tip)) % 360

            right_kps.sort(key=lambda x: -x[0])
            # 取最高置信度2个，按角度分左右
            top2 = right_kps[:2]
            angles = []
            for conf, (rx, ry) in top2:
                a = math.degrees(math.atan2(rx - pivot[0], -(ry - pivot[1]))) % 360
                angles.append(a)

            # 顺时针从ptr到a1的距离 vs ptr到a2的距离
            def cw_dist(a1, a2):
                return (a2 - a1) % 360

            d1 = cw_dist(ptr_angle, angles[0])
            d2 = cw_dist(ptr_angle, angles[1])
            # 距离小的在指针顺时针前方 → right; 距离大的在后方 → left
            if d1 < d2:
                right_kps = [top2[0]]
                left_kps = [(top2[1][0], top2[1][1])]
            else:
                right_kps = [top2[1]]
                left_kps = [(top2[0][0], top2[0][1])]
            scale_info['split_right'] = True

        elif len(right_kps) == 0 and len(left_kps) >= 2 and len(pointer_kps) > 0:
            # 对称情况：2个left_rect检测 → 用指针分左右
            dx_tip = tip[0] - pivot[0]
            dy_tip = tip[1] - pivot[1]
            ptr_angle = math.degrees(math.atan2(dx_tip, -dy_tip)) % 360

            left_kps.sort(key=lambda x: -x[0])
            top2 = left_kps[:2]
            angles = []
            for conf, (lx, ly) in top2:
                a = math.degrees(math.atan2(lx - pivot[0], -(ly - pivot[1]))) % 360
                angles.append(a)

            d1 = (ptr_angle - angles[0]) % 360
            d2 = (ptr_angle - angles[1]) % 360
            # 距离小的(指针逆时针方向更近) → left
            if d1 < d2:
                left_kps = [top2[0]]
                right_kps = [(top2[1][0], top2[1][1])]
            else:
                left_kps = [top2[1]]
                right_kps = [(top2[0][0], top2[0][1])]
            scale_info['split_left'] = True

        # --- 提取刻度起止点 ---
        scale_points = None
        if len(left_kps) > 0 and len(right_kps) > 0:
            left_kps.sort(key=lambda x: -x[0])
            right_kps.sort(key=lambda x: -x[0])
            sp_left = (left_kps[0][1][0] / scale_factor,
                         left_kps[0][1][1] / scale_factor)
            sp_right = (right_kps[0][1][0] / scale_factor,
                          right_kps[0][1][1] / scale_factor)
            scale_points = (sp_left, sp_right)
        elif len(left_kps) > 0:
            left_kps.sort(key=lambda x: -x[0])
            scale_info['partial'] = 'left_only'
            scale_info['left_point'] = (left_kps[0][1][0] / scale_factor,
                                          left_kps[0][1][1] / scale_factor)
        elif len(right_kps) > 0:
            right_kps.sort(key=lambda x: -x[0])
            scale_info['partial'] = 'right_only'
            scale_info['right_point'] = (right_kps[0][1][0] / scale_factor,
                                           right_kps[0][1][1] / scale_factor)

        return {
            'angle': angle,
            'pivot': pivot,
            'tip': tip,
            'confidence': combined_conf,      # 综合置信度 (box+kp均值)
            'pointer_conf': kp_conf_mean,     # 指针关键点平均置信度
            'box_conf': best_box_conf,        # 检测框置信度
            'kp_conf_min': kp_conf_min,       # 两关键点中较低的置信度
            'pointer_lowconf': pointer_lowconf,  # 关键点是否不可靠
            'ptr_length': ptr_length,         # 指针长度 (像素)
            'length_ratio': length_ratio,     # 指针长度/裁剪对角线
            'scale_points': scale_points,
            'scale_info': scale_info,
        }

    def refine_tip_cv(self, roi: np.ndarray, pivot: Tuple[float, float],
                       pointer_angle: float = None, half_angle: float = 25
                       ) -> Optional[Tuple[float, float]]:
        """
        CV精修指针针尖 —— 参考Nanodet-YOLOv8-Pose仓库getPointerLines方案
        在ROI上做灰度→均衡化→反相→中值滤波→腐蚀→闭运算→阈值→骨架化→霍夫线

        pointer_angle约束 —— 只在预测方向±half_angle扇形内搜索，
        避免检测到刻度线/文字/边缘等其他线性结构。
        (默认25°, 关键点针尖可疑时用60°宽扇形摆脱错误方向)
        """
        import cv2
        h, w = roi.shape[:2]
        if h < 20 or w < 20:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        inv = cv2.bitwise_not(eq)
        blur = cv2.medianBlur(inv, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded = cv2.erode(blur, kernel)
        closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
        thresh_val = max(np.mean(closed) * 1.3, 180)
        _, binary = cv2.threshold(closed, thresh_val, 255, cv2.THRESH_BINARY)

        # ★ 扇形约束: 只保留指针方向±half_angle内的像素
        if pointer_angle is not None:
            wedge_mask = self._create_wedge_mask(
                binary.shape, pivot, pointer_angle, half_angle=half_angle)
            binary = cv2.bitwise_and(binary, binary, mask=wedge_mask)

        # 骨架化（距离变换法，快速）
        if max(binary.shape) > 600:
            scale = 500.0 / max(binary.shape)
            bw, bh = int(binary.shape[1]*scale), int(binary.shape[0]*scale)
            binary_small = cv2.resize(binary, (bw, bh))
        else:
            binary_small = binary
            scale = 1.0

        dist = cv2.distanceTransform(binary_small, cv2.DIST_L2, 5)
        dilated = cv2.dilate(dist, np.ones((3, 3), np.uint8))
        skeleton_small = ((dist == dilated) & (dist > 0)).astype(np.uint8) * 255

        if scale < 1.0:
            skeleton = cv2.resize(skeleton_small, (binary.shape[1], binary.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        else:
            skeleton = skeleton_small

        if np.sum(skeleton > 0) < 20:
            return None

        lines = cv2.HoughLinesP(skeleton, 1, np.pi/180,
                                threshold=30, minLineLength=30, maxLineGap=50)
        if lines is None or len(lines) == 0:
            return None

        # 找最长线 (兼容不同OpenCV版本的HoughLinesP返回格式)
        def line_len(l):
            if len(l.shape) == 2 and l.shape[1] == 4:   # (1,4) 格式
                return np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2)
            else:                                         # (4,) 格式
                return np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2)
        best = max(lines, key=line_len)
        if len(best.shape) == 2:
            x1, y1, x2, y2 = best[0]
        else:
            x1, y1, x2, y2 = best

        # 两端点: 离pivot更远的 = 针尖
        d1 = math.hypot(x1 - pivot[0], y1 - pivot[1])
        d2 = math.hypot(x2 - pivot[0], y2 - pivot[1])
        if d1 > d2:
            return (float(x1), float(y1))
        else:
            return (float(x2), float(y2))

    @staticmethod
    def _create_wedge_mask(shape, pivot, pointer_angle, half_angle=25):
        """创建扇形mask: 以pivot为中心, pointer_angle±half_angle方向内的像素为255"""
        import cv2
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        px, py = int(pivot[0]), int(pivot[1])

        # 将12点顺时针角度转为从pivot出发的数学角度 (弧度)
        # pointer_angle: 12点顺时针, 0=12点, 90=3点
        # 转为atan2坐标系: 从+x轴逆时针
        math_angle = math.radians(90 - pointer_angle)  # 转为数学角

        # 对每个像素检查是否在扇形内 (只用pivot附近的带状区域检查)
        # 更高效: 用多边形填充
        r = max(h, w) * 1.5  # 足够长的射线
        a1 = pointer_angle - half_angle
        a2 = pointer_angle + half_angle

        # 多个角度步进, 构建扇形多边形
        steps = max(10, half_angle)  # 度数步进
        pts = [(px, py)]
        for a in np.arange(a1, a2 + 1, steps):
            rad = math.radians(90 - a)  # 转为数学角
            x = int(px + r * math.cos(rad))
            y = int(py - r * math.sin(rad))
            pts.append((x, y))
        # 闭合回pivot
        pts.append((px, py))

        pts_array = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts_array], 255)
        return mask

    def scale_points_to_angles(self, center, scale_points):
        """将刻度起止点转为12点顺时针角度"""
        if scale_points is None:
            return None
        angles = []
        for px, py in scale_points:
            dx = px - center[0]
            dy = py - center[1]
            a = math.degrees(math.atan2(dx, -dy)) % 360
            angles.append(a)
        return sorted(angles)
