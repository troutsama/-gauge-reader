"""
完整读数管线 (最终版)
V3 模型(指针+刻度) + EasyOCR(读最大量程) + 角度比例法读数

用法:
    reader = FullReader()
    reader.load_models()
    result = reader.read(image)  # {'reading', 'max_value', 'angle', ...}
"""
import math
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2

from src.keypoint_reader_v2 import KeypointReaderV2
from src.config import KEYPOINT_PATH

logger = logging.getLogger(__name__)

# 量程识别不再依赖预设列表 (用表盘数字的小数点/前导0信号推断, 支持任意量程)


class FullReader:
    """V3 + OCR 端到端读数器"""

    def __init__(self, keypoint_path: str = None, min_value: float = 0.0,
                 use_ensemble: bool = True):
        self.keypoint_path = keypoint_path or KEYPOINT_PATH
        self.min_value = min_value
        self.use_ensemble = use_ensemble  # 多模型投票
        self.keypoint = None
        self.keypoint_ensemble = []  # 投票用附加模型
        self.ocr = None
        self.digit_det = None  # 值分类数字检测器
        # 数字检测器的类别 = 数字值 (与 digit_det_dataset3 训练一致, 19类)
        self.VALUE_CLASSES = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.4, 0.8,
                              1.2, 1.6, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0,
                              15.0, 20.0, 25.0]

    def load_models(self):
        import torch
        from ultralytics import YOLO
        self.keypoint = KeypointReaderV2(self.keypoint_path)
        self.keypoint.load_model()

        # 附加模型用于环形中位数投票 (可选, 有就用)
        if self.use_ensemble:
            from src.config import KEYPOINT_V3
            for extra_path in [KEYPOINT_V3]:
                try:
                    if extra_path and extra_path != self.keypoint_path and Path(extra_path).exists():
                        ekp = KeypointReaderV2(extra_path)
                        ekp.load_model()
                        self.keypoint_ensemble.append(ekp)
                except Exception as e:
                    logger.warning(f"附加模型加载失败 {extra_path}: {e}")

        # 数字检测器 (值分类, 检测刻度数字框 + 直接输出值)
        try:
            from src.config import DIGIT_DET_PATH
            if Path(DIGIT_DET_PATH).exists():
                self.digit_det = YOLO(DIGIT_DET_PATH)
                logger.info(f"数字检测器已加载: {DIGIT_DET_PATH}")
        except Exception as e:
            logger.warning(f"数字检测器加载失败: {e}")

        if torch.cuda.is_available():
            models = [self.keypoint.model] + [e.model for e in self.keypoint_ensemble]
            if self.digit_det:
                models.append(self.digit_det)
            for m in models:
                try:
                    m.model.half()
                    m.model.fuse()
                except Exception:
                    pass

        import easyocr
        import logging as _logging
        _logging.disable(_logging.WARNING)
        self.ocr = easyocr.Reader(['en'], gpu=True, verbose=False)
        logger.info(f"V3 + {len(self.keypoint_ensemble)}附加模型 + EasyOCR + 数字检测器 已加载")

    @staticmethod
    def _circular_median(angles):
        """环形中位数: 取使到各角度总距离最小的候选"""
        if not angles:
            return None
        if len(angles) == 1:
            return angles[0]
        best_a, best_d = None, float('inf')
        for cand in angles:
            d = sum(min(abs(cand - a), 360 - abs(cand - a)) for a in angles)
            if d < best_d:
                best_d, best_a = d, cand
        return best_a

    # ============================================================
    # OCR 量程识别
    # ============================================================

    @staticmethod
    def _parse_number(text: str) -> Optional[float]:
        """解析 OCR 数字文本, 拒绝噪声"""
        t = text.replace(' ', '')
        if not t or not any(c.isdigit() for c in t):
            return None
        cleaned = re.sub(r'[^0-9.]', '', t)
        if not cleaned or cleaned.count('.') > 1:
            return None
        try:
            v = float(cleaned)
            # 前导0的两位数 (如 "04") 实际是小数 "0.4"
            if '.' not in cleaned and len(cleaned) == 2 and cleaned[0] == '0':
                return v / 10
            return v
        except ValueError:
            return None

    @staticmethod
    def _in_arc(angle, start, end, tolerance=12):
        """判断 angle 是否在 start→end 的顺时针弧段内 (含端点, 带容差)"""
        # 容差: 数字有宽度, 中心可能略越过端点, 但仍是刻度数字
        if end >= start:
            return start - tolerance <= angle <= end + tolerance
        return angle >= start - tolerance or angle <= end + tolerance

    def _infer_max_value(self, nums, left_a: float, right_a: float) -> Optional[float]:
        """
        从数字+空间位置推断最大量程值 (nums: [(angle, radius_ratio, text)])
        核心: 用"表盘上是否有小数点/前导0数字"判断小数刻度, 从而正确补小数点
        不依赖预设量程列表, 支持任意量程
        """
        # 只保留"位于左刻度→右刻度顺时针弧段内"的数字 (带容差)
        # 且半径比在刻度环附近 (0.35~1.05), 避免远处噪声(如背景序列号)混入
        # (序列号/品牌字在表盘下方缺口, 即弧段之外或远离刻度环)
        arc_nums = []
        for a, r, text in nums:
            if self._in_arc(a, left_a, right_a) and 0.35 <= r <= 1.05:
                arc_nums.append((a, text))
        if not arc_nums:
            return None

        # 找最接近右刻度的有效数字 = 最大量程
        # 跳过残缺数字(如"0."无尾数, parse出0/None的)
        max_text = None
        best_dist = float('inf')
        for a, text in arc_nums:
            cand = self._parse_number(text)
            if cand is None or cand == 0:  # 跳过残缺/零值
                continue
            d = min(abs(a - right_a), 360 - abs(a - right_a))
            if d < best_dist:
                best_dist, max_text, raw = d, text, cand
        if max_text is None:
            return None

        # 小数刻度信号: 弧内任一数字带小数点, 或前导0两位数(如"04"=0.4)
        # 只看弧内数字, 避免弧外品牌字/噪声误触发
        has_decimal = any('.' in text for _, text in arc_nums)
        has_leading_zero = any(
            len(re.sub(r'[^0-9]', '', text)) == 2
            and re.sub(r'[^0-9]', '', text)[0] == '0'
            for _, text in arc_nums)
        is_decimal_scale = has_decimal or has_leading_zero

        # 若小数刻度且右刻度是>=10的整数(漏了小数点), 补 /10
        if is_decimal_scale and raw == int(raw) and raw >= 10:
            raw = raw / 10

        # ★ 量程合理性校验 (B线): 拦离谱量程
        #   用邻近刻度的数字序列判断, 若量程和序列完全不成比例 → 误读, 用序列最大值兜底
        #   保守: 只对比"有小数刻度信号"的序列, 且不吃远距噪声
        seq_vals = []
        for _, text in arc_nums:
            v = self._parse_number(text)
            if v is not None and v > 0:
                seq_vals.append(v)
        if seq_vals:
            seq_sorted = sorted(seq_vals)
            # 去重(容差1%)
            uniq = []
            for v in seq_sorted:
                if not uniq or abs(v - uniq[-1]) > 0.01 * max(1.0, abs(v)):
                    uniq.append(v)
            if len(uniq) >= 2:
                # 量程与序列最大值比: 若远超(>5倍), 且序列本身合理(相邻差不极端)
                # 则量程是误读, 用序列最大值兜底
                seq_max = uniq[-1]
                if raw > 5 * seq_max and seq_max >= 1:
                    raw = seq_max

        return raw

    def _read_scale_digit(self, image: np.ndarray, scale_pt, pivot,
                          work_image: np.ndarray = None) -> Optional[str]:
        """
        在刻度点内侧(朝圆心方向)裁剪数字区域, 用OCR读取
        比整图OCR更准确(聚焦刻度数字, 避开文字/噪声)
        """
        img = image
        cx, cy = float(pivot[0]), float(pivot[1])
        h, w = img.shape[:2]
        scale_r = math.hypot(scale_pt[0] - cx, scale_pt[1] - cy)
        if scale_r < 10:
            return None

        best, best_conf = None, -1
        # 数字在刻度内侧 0.5~0.7 半径处, 尝试多个位置+尺寸
        for ratio in [0.5, 0.55, 0.6, 0.65, 0.7]:
            nx = scale_pt[0] + (cx - scale_pt[0]) * (1 - ratio)
            ny = scale_pt[1] + (cy - scale_pt[1]) * (1 - ratio)
            size = int(scale_r * 0.5)
            x1, y1 = int(nx) - size // 2, int(ny) - size // 2
            x2, y2 = x1 + size, y1 + size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            res = self.ocr.readtext(crop)
            for _, text, conf in res:
                if not any(c.isdigit() for c in text):
                    continue
                n_digits = sum(c.isdigit() for c in text)
                if n_digits <= 3 and float(conf) > best_conf:
                    best_conf, best = float(conf), text

        return best

    @staticmethod
    def _angle_of(pt, cx, cy):
        return math.degrees(math.atan2(pt[0] - cx, -(pt[1] - cy))) % 360

    def _read_digits_by_detector(self, image: np.ndarray, center) -> List:
        """
        值分类数字检测器: 一次输出数字框 + 数字值 (类别映射到值字符串)
        彻底摆脱OCR, 比"框+OCR读值"准确可靠
        返回列表: [(angle, value_str)]
        """
        if self.digit_det is None:
            return []
        cx, cy = center

        import torch
        with torch.inference_mode():
            r = self.digit_det(image, conf=0.25, verbose=False)
        boxes = r[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        found = []
        for b in boxes:
            bx = (b.xyxy[0][0] + b.xyxy[0][2]) / 2
            by = (b.xyxy[0][1] + b.xyxy[0][3]) / 2
            a = self._angle_of((float(bx), float(by)), cx, cy)
            ci = int(b.cls[0])
            val = self.VALUE_CLASSES[ci] if ci < len(self.VALUE_CLASSES) else None
            if val is not None:
                # 浮点值转字符串: 25.0→'25', 0.4→'0.4' (与_infer_max_value字符串逻辑兼容)
                valstr = str(int(val)) if float(val).is_integer() else str(val)
                found.append((a, valstr))

        return found

    def _detect_rotation(self, image: np.ndarray, kp=None) -> Optional[float]:
        """
        检测表盘内容旋转: 用"左刻度角度 vs 标准225°"估计旋转量
        标准表盘的零刻度在左下(225°), 旋转的量 = 左刻度角 - 225°
        返回旋转角(度), 0或None表示正对无需校正
        """
        if kp is None or not kp.get('scale_points'):
            return 0
        pivot = kp['pivot']
        left = kp['scale_points'][0]
        cx, cy = float(pivot[0]), float(pivot[1])
        left_a = self._angle_of(left, cx, cy)
        rot = (left_a - 225.0) % 360
        # 接近0(正对)或接近360, 视为无旋转
        if rot < 15 or rot > 345:
            return 0
        # 180°倒置少见, 跳过
        if abs(rot - 180) < 15:
            return 0
        return rot

    def _rotate_correct(self, image: np.ndarray, rot: float, center) -> np.ndarray:
        """把表盘按 rot 度旋转摆正 (让左刻度回到225°标准位置)"""
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((center[0], center[1]), rot, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderValue=255)

    # ============================================================
    # 透视矫正 (解决表盘倾斜问题)
    # ============================================================

    def _detect_tilt(self, image: np.ndarray, center=None) -> Optional[dict]:
        """
        检测表盘倾斜: 用霍夫圆定位表盘 + 径向扫描外圈 + 椭圆拟合
        返回 {center, major, minor, angle, ratio} 或 None (无倾斜/检测失败)
        """
        h, w = image.shape[:2]
        if center is None:
            cx, cy = w / 2, h / 2
        else:
            cx, cy = center

        # 转灰度 + 均衡化
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.medianBlur(gray, 5)

        # 霍夫圆定位表盘 (找圆心和近似半径)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=100, param2=30,
                                   minRadius=int(min(h, w) * 0.1),
                                   maxRadius=int(min(h, w) // 2))
        if circles is None:
            return None
        c = circles[0][0]
        cx, cy, r0 = float(c[0]), float(c[1]), float(c[2])
        if r0 < 20:
            return None

        # 二值化找表盘区域
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 5)

        # 沿 72 个方向找表盘外圈半径
        radii = []
        for i in range(72):
            ang = np.deg2rad(i * 5)
            # 从外向内找第一个表盘像素
            found = r0
            for step in range(int(r0 * 1.3), int(r0 * 0.6), -3):
                xx = int(cx + np.cos(ang) * step)
                yy = int(cy + np.sin(ang) * step)
                if 0 <= xx < w and 0 <= yy < h and binary[yy, xx] > 0:
                    found = step
                    break
            radii.append(found)
        radii = np.array(radii)

        # 排除 outlier: 半径超过中位数 ±25% 的方向丢弃 (刻度线/指针的杂点)
        # 真实表盘半径应大致一致, 只有少数方向的刻度/指针会偏离
        med = np.median(radii)
        valid_idx = np.abs(radii - med) <= med * 0.25
        if valid_idx.sum() < 20:  # 有效方向太少则放弃
            return None

        # 拟合椭圆 (只用有效方向)
        pts = []
        for i, r in enumerate(radii):
            if not valid_idx[i]:
                continue
            ang = np.deg2rad(i * 5)
            pts.append([cx + np.cos(ang) * r, cy + np.sin(ang) * r])
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        try:
            (ex, ey), (MA, ma), angle = cv2.fitEllipse(pts)
        except cv2.error:
            return None

        if MA < 10 or ma < 10 or MA < ma:
            return None
        ratio = MA / max(ma, 1.0)

        # 接近圆 (长短轴比 < 1.08) 视为无倾斜
        if ratio < 1.08:
            return None

        return {
            'center': (ex, ey),
            'major': MA, 'minor': ma,
            'angle': angle,
            'ratio': ratio,
        }

    def _correct_tilt(self, image: np.ndarray, tilt: dict) -> np.ndarray:
        """
        仿射矫正: 把椭圆表盘压回正圆
        沿椭圆短轴方向拉伸, 使短轴 = 长轴
        """
        (ex, ey) = tilt['center']
        angle = tilt['angle']
        ratio = tilt['ratio']

        # 构造变换矩阵序列:
        # 1. 平移到中心
        T1 = np.float32([[1, 0, -ex], [0, 1, -ey], [0, 0, 1]])
        # 2. 旋转使长轴水平
        rad = np.deg2rad(-angle)
        R = np.float32([[np.cos(rad), -np.sin(rad), 0],
                        [np.sin(rad), np.cos(rad), 0],
                        [0, 0, 1]])
        # 3. 沿短轴(垂直)方向拉伸 ratio 倍
        S = np.float32([[1, 0, 0], [0, ratio, 0], [0, 0, 1]])
        # 4. 旋转回去
        R2 = np.float32([[np.cos(-rad), -np.sin(-rad), 0],
                         [np.sin(-rad), np.cos(-rad), 0],
                         [0, 0, 1]])
        # 5. 平移回去
        T2 = np.float32([[1, 0, ex], [0, 1, ey], [0, 0, 1]])

        M = T2 @ R2 @ S @ R @ T1
        h, w = image.shape[:2]
        return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)

    def _correct_angle_for_tilt(self, angle: float, tilt: dict) -> float:
        """
        角度补偿: 对未矫正图像的角度进行近似修正
        仅当矫正图重跑关键点失败时使用 (精度有限, 作为最后手段)
        """
        # 椭圆倾斜的近似角度补偿
        # 沿椭圆短轴方向的角度被压缩, 需要拉伸
        ratio = tilt['ratio']
        ang_ellipse = tilt['angle']
        if ratio <= 1:
            return angle

        # 转成相对椭圆长轴的夹角
        rad = np.deg2rad(angle - ang_ellipse)
        # 在椭圆坐标下, 短轴方向(垂直长轴)的分量被压缩
        x = math.sin(rad)  # 长轴方向分量
        y = math.cos(rad)  # 短轴方向分量 (被压缩)
        # 还原: y 分量乘 ratio
        corrected_rad = math.atan2(x, y * ratio)
        corrected = math.degrees(corrected_rad)
        corrected = (corrected + ang_ellipse) % 360
        return corrected

    # ============================================================
    # OCR 图像预处理（提高模糊图片识别率）
    # ============================================================

    def _preprocess_for_ocr(self, image: np.ndarray) -> List[np.ndarray]:
        """
        对图像进行多种预处理，返回多个版本供OCR尝试
        适用于不同模糊程度和光照条件
        """
        variants = []
        
        # 1. 原图
        variants.append(image)
        
        # 2. 对比度增强 (CLAHE)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))
        
        # 3. 放大2倍 (提高小数字识别率)
        h, w = image.shape[:2]
        upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        variants.append(upscaled)
        
        # 4. 锐化
        kernel_sharp = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel_sharp)
        variants.append(sharpened)
        
        return variants

    def _ocr_with_preprocessing(self, image: np.ndarray) -> List:
        """
        使用多种预处理方式尝试OCR，合并结果
        """
        all_results = []
        seen_texts = set()
        
        variants = self._preprocess_for_ocr(image)
        
        for variant in variants:
            try:
                results = self.ocr.readtext(variant)
                for box, text, conf in results:
                    # 去重
                    text_key = text.strip()
                    if text_key not in seen_texts:
                        seen_texts.add(text_key)
                        all_results.append((box, text, conf))
            except Exception as e:
                logger.debug(f"OCR预处理变体失败: {e}")
                continue
        
        return all_results

    # ============================================================
    # 端到端读数
    # ============================================================

    def read(self, image: np.ndarray) -> Optional[dict]:
        """
        端到端读数
        返回: {'reading', 'max_value', 'angle', 'ratio', 'confidence', 'reliable'}
        """
        h, w = image.shape[:2]

        # 0. 透视矫正: 检测表盘倾斜, 若明显倾斜则矫正图像
        #    (矫正后关键点+OCR 都在正圆表盘上计算, 消除透视失真)
        corrected_image = None
        tilt = self._detect_tilt(image, center=(w / 2, h / 2))
        if tilt:
            corrected_image = self._correct_tilt(image, tilt)

        # 1. 指针 + 刻度检测 (先用图像中心做粗定位)
        #    优先用矫正图 (正圆), 失败则回退原图
        work_image = corrected_image if corrected_image is not None else image
        kp = self.keypoint.extract(work_image, gauge_center=(w / 2, h / 2))
        if kp is None and corrected_image is not None:
            # 矫正图检测失败, 回退原图
            work_image = image
            kp = self.keypoint.extract(image, gauge_center=(w / 2, h / 2))
        if kp is None:
            return {'error': '指针检测失败'}
        pivot, tip = kp['pivot'], kp['tip']
        scale_points = kp.get('scale_points')
        if scale_points is None:
            return {'error': '刻度检测失败'}

        # ★ 用枢轴作为表盘实际圆心 (指针绕枢轴转, 枢轴≈圆心, 比图像中心准确)
        cx, cy = float(pivot[0]), float(pivot[1])
        left, right = scale_points
        left_a = self._angle_of(left, cx, cy)
        right_a = self._angle_of(right, cx, cy)

        # ★ 旋转校正: 若左刻度偏离标准225°太多, 说明表盘内容旋转, 摆正后重测
        #   (旋转的表盘数字方向错, OCR读不出, 摆正后能正确识别)
        rot_corrected = False
        rot = self._detect_rotation(work_image, kp)
        if rot:
            work_image = self._rotate_correct(work_image, rot, (cx, cy))
            rot_corrected = True
            # 摆正后重测关键点
            kp = self.keypoint.extract(work_image, gauge_center=(w / 2, h / 2))
            if kp is not None and kp.get('scale_points'):
                pivot, tip = kp['pivot'], kp['tip']
                scale_points = kp['scale_points']
                cx, cy = float(pivot[0]), float(pivot[1])
                left, right = scale_points
                left_a = self._angle_of(left, cx, cy)
                right_a = self._angle_of(right, cx, cy)

        # ★ 多模型投票: 环形中位数角度 (提升个别图的方向准确度)
        ptr_angle = self._angle_of(tip, cx, cy)  # 主模型角度
        if self.keypoint_ensemble:
            angles = [ptr_angle]
            for ekp in self.keypoint_ensemble:
                try:
                    ek = ekp.extract(work_image, gauge_center=(w / 2, h / 2))
                    if ek is not None and ek.get('scale_points'):
                        angles.append(self._angle_of(ek['tip'], float(ek['pivot'][0]), float(ek['pivot'][1])))
                except Exception:
                    continue
            voted = self._circular_median(angles)
            if voted is not None:
                ptr_angle = voted

        # 2. 读量程数字: 数字检测器定位框 + 框内OCR读值
        #    检测器聚焦刻度数字, 比整图OCR准 (避开序列号/文字)
        scale_radius = math.hypot(right[0] - cx, right[1] - cy)
        det_digits = self._read_digits_by_detector(work_image, (cx, cy))
        nums = [(a, 0.6, text) for a, text in det_digits]

        # 若检测器不可用或无结果, 回退整图OCR
        if not nums:
            ocr_result = self._ocr_with_preprocessing(work_image)
            for box, text, conf in ocr_result:
                if conf < 0.15 or not any(c.isdigit() for c in text):
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                bx, by = sum(xs) / 4, sum(ys) / 4
                a = self._angle_of((bx, by), cx, cy)
                radius = math.hypot(bx - cx, by - cy)
                r_ratio = radius / max(scale_radius, 1.0)
                nums.append((a, r_ratio, text))

        if not nums:
            return {'error': 'OCR 未读到数字'}

        max_value = self._infer_max_value(nums, left_a, right_a)
        if max_value is None:
            return {'error': '量程识别失败'}

        # 3. 读数: 指针比例 × 最大量程
        span = (right_a - left_a) % 360
        ratio = ((ptr_angle - left_a) % 360) / max(span, 1.0)
        ratio = max(0.0, min(1.0, ratio))
        reading = self.min_value + ratio * (max_value - self.min_value)

        # 置信度: 综合指针 + OCR
        confidence = kp.get('pointer_conf', 0.8)

        # 指针长度合理性: 针尖应在刻度半径 0.3~2.0 倍范围内
        # (放宽阈值: 避免把"方向对但长度异常"的正常图误判为不可靠)
        ptr_len = math.hypot(tip[0] - pivot[0], tip[1] - pivot[1])
        len_ok = 0.3 <= ptr_len / max(scale_radius, 1) <= 2.0

        # 投票一致性: 各模型角度差越小越可靠
        vote_ok = True
        if self.keypoint_ensemble:
            base_angle = self._angle_of(tip, cx, cy)
            spread = 0
            for ekp in self.keypoint_ensemble:
                try:
                    ek = ekp.extract(work_image, gauge_center=(w / 2, h / 2))
                    if ek is not None and ek.get('scale_points'):
                        ea = self._angle_of(ek['tip'], float(ek['pivot'][0]), float(ek['pivot'][1]))
                        spread += min(abs(base_angle - ea), 360 - abs(base_angle - ea))
                except Exception:
                    continue
            if self.keypoint_ensemble:
                spread /= len(self.keypoint_ensemble)
                vote_ok = spread < 30  # 平均差异 <30° 认为一致

        # 综合可靠性: 指针置信度高 + 长度合理 + 投票一致 + OCR读到量程
        reliable = (confidence >= 0.5 and max_value is not None
                    and len_ok and vote_ok)

        return {
            'reading': reading,
            'max_value': max_value,
            'min_value': self.min_value,
            'angle': ptr_angle,
            'ratio': ratio,
            'span': span,
            'confidence': confidence,
            'reliable': reliable,
            'len_ok': len_ok,
            'vote_ok': vote_ok,
            'pivot': pivot,
            'tip': tip,
            'scale_points': scale_points,
            'tilt_corrected': corrected_image is not None,
            'tilt_ratio': round(tilt['ratio'], 3) if tilt else None,
            'rot_corrected': rot_corrected,
        }
