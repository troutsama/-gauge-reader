"""
混合指针检测：CV霍夫线(主) + 关键点模型(枢轴辅助)
CV对各种表盘都有效，关键点只用来定位枢轴
"""
import math, logging
import numpy as np, cv2

logger = logging.getLogger(__name__)


def extract_pointer_line_cv(roi):
    """纯CV指针线检测 — 参考仓库getPointerLines方案
    返回: (x1,y1,x2,y2) 或 None
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    inv = cv2.bitwise_not(eq)
    blur = cv2.medianBlur(inv, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(blur, kernel)
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
    thresh_val = max(np.mean(closed) * 1.3, 180)
    _, binary = cv2.threshold(closed, thresh_val, 255, cv2.THRESH_BINARY)

    # 大图缩小加速
    if max(binary.shape) > 600:
        scale = 500.0 / max(binary.shape)
        bw, bh = int(binary.shape[1] * scale), int(binary.shape[0] * scale)
        binary_small = cv2.resize(binary, (bw, bh))
    else:
        binary_small = binary; scale = 1.0

    # 骨架化
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

    lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180,
                            threshold=30, minLineLength=30, maxLineGap=50)
    if lines is None or len(lines) == 0:
        return None

    # 取最长线
    best_len = 0; best_line = None
    for l in lines:
        if len(l.shape) == 2: x1, y1, x2, y2 = l[0]
        else: x1, y1, x2, y2 = l
        length = math.hypot(x2 - x1, y2 - y1)
        if length > best_len:
            best_len = length; best_line = (x1, y1, x2, y2)
    return best_line


class HybridPointerDetector:
    """混合指针检测器：CV找线 + 模型找枢轴"""

    def __init__(self, keypoint_model=None):
        """
        keypoint_model: YOLOv8-Pose模型(可选)，用于找枢轴。
                       不传则用几何中心作为枢轴。
        """
        self.kp_model = keypoint_model

    def detect(self, roi: np.ndarray, center_hint=None) -> dict:
        h, w = roi.shape[:2]
        default_center = center_hint or (w / 2, h / 2)

        # Step 1: 关键点模型（主方案，精度高）
        kp_result = None
        if self.kp_model is not None:
            try:
                import torch
                with torch.inference_mode():
                    results = self.kp_model(roi, conf=0.15, verbose=False)
                r = results[0]
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    for box, kp_tensor in zip(r.boxes, r.keypoints.data):
                        if int(box.cls[0]) == 0:
                            kp = kp_tensor.cpu().numpy()
                            if kp[0][2] > 0.5 and kp[1][2] > 0.5:
                                kp0 = (kp[0][0], kp[0][1])
                                kp1 = (kp[1][0], kp[1][1])
                                d0 = math.hypot(kp0[0]-default_center[0], kp0[1]-default_center[1])
                                d1 = math.hypot(kp1[0]-default_center[0], kp1[1]-default_center[1])
                                pivot = kp0 if d0 < d1 else kp1
                                tip = kp1 if d0 < d1 else kp0
                                dx, dy = tip[0]-pivot[0], tip[1]-pivot[1]
                                angle = math.degrees(math.atan2(dx, -dy)) % 360
                                kp_result = {
                                    'angle': angle, 'pivot': pivot, 'tip': tip,
                                    'confidence': float(box.conf[0]), 'method': 'keypoint'
                                }
                                break
            except Exception:
                pass

        # Step 2: 关键点高置信度(>0.3) → 直接信任
        if kp_result and kp_result['confidence'] > 0.3:
            return kp_result

        # Step 3: 关键点低置信度或无结果 → CV回退
        # 注意: roi是RGB格式, CV需要BGR
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR) if len(roi.shape) == 3 else roi
        cv_line = extract_pointer_line_cv(roi_bgr)

        if cv_line is not None:
            # 用关键点的枢轴(如果有)，否则用几何中心
            pivot = kp_result['pivot'] if kp_result else default_center

            lx1, ly1, lx2, ly2 = cv_line
            d1 = math.hypot(lx1-pivot[0], ly1-pivot[1])
            d2 = math.hypot(lx2-pivot[0], ly2-pivot[1])
            tip = (lx1, ly1) if d1 > d2 else (lx2, ly2)

            dx, dy = tip[0]-pivot[0], tip[1]-pivot[1]
            angle = math.degrees(math.atan2(dx, -dy)) % 360
            line_len = math.hypot(lx2-lx1, ly2-ly1)
            conf = min(0.85, line_len / max(w, h))

            # CV与关键点交叉验证
            if kp_result:
                diff = min(abs(angle - kp_result['angle']),
                          360 - abs(angle - kp_result['angle']))
                if diff < 20:  # 一致 → 用更准的关键点
                    return kp_result

            return {
                'angle': angle, 'pivot': pivot, 'tip': tip,
                'line': cv_line, 'method': 'cv_fallback',
                'confidence': conf,
            }

        # Step 4: 都失败 → 返回低置信度关键点结果
        if kp_result:
            kp_result['method'] = 'kp_lowconf'
            return kp_result

        return None
