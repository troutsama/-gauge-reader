"""
自动标注表盘刻度数字 → YOLO 数字检测数据集
用 V3 定位表盘 + EasyOCR 读刻度环附近数字, 生成数字框标注
目标: 训练专用数字检测器替代 EasyOCR
"""
import sys, math, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import cv2
from PIL import Image, ImageDraw
from src.keypoint_reader_v2 import KeypointReaderV2
import torch

V3 = r"D:\揭榜挂帅\指针仪表数据集\关键点检测(YoloV8Pose)\runs\train_v3_robust\weights\best.pt"
SRC = Path(r"D:\揭榜挂帅\指针仪表数据集\关键点检测(YoloV8Pose)\data_pose")
DST = Path(r"D:\揭榜挂帅\gauge-reader\output\digit_dataset")
DST.mkdir(parents=True, exist_ok=True)


def main():
    kp = KeypointReaderV2(V3)
    kp.load_model()
    if torch.cuda.is_available():
        kp.model.model.half(); kp.model.model.fuse()

    import easyocr
    import logging
    logging.disable(logging.WARNING)
    ocr = easyocr.Reader(['en'], gpu=True, verbose=False)

    imgs = sorted((SRC / 'images' / 'train').glob('*.jpg'))[:400]
    n_img = 0
    n_ann = 0
    for ip in imgs:
        img = np.array(Image.open(ip).convert('RGB'))
        h, w = img.shape[:2]
        r = kp.extract(img, gauge_center=(w / 2, h / 2))
        if r is None or not r.get('scale_points'):
            continue
        pivot = r['pivot']
        left, right = r['scale_points']
        cx, cy = float(pivot[0]), float(pivot[1])
        la = math.degrees(math.atan2(left[0] - cx, -(left[1] - cy))) % 360
        ra = math.degrees(math.atan2(right[0] - cx, -(right[1] - cy))) % 360
        scale_r = math.hypot(right[0] - cx, right[1] - cy)
        if scale_r < 15:
            continue

        ocr_res = ocr.readtext(img)
        annots = []
        for box, text, conf in ocr_res:
            if conf < 0.4 or not any(c.isdigit() for c in text):
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            bx, by = sum(xs) / 4, sum(ys) / 4
            a = math.degrees(math.atan2(bx - cx, -(by - cy))) % 360
            rr = math.hypot(bx - cx, by - cy) / scale_r
            in_arc = (a >= la - 12 or a <= ra + 12) if ra < la else (la - 12 <= a <= ra + 12)
            if in_arc and 0.4 <= rr <= 1.0:
                x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
                # 归一化框
                annots.append((x1 / w, y1 / h, x2 / w, y2 / h, text))

        if len(annots) < 2:
            continue

        # 保存图片 + 标签 (标签用归一化 YOLO, class=0 number)
        out_img = DST / 'images' / ip.name
        out_lbl = DST / 'labels' / (ip.stem + '.txt')
        (DST / 'images').mkdir(exist_ok=True)
        (DST / 'labels').mkdir(exist_ok=True)
        shutil.copy2(ip, out_img)
        lines = []
        for x1, y1, x2, y2, text in annots:
            bw = max(x2 - x1, 0.02)
            bh = max(y2 - y1, 0.02)
            lines.append(f"0 {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {bw:.6f} {bh:.6f} {text}")
        with open(out_lbl, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        n_img += 1
        n_ann += len(annots)

    print(f"自动标注完成: {n_img} 张, {n_ann} 个数字")


if __name__ == '__main__':
    main()
