"""
用手动标注的刻度数字训练 YOLOv8 数字检测器
标注: pointer_annotation/digits (class=0 number, 5含值)
"""
import sys, shutil
from pathlib import Path
import random
sys.path.insert(0, str(Path(__file__).parent.parent))

ANNOT = Path(r"D:\揭榜挂帅\pointer_annotation\digits")
SRC_IMG = Path(r"D:\揭榜挂帅\指针仪表数据集\关键点检测(YoloV8Pose)\data_pose\images\train")
DST = Path(r"D:\揭榜挂帅\digit_det_dataset")


def main():
    if DST.exists():
        shutil.rmtree(DST)
    for split in ['train', 'val']:
        (DST / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DST / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 收集标注
    labels = sorted(ANNOT.glob('*.txt'))
    random.seed(42)
    random.shuffle(labels)
    n_val = max(3, int(len(labels) * 0.2))
    val_labels = labels[:n_val]
    train_labels = labels[n_val:]

    for split, lbl_list in [('train', train_labels), ('val', val_labels)]:
        for lbl in lbl_list:
            name = lbl.stem
            img = SRC_IMG / f"{name}.jpg"
            if not img.exists():
                continue
            shutil.copy2(img, DST / 'images' / split / f"{name}.jpg")
            # 重写label: 去掉最后的值列, 保留 class cx cy w h
            lines = []
            for line in lbl.read_text(encoding='utf-8').strip().split('\n'):
                parts = line.split()
                if len(parts) >= 5:
                    lines.append(' '.join(parts[:5]))  # class cx cy w h
            (DST / 'labels' / split / f"{name}.txt").write_text('\n'.join(lines))

    train_n = len(list((DST / 'images' / 'train').glob('*.jpg')))
    val_n = len(list((DST / 'images' / 'val').glob('*.jpg')))
    print(f"数据集: train={train_n} val={val_n}")

    yaml = f"train: {DST.as_posix()}/images/train\nval: {DST.as_posix()}/images/val\n\nnc: 1\nnames: ['number']\n"
    (DST / 'data.yaml').write_text(yaml, encoding='utf-8')

    # 训练
    from ultralytics import YOLO
    model = YOLO("D:/揭榜挂帅/yolov8n.pt")
    results = model.train(
        data=str(DST / 'data.yaml'),
        epochs=150,
        imgsz=640,
        batch=8,
        device="0",
        workers=0,
        project=str(DST / 'runs'),
        name='train_digit_yolo',
        optimizer='auto', lr0=0.01, lrf=0.01, cos_lr=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=15.0, translate=0.1, scale=0.5, shear=3.0,
        mosaic=0.5, mixup=0.1, copy_paste=0.1,
        patience=60, pretrained=True, verbose=True, seed=42, amp=True,
    )
    print(f"训练完成: {results.save_dir}/weights/best.pt")


if __name__ == '__main__':
    main()
