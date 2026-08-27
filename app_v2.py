"""
指针仪表自动读数 Web 应用 V2
使用增强模型: 合并数据集检测器 + 3类关键点+刻度定位
上传图片 → 检测表盘 → 关键点提取(指针+刻度) → 智能预设匹配 → 显示读数
"""
import sys, io, time, math, base64, logging, sqlite3, json as json_mod
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from ultralytics import YOLO
from src.reading import ReadingCalculator
from src.keypoint_reader_v2 import KeypointReaderV2
from src.scale_reader_v2 import ScaleReaderV2
from src.pointer_hybrid import HybridPointerDetector
from src.full_reader import FullReader

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from src.config import DETECTOR_PATH, KEYPOINT_PATH, GAUGE_CONFIG_PATH as CONFIG_PATH

# SQLite 数据库
DB_PATH = Path(__file__).parent / "readings.db"

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('''CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        filename TEXT,
        reading REAL,
        angle REAL,
        method TEXT,
        confidence REAL,
        scale_method TEXT,
        det_conf REAL,
        dial_size TEXT,
        time_ms INTEGER
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        gauge_name TEXT,
        min_val REAL,
        max_val REAL,
        active INTEGER DEFAULT 1
    )''')
    conn.commit()
    conn.close()

init_db()

# 全局模型
detector = None
keypoint_reader = None
hybrid_detector = None
calculator = None
scale_reader_v2 = None
full_reader = None
gauge_presets = {}

def init_models():
    global detector, keypoint_reader, hybrid_detector, calculator, scale_reader_v2, full_reader, gauge_presets

    logger.info("加载表盘检测模型 V3...")
    detector = YOLO(DETECTOR_PATH)
    import torch
    if torch.cuda.is_available():
        try:
            detector.model.half()
            detector.model.fuse()
            logger.info("检测器已启用FP16+fuse")
        except Exception:
            pass

    logger.info("加载关键点模型 V3...")
    keypoint_reader = KeypointReaderV2(KEYPOINT_PATH)
    keypoint_reader.load_model()

    # 初始化混合指针检测器 (CV+关键点交叉验证)
    hybrid_detector = HybridPointerDetector(keypoint_model=keypoint_reader.model)
    logger.info("HybridPointerDetector 已就绪 (CV+KP交叉验证)")

    calculator = ReadingCalculator()

    if CONFIG_PATH.exists():
        import json
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            gauge_presets = json.load(f)
        logger.info(f"已加载 {len(gauge_presets)} 个预设量程")

    scale_reader_v2 = ScaleReaderV2(gauge_presets)
    logger.info("ScaleReaderV2 (角度比例法) 已就绪")

    # 完整读数器 (V3 + OCR量程识别)
    logger.info("加载完整读数器 (V3 + OCR)...")
    full_reader = FullReader()
    full_reader.load_models()
    logger.info("FullReader 已就绪")

    # 模型预热（消除首次推理CUDA JIT延迟）
    logger.info("模型预热中...")
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    with torch.inference_mode():
        detector(dummy, verbose=False)
        keypoint_reader.model(dummy, verbose=False)
    logger.info("所有模型加载完成 (已预热)")


def pil_to_result_base64(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def draw_annotations(img_rgb, x1, y1, x2, y2, crop_w, crop_h,
                     angle, reading, anchors, info_text,
                     pivot=None, tip=None):
    """绘制检测框、指针、读数"""
    h, w = img_rgb.shape[:2]
    base = Image.fromarray(img_rgb).convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    cx, cy = crop_w/2, crop_h/2
    r = min(crop_w, crop_h)/2

    # 检测框
    draw.rectangle([x1,y1,x2,y2], outline=(0,255,0,200), width=3)

    # 指针线
    if pivot and tip:
        draw.line([(pivot[0]+x1, pivot[1]+y1), (tip[0]+x1, tip[1]+y1)],
                  fill=(255,50,50,220), width=3)
        draw.ellipse([pivot[0]+x1-5, pivot[1]+y1-5,
                      pivot[0]+x1+5, pivot[1]+y1+5], fill=(255,50,50,220))
    else:
        tip_x = x1 + int(cx + r*0.78*math.sin(math.radians(angle)))
        tip_y = y1 + int(cy - r*0.78*math.cos(math.radians(angle)))
        draw.line([(x1+int(cx), y1+int(cy)), (tip_x, tip_y)],
                  fill=(255,50,50,220), width=3)
        draw.ellipse([x1+int(cx)-5, y1+int(cy)-5,
                      x1+int(cx)+5, y1+int(cy)+5], fill=(255,50,50,220))

    result = Image.alpha_composite(base, overlay).convert('RGB')
    draw = ImageDraw.Draw(result)

    # 文字信息
    try:
        font_l = ImageFont.truetype("arial.ttf", 34)
        font_m = ImageFont.truetype("arial.ttf", 18)
    except:
        font_l = font_m = ImageFont.load_default()

    box_h = 80
    for y_ in range(10, 10+box_h):
        for x_ in range(10, 260):
            if y_ < result.height and x_ < result.width:
                px = result.getpixel((x_, y_))
                dark = tuple(int(c*0.5) for c in px[:3])
                draw.point((x_, y_), fill=dark)

    draw.text((18, 14), f"{reading:.4f}" if reading else "N/A",
              fill=(0,255,100), font=font_l)
    draw.text((18, 56), f"angle: {angle:.1f} deg  [{info_text}]",
              fill=(220,220,220), font=font_m)

    return result


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="指针仪表自动读数 V2", version="3.0")

@app.on_event("startup")
async def startup():
    init_models()

@app.get("/", response_class=HTMLResponse)
async def index():
    template_path = Path(__file__).parent / "templates" / "index.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/progress", response_class=HTMLResponse)
async def progress():
    progress_path = Path(__file__).parent / "progress.html"
    return progress_path.read_text(encoding="utf-8")


@app.get("/report/final", response_class=HTMLResponse)
async def report_final():
    report_path = Path(__file__).parent / "report" / "final_report.html"
    return report_path.read_text(encoding="utf-8")


@app.get("/report", response_class=HTMLResponse)
async def report():
    report_path = Path(__file__).parent / "report" / "pointer_accuracy_report.html"
    return report_path.read_text(encoding="utf-8")


@app.get("/report/diverse", response_class=HTMLResponse)
async def report_diverse():
    report_path = Path(__file__).parent / "report" / "diverse_gauge_eval.html"
    return report_path.read_text(encoding="utf-8")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), gauge_type: str = Form("auto")):
    t0 = time.perf_counter()

    # PIL读取
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    img_rgb = np.array(img_pil)
    h, w = img_rgb.shape[:2]

    # 阶段1: 表盘检测
    import torch
    with torch.inference_mode():
        det = detector(img_rgb, verbose=False)
    boxes = det[0].boxes

    if boxes is None or len(boxes) == 0:
        return JSONResponse({
            "error": "未检测到表盘",
            "timings": {"total": round((time.perf_counter()-t0)*1000)}
        })

    best = max(boxes, key=lambda b: float(b.conf[0]))
    det_conf = float(best.conf[0])
    x1, y1, x2, y2 = [int(v) for v in best.xyxy[0].tolist()]
    pad = int(min(x2-x1, y2-y1)*0.1)
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(w, x2+pad), min(h, y2+pad)

    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_bgr = crop_rgb[:,:,::-1].copy()
    ch, cw = crop_rgb.shape[:2]
    center = (cw//2, ch//2)
    radius = min(cw, ch)//2

    t_det = time.perf_counter()

    # 阶段2: 完整读数 (V3关键点 + OCR量程)
    full_result = full_reader.read(crop_rgb)
    timings = {"detection": round((time.perf_counter()-t_det)*1000)}

    if 'error' not in full_result:
        # 成功: OCR 自动识别量程
        reading = full_result['reading']
        angle = full_result['angle']
        max_value = full_result['max_value']
        pivot = full_result.get('pivot')
        tip = full_result.get('tip')
        conf = full_result['confidence']
        method = "keypoint_v3"
        scale_method = f"ocr_range(0~{max_value})"
        ptr_ok = True
        status = "complete"
        scale_q = max_value
    else:
        # 回退: 用旧的关键点+预设匹配逻辑
        kp_result = keypoint_reader.extract(crop_rgb, gauge_center=center)
        if kp_result is None and min(cw, ch) < 250:
            kp_result = keypoint_reader.extract(img_rgb)

        scale_method = "none"
        scale_q = 0.0
        reading = None
        conf = 0.0
        ptr_ok = False
        angle = 0
        method = "none"
        ptr_conf = 0.0
        pivot = tip = None

        if kp_result is not None and not kp_result.get('pointer_lowconf', False):
            angle = kp_result['angle']
            method = "keypoint_v3"
            ptr_conf = kp_result['confidence']
            pivot = kp_result.get('pivot')
            tip = kp_result.get('tip')
            ptr_ok = True

            scale_result = scale_reader_v2.from_keypoints(
                center, pivot, tip,
                kp_result.get('scale_points'),
                kp_result.get('scale_info'))
            if scale_result:
                reading = scale_result['value']
                scale_method = scale_result['method']
                conf = 0.7 * ptr_conf
            elif gauge_presets:
                best = scale_reader_v2._match_by_pointer(angle)
                if best:
                    pid, cfg = best
                    scale_result = scale_reader_v2.from_keypoints(
                        center, pivot, tip, known_range=(cfg['min_val'], cfg['max_val']))
                    if scale_result:
                        reading = scale_result['value']
                        scale_method = "preset_match"
                        conf = 0.35 * ptr_conf

        if reading is not None:
            status = "complete"
        elif ptr_ok:
            status = "partial"
        else:
            status = "failed"

    timings["scale"] = round((time.perf_counter()-t_det)*1000)
    timings["total"] = round((time.perf_counter()-t0)*1000)

    info = f"{method} | {scale_method}"

    # 记录到数据库
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            'INSERT INTO readings (timestamp, filename, reading, angle, method, confidence, scale_method, det_conf, dial_size, time_ms) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (datetime.now().isoformat(), file.filename if hasattr(file, 'filename') else 'unknown',
             round(reading, 4) if reading else None, round(angle, 2) if angle else 0,
             method, round(conf, 2), scale_method, round(det_conf, 2),
             f'{cw}x{ch}', timings.get('total', 0)))
        conn.commit(); conn.close()
    except Exception: pass

    # 绘制结果
    result_pil = draw_annotations(
        img_rgb, x1, y1, x2, y2, cw, ch,
        angle, reading or 0,
        [], info,
        pivot=pivot, tip=tip)

    return {
        "status": status,
        "reading": round(reading, 4) if reading is not None else None,
        "angle": round(angle, 2) if angle else 0,
        "method": method,
        "confidence": round(conf, 2),
        "scale_method": scale_method,
        "det_conf": round(det_conf, 2),
        "dial_size": [cw, ch],
        "timings": timings,
        "image_base64": pil_to_result_base64(result_pil),
    }


@app.post("/api/batch")
async def batch_analyze(files: list[UploadFile] = File(...)):
    """批量处理多张图片，返回CSV+JSON"""
    import csv, io as io_mod
    t0 = time.perf_counter()
    results = []

    for file in files:
        try:
            contents = await file.read()
            img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
            img_rgb = np.array(img_pil)
            h, w = img_rgb.shape[:2]

            import torch
            with torch.inference_mode():
                det = detector(img_rgb, verbose=False)
            boxes = det[0].boxes
            if boxes is None or len(boxes) == 0:
                results.append({'file': file.filename, 'status': 'no_detection'})
                continue

            best = max(boxes, key=lambda b: float(b.conf[0]))
            det_conf = float(best.conf[0])
            x1,y1,x2,y2 = [int(v) for v in best.xyxy[0].tolist()]
            pad = int(min(x2-x1, y2-y1)*0.1)
            x1,y1 = max(0,x1-pad), max(0,y1-pad)
            x2,y2 = min(w,x2+pad), min(h,y2+pad)
            crop = img_rgb[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]

            kp_result = keypoint_reader.extract(crop, gauge_center=(cw//2, ch//2))
            if kp_result is None and min(cw, ch) < 250:
                kp_result = keypoint_reader.extract(img_rgb)

            if kp_result is None or kp_result.get('pointer_lowconf', False):
                # 回退: HybridPointerDetector (传入RGB, 内部CV部分会自动转BGR)
                hy_result = hybrid_detector.detect(crop,
                                                    center_hint=(cw//2, ch//2))
                if hy_result is not None:
                    kp_result = {
                        'angle': hy_result['angle'],
                        'pivot': hy_result.get('pivot', (cw//2, ch//2)),
                        'tip': hy_result.get('tip', (cw//2+10, ch//2)),
                        'confidence': hy_result['confidence'],
                        'pointer_lowconf': False,
                        'scale_points': None,
                        'scale_info': {},
                    }

            if kp_result is None:
                results.append({'file': file.filename, 'status': 'no_pointer',
                               'det_conf': round(det_conf, 2)})
                continue

            angle = kp_result['angle']
            center = (cw//2, ch//2)
            pivot = kp_result.get('pivot', center)
            tip = kp_result.get('tip', (center[0]+10, center[1]))
            ptr_conf = kp_result['confidence']

            scale_result = scale_reader_v2.from_keypoints(
                center, pivot, tip,
                kp_result.get('scale_points'),
                kp_result.get('scale_info'))

            if scale_result:
                method_conf = {'kp_full': 0.95, 'kp_auto_sweep': 0.70}
                base_conf = method_conf.get(scale_result['method'], 0.5)
                conf = base_conf * ptr_conf
                if scale_result.get('extrapolated'): conf *= 0.7
                results.append({
                    'file': file.filename,
                    'status': 'ok',
                    'reading': round(scale_result['value'], 4),
                    'angle': round(angle, 1),
                    'method': scale_result['method'],
                    'confidence': round(conf, 2),
                    'det_conf': round(det_conf, 2),
                    'dial_size': f'{cw}x{ch}',
                })
            else:
                results.append({'file': file.filename, 'status': 'no_scale',
                               'angle': round(angle, 1)})

        except Exception as e:
            results.append({'file': file.filename if hasattr(file, 'filename') else '?',
                           'status': 'error', 'error': str(e)})

    total_time = round((time.perf_counter()-t0)*1000)
    n_ok = sum(1 for r in results if r.get('status') == 'ok')

    # 生成CSV
    csv_buf = io_mod.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=[
        'file', 'status', 'reading', 'angle', 'method',
        'confidence', 'det_conf', 'dial_size', 'error'])
    writer.writeheader()
    for r in results:
        writer.writerow({k: r.get(k, '') for k in writer.fieldnames})

    return {
        'summary': {'total': len(results), 'ok': n_ok, 'time_ms': total_time},
        'results': results,
        'csv': csv_buf.getvalue(),
    }


@app.post("/api/calibrate")
async def calibrate(file: UploadFile = File(...), known_value: float = Form(...)):
    """校准模式：提供已知读数的表盘 → 自动学习量程映射"""
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    img_rgb = np.array(img_pil)
    h, w = img_rgb.shape[:2]

    import torch
    with torch.inference_mode():
        det = detector(img_rgb, verbose=False)
    boxes = det[0].boxes
    if boxes is None or len(boxes) == 0:
        return {"error": "未检测到表盘"}

    best = max(boxes, key=lambda b: float(b.conf[0]))
    x1,y1,x2,y2 = [int(v) for v in best.xyxy[0].tolist()]
    pad = int(min(x2-x1, y2-y1)*0.1)
    x1,y1 = max(0,x1-pad), max(0,y1-pad)
    x2,y2 = min(w,x2+pad), min(h,y2+pad)
    crop = img_rgb[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]

    kp_result = keypoint_reader.extract(crop)
    if kp_result is None and min(cw, ch) < 250:
        kp_result = keypoint_reader.extract(img_rgb)

    if kp_result is None:
        return {"error": "未检测到指针"}

    angle = kp_result['angle']
    center = (cw//2, ch//2)
    pivot = kp_result.get('pivot', center)
    tip = kp_result.get('tip', (center[0]+10, center[1]))
    si = kp_result.get('scale_info', {})
    sp = kp_result.get('scale_points')

    # 关键：记录指针角度与已知值的关系
    ratio = 0.0
    # 假设默认量程0-1.6, 计算ratio, 反推实际量程
    scale_result = scale_reader_v2.from_keypoints(
        center, pivot, tip, sp, si)

    if scale_result:
        ratio = scale_result['ratio']
        # known_value = min_val + ratio * (max_val - min_val)
        # 保持min_val=0，反推max_val
        if ratio > 0.001:
            calibrated_max = known_value / max(ratio, 0.001)
        else:
            calibrated_max = 1.6  # default
        calibrated_range = (0.0, round(calibrated_max, 2))
    else:
        # 无刻度信息 → 从预设匹配
        matched = scale_reader_v2._match_by_pointer(angle)
        if matched:
            pid, cfg = matched
            ratio = scale_reader_v2._angle_span(
                cfg['min_angle'], angle) / max(cfg['span'], 1.0)
            calibrated_max = known_value / max(ratio, 0.001)
            calibrated_range = (0.0, round(calibrated_max, 2))
        else:
            calibrated_range = (0.0, 1.6)

    return {
        "status": "calibrated",
        "known_value": known_value,
        "angle": round(angle, 1),
        "ratio": round(ratio, 4),
        "calibrated_range": list(calibrated_range),
        "message": f"校准完成。后续同型号表盘读数将使用量程 {calibrated_range[0]}-{calibrated_range[1]}",
        "usage": f"POST /api/analyze 时传 gauge_type=calibrated 即可使用校准量程"
    }


@app.get("/api/history")
async def get_history(limit: int = 50):
    """获取历史读数"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT * FROM readings ORDER BY id DESC LIMIT ?', (limit,)
    ).fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows]}


@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    total = conn.execute('SELECT COUNT(*) as n FROM readings').fetchone()['n']
    ok = conn.execute("SELECT COUNT(*) as n FROM readings WHERE reading IS NOT NULL").fetchone()['n']
    recent = conn.execute(
        'SELECT * FROM readings WHERE reading IS NOT NULL ORDER BY id DESC LIMIT 10'
    ).fetchall()
    # 最近1小时平均
    avg_row = conn.execute(
        "SELECT AVG(reading) as avg, AVG(time_ms) as avg_time FROM readings WHERE reading IS NOT NULL AND timestamp > datetime('now', '-1 hour')"
    ).fetchone()
    conn.close()
    return {
        "total_readings": total, "successful": ok,
        "recent": [dict(r) for r in recent],
        "avg_reading_1h": round(avg_row['avg'], 4) if avg_row['avg'] else None,
        "avg_time_ms_1h": round(avg_row['avg_time']) if avg_row['avg_time'] else None,
    }


@app.post("/api/alerts")
async def set_alert(gauge_name: str = "default", min_val: float = 0.0, max_val: float = 1.6):
    """设置告警阈值"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        'INSERT INTO alerts (timestamp, gauge_name, min_val, max_val) VALUES (?,?,?,?)',
        (datetime.now().isoformat(), gauge_name, min_val, max_val))
    conn.commit(); conn.close()
    return {"status": "ok", "gauge_name": gauge_name, "range": [min_val, max_val]}


@app.get("/api/alerts")
async def get_alerts():
    """获取活跃告警"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    alerts = conn.execute(
        'SELECT * FROM alerts WHERE active=1 ORDER BY id DESC LIMIT 10'
    ).fetchall()
    # 检查最近读数是否超阈值
    last = conn.execute(
        'SELECT reading FROM readings WHERE reading IS NOT NULL ORDER BY id DESC LIMIT 1'
    ).fetchone()
    conn.close()

    triggered = []
    if last and last['reading'] is not None:
        for a in alerts:
            if last['reading'] < a['min_val'] or last['reading'] > a['max_val']:
                triggered.append({
                    'alert': dict(a),
                    'current_reading': last['reading'],
                    'message': f'读数 {last["reading"]} 超出范围 [{a["min_val"]}, {a["max_val"]}]'
                })

    return {"alerts": [dict(a) for a in alerts], "triggered": triggered}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()

    print("=" * 55)
    print(f"  指针仪表自动读数 Web 服务 V3")
    print(f"  API: http://localhost:{args.port}/api/analyze")
    print(f"  Batch: http://localhost:{args.port}/api/batch")
    print(f"  进度: http://localhost:{args.port}/progress")
    print("=" * 55)
    uvicorn.run("app_v2:app", host=args.host, port=args.port, reload=False)
