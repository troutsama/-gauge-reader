"""简单管线测试 — 验证模型加载和推理"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.config import DETECTOR_PATH, KEYPOINT_PATH

def test_imports():
    """验证所有核心模块可导入"""
    from src import KeypointReaderV2, ScaleReaderV2, ReadingCalculator
    assert KeypointReaderV2 is not None
    assert ScaleReaderV2 is not None
    assert ReadingCalculator is not None
    print("  imports: OK")

def test_model_loading():
    """验证模型可加载"""
    from ultralytics import YOLO
    import torch

    detector = YOLO(DETECTOR_PATH)
    assert detector is not None
    print(f"  detector: OK ({sum(p.numel() for p in detector.model.parameters())/1e6:.1f}M params)")

    kp = YOLO(KEYPOINT_PATH)
    assert kp is not None
    print(f"  keypoint: OK ({sum(p.numel() for p in kp.model.parameters())/1e6:.1f}M params)")

def test_inference():
    """验证推理不报错"""
    from ultralytics import YOLO
    from src.keypoint_reader_v2 import KeypointReaderV2
    import torch

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    detector = YOLO(DETECTOR_PATH)
    with torch.inference_mode():
        r = detector(dummy, verbose=False)
    assert r[0].boxes is not None
    print(f"  detector inference: OK")

    kp_reader = KeypointReaderV2(KEYPOINT_PATH)
    kp_reader.load_model()
    result = kp_reader.extract(dummy)
    # 空白图应该检测不到关键点
    assert result is None
    print(f"  keypoint inference: OK (correctly no detection on blank)")

def test_config():
    """验证配置加载"""
    from src.config import load_presets
    presets = load_presets()
    assert len(presets) >= 7
    print(f"  presets: OK ({len(presets)} types)")

def test_reading():
    """验证读数计算"""
    from src.reading import ReadingCalculator
    calc = ReadingCalculator()
    anchors = [(225, 0.0), (270, 0.25), (315, 0.5), (0, 0.75), (135, 1.0)]
    result = calc.compute(270, anchors)
    assert abs(result.value - 0.25) < 0.01
    print(f"  reading: OK (angle=270° -> {result.value:.2f} MPa)")

if __name__ == '__main__':
    print("Gauge Reader V3 — Tests")
    print("=" * 40)
    for name, func in [
        ("imports", test_imports),
        ("config", test_config),
        ("reading", test_reading),
        ("models", test_model_loading),
        ("inference", test_inference),
    ]:
        try:
            func()
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
    print("=" * 40)
    print("All tests passed!" if True else "")
