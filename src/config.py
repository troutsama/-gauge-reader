"""集中配置 —— 模型路径、预设、参数
模型权重默认放 models/ 下, 也支持环境变量覆盖(本地训练路径)
"""
from pathlib import Path
import json
import os

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
# 模型目录 (项目内, git 不入, 用说明放置)
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================
# 模型路径 (默认项目内 models/, 可环境变量覆盖)
# ============================================================
DETECTOR_V3 = str(MODELS_DIR / "det.pt")
KEYPOINT_V3 = str(MODELS_DIR / "keypoint.pt")
SCALE_MODEL = str(MODELS_DIR / "scale.pt")
DIGIT_DET = str(MODELS_DIR / "digit_det.pt")

# 备用 (本机训练路径, 存在则优先)
_DETECTOR_LOCAL = [
    str(PROJECT_ROOT / ".." / "merged_dataset_v3" / "runs" / "train_v3_robust" / "weights" / "best.pt"),
]
_KEYPOINT_LOCAL = [
    str(PROJECT_ROOT / ".." / "指针仪表数据集" / "关键点检测(YoloV8Pose)" / "runs" / "train_v3_robust" / "weights" / "best.pt"),
]

def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return paths[0]

# ============================================================
# 默认模型 (环境变量 > 项目models > 本地训练回退)
# ============================================================
DETECTOR_PATH = os.environ.get("GAUGE_DETECTOR",
    _first_existing([DETECTOR_V3] + _DETECTOR_LOCAL))
KEYPOINT_PATH = os.environ.get("GAUGE_KEYPOINT",
    _first_existing([KEYPOINT_V3] + _KEYPOINT_LOCAL))
SCALE_PATH = os.environ.get("GAUGE_SCALE", SCALE_MODEL)
DIGIT_DET_PATH = os.environ.get("GAUGE_DIGIT_DET", DIGIT_DET)

# ============================================================
# 预设量程
# ============================================================
GAUGE_CONFIG_PATH = PROJECT_ROOT / "gauge_configs.json"

def load_presets():
    if GAUGE_CONFIG_PATH.exists():
        with open(GAUGE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# ============================================================
# 推理参数
# ============================================================
DET_CONF_THRESHOLD = 0.25
KP_CONF_THRESHOLD = 0.15
CV_REFINE_MAX_SIZE = 600  # CV精修最大像素
SMALL_CROP_THRESHOLD = 250  # 小裁剪阈值，触发全图回退
