"""
gauge-reader V3: 指针仪表自动读数系统
核心模块: keypoint_reader_v2 + scale_reader_v2 + reading
"""
from .keypoint_reader_v2 import KeypointReaderV2
from .scale_reader_v2 import ScaleReaderV2
from .reading import ReadingCalculator, ReadingResult
from .config import DETECTOR_PATH, KEYPOINT_PATH, load_presets

__version__ = "3.0.0"
