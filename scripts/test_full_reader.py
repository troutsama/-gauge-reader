"""
测试完整三模型读数管线
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from PIL import Image

from src.full_reader import FullReader
from src.config import load_presets


def main():
    print("加载三模型...")
    presets = load_presets()
    reader = FullReader(presets=presets, default_range=(0.0, 1.6))
    reader.load_models()
    print("模型加载完成\n")

    val_dir = Path(r"D:\揭榜挂帅\keypoint_dataset\images\val")
    img_paths = sorted(val_dir.glob('*.jpg'))[:15]

    methods = {}
    results = []
    for img_path in img_paths:
        img = np.array(Image.open(img_path).convert('RGB'))
        r = reader.read(img, unit="MPa")

        if 'error' in r:
            print(f'{img_path.name}: ❌ {r["error"]}')
            methods['error'] = methods.get('error', 0) + 1
            continue

        method = r['method']
        methods[method] = methods.get(method, 0) + 1
        span_str = f'{r["span"]:.0f}°' if r['span'] else 'N/A'
        print(f'{img_path.name}: 读数={r["reading"]:.3f} {r["unit"]}  角度={r["angle"]:.1f}° 量程={span_str} [{method}]')
        results.append(r)

    print(f'\n{"="*50}')
    print(f'成功: {len(results)}/{len(img_paths)}')
    print(f'方法分布: {methods}')
    if results:
        readings = [r['reading'] for r in results]
        print(f'读数范围: [{min(readings):.3f}, {max(readings):.3f}]')
        print(f'量程检测成功: {sum(1 for r in results if r["method"]=="scale_det")}/{len(results)}')


if __name__ == '__main__':
    main()
