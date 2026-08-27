"""
可视化报告 v3: 修复后的端到端读数 (枢轴圆心 + OCR量程改进)
"""
import sys, math, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.full_reader import FullReader

V3 = r"D:\揭榜挂帅\指针仪表数据集\关键点检测(YoloV8Pose)\runs\train_v3_robust\weights\best.pt"
DATA = Path(r"D:\揭榜挂帅\指针仪表数据集\关键点检测(YoloV8Pose)\data_pose")
OUT_DIR = Path(r"D:\揭榜挂帅\gauge-reader\output\viz_report3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

font = None
for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf"]:
    if Path(fp).exists():
        try: font = ImageFont.truetype(fp, 24); break
        except: pass


def main():
    reader = FullReader()
    reader.load_models()

    # 采样 80 张 (均衡随机, 展示多种量程)
    import random
    random.seed(42)
    all_img = sorted((DATA / 'images' / 'val').glob('*.jpg'))
    random.shuffle(all_img)
    val_img = all_img[:80]

    gallery = []
    stats = {'ok': 0, 'err': 0, 'unreliable': 0, 'ranges': set()}

    for ip in val_img:
        img = np.array(Image.open(ip).convert('RGB'))
        h, w = img.shape[:2]

        r = reader.read(img)
        if 'error' in r:
            stats['err'] += 1
            continue
        stats['ok'] += 1
        stats['ranges'].add(r['max_value'])
        if not r['reliable']:
            stats['unreliable'] += 1

        # 可视化
        pivot, tip = r['pivot'], r['tip']
        left, right = r['scale_points']
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)

        # 指针 (红线) - 以枢轴为圆心
        draw.line([(pivot[0], pivot[1]), (tip[0], tip[1])], fill=(255, 50, 50), width=4)
        draw.ellipse([pivot[0]-7, pivot[1]-7, pivot[0]+7, pivot[1]+7], fill=(255, 50, 50))
        # 刻度 (绿=左/零, 蓝=右/最大)
        draw.ellipse([left[0]-9, left[1]-9, left[0]+9, left[1]+9], outline=(0, 255, 0), width=4)
        draw.ellipse([right[0]-9, right[1]-9, right[0]+9, right[1]+9], outline=(0, 120, 255), width=4)

        # 不可靠标记
        reliable_color = (255, 80, 80) if not r['reliable'] else (0, 255, 0)
        draw.text((10, 10), f"量程 0~{r['max_value']}", fill=reliable_color, font=font)
        draw.text((10, 42), f"读数 {r['reading']:.2f}", fill=(255, 255, 0), font=font)
        draw.text((10, 74), f"指针 {r['angle']:.0f}° 量程角 {r['span']:.0f}°", fill=(255, 255, 255), font=font)
        if not r['reliable']:
            draw.text((10, 106), "⚠ 读数不可靠(指针异常)", fill=(255, 80, 80), font=font)

        out_path = OUT_DIR / f"{ip.stem}_viz.jpg"
        pil.save(out_path, quality=90)
        gallery.append({
            'file': f"{ip.stem}_viz.jpg", 'name': ip.stem,
            'max': r['max_value'], 'reading': r['reading'],
            'angle': r['angle'], 'span': r['span'],
            'reliable': r['reliable'],
        })

    # HTML 报告
    ranges_str = ' '.join(map(str, sorted(stats['ranges'])))
    html = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8"><title>指针仪表读数 — 投票+置信度报告</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:"Segoe UI",system-ui,sans-serif;padding:28px}
h1{color:#58a6ff;margin-bottom:4px;font-size:22px}
.sub{color:#8b949e;margin-bottom:24px;font-size:13px}
.banner{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;margin-bottom:24px;display:flex;gap:32px;flex-wrap:wrap}
.banner .item{font-size:13px}.banner .item b{color:#58a6ff;font-size:20px;display:block;margin-bottom:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}
.card img{width:100%;display:block}
.card .info{padding:10px 14px;font-size:12px;color:#8b949e}
.card .info .name{color:#c9d1d9;font-weight:600;font-size:13px}
.card .info .val{color:#3fb950;font-weight:600}
.card .info .warn{color:#ff5050;font-weight:600}
.legend{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px;font-size:12px;color:#8b949e}
.legend span{display:flex;align-items:center;gap:6px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
</style></head><body>
<h1>指针仪表读数 — 多模型投票 + 置信度标记</h1>
<p class="sub">枢轴圆心 · 环形中位数投票 · 不可靠读数红字标记</p>
<div class="banner">
  <div class="item"><b id="n"></b>成功样本</div>
  <div class="item"><b id="err"></b>失败</div>
  <div class="item"><b id="unrel"></b>不可靠标记</div>
  <div class="item"><b id="ranges"></b>量程</div>
</div>
<div class="legend">
  <span><span class="dot" style="background:#ff3232"></span> 指针</span>
  <span><span class="dot" style="background:#00ff00"></span> 零刻度</span>
  <span><span class="dot" style="background:#0078ff"></span> 最大刻度</span>
  <span><span class="dot" style="background:#ff5050"></span> ⚠ 不可靠</span>
</div>
<div class="grid">
"""
    for g in gallery:
        warn_html = '<span class="warn">⚠不可靠</span>' if not g['reliable'] else ''
        html += f'<div class="card"><img src="{g["file"]}" loading="lazy"><div class="info"><span class="name">{g["name"]}</span> · 量程0~<span class="val">{g["max"]}</span> · 读数<span class="val">{g["reading"]:.2f}</span> · 指针{g["angle"]:.0f}° {warn_html}</div></div>\n'

    html += f'</div><script>document.getElementById("n").textContent={stats["ok"]};document.getElementById("err").textContent={stats["err"]};document.getElementById("unrel").textContent={stats["unreliable"]};document.getElementById("ranges").textContent="{ranges_str}";</script></body></html>'

    report_path = OUT_DIR / "report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"完成: 成功{stats['ok']} 失败{stats['err']} 不可靠{stats['unreliable']}")
    print(f"量程: {sorted(stats['ranges'])}")
    print(f"报告: {report_path}")


if __name__ == '__main__':
    main()
