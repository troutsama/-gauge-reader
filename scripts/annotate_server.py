"""
浏览器标注工具 —— 4点标注: 指针(枢轴+针尖) + 刻度(零+最大)
启动: python annotate_server.py [图片目录] [输出目录]
访问 http://localhost:8765
"""
import sys, os, json, shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_file

SRC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\揭榜挂帅\仪表图片\仪表图片")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"D:\揭榜挂帅\pointer_annotation\manual_keypoints")
TRASH_DIR = OUT_DIR.parent / "deleted"
for d in [OUT_DIR, TRASH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
state = {"imgs": [], "idx": 0, "done": [], "deleted": []}


def load_progress():
    progress_file = OUT_DIR / "progress.json"
    deleted_file = OUT_DIR / "deleted.json"
    if progress_file.exists():
        state["done"] = json.load(open(progress_file))
    if deleted_file.exists():
        state["deleted"] = json.load(open(deleted_file))
    state["imgs"] = sorted([f.name for f in SRC_DIR.glob("*.jpg")])


def save_progress():
    json.dump(state["done"], open(OUT_DIR / "progress.json", "w"))
    json.dump(state["deleted"], open(OUT_DIR / "deleted.json", "w"))


@app.route("/")
def index():
    return '''
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>指针+刻度标注</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:"Segoe UI",sans-serif;display:flex;flex-direction:column;align-items:center;padding:16px}
h1{color:#00e676;font-size:18px;margin-bottom:4px}
#status{color:#888;font-size:13px;margin-bottom:10px}
#container{position:relative;display:inline-block;cursor:crosshair;border:2px solid #0f3460;border-radius:4px;max-width:95vw;max-height:70vh}
#container img{display:block;max-width:95vw;max-height:70vh}
#overlay{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
#info{font-size:14px;color:#ffab40;margin:8px 0;font-weight:bold}
.btns{display:flex;gap:10px;margin-top:8px;flex-wrap:wrap;justify-content:center}
.btn{padding:10px 24px;border:none;border-radius:4px;font-size:14px;cursor:pointer;font-weight:bold}
.btn-save{background:#1b5e20;color:#69f0ae}.btn-skip{background:#444;color:#ccc}.btn-del{background:#b71c1c;color:#ef9a9a}
.btn:hover{opacity:0.85}
#counter{font-size:13px;color:#aaa;margin-top:4px}
.key{display:inline-block;padding:2px 8px;border-radius:3px;background:#333;font-family:monospace;margin:0 2px}
.point-tip{font-size:12px;color:#666;margin-top:6px}
</style></head><body>
<h1>指针+刻度标注 (4点)</h1>
<div id="status">依次点击: <span style="color:#ff3232">1枢轴</span> <span style="color:#3232ff">2针尖</span> <span style="color:#00ff00">3零刻度</span> <span style="color:#ffff00">4最大刻度</span></div>
<div id="counter"></div>
<div id="container"><img id="img" src=""><canvas id="overlay"></canvas></div>
<div id="info">点击标第 1 点</div>
<div class="btns">
  <button class="btn btn-save" onclick="save()">保存 (Enter)</button>
  <button class="btn btn-skip" onclick="skip()">跳过 (Esc)</button>
  <button class="btn btn-del" onclick="del()">删除 (D)</button>
</div>
<div style="margin-top:12px;font-size:12px;color:#666">
  <span class="key">Enter</span>保存 <span class="key">Esc</span>跳过 <span class="key">D</span>删除 <span class="key">右键</span>撤销
</div>
<script>
let imgW=0, imgH=0, points=[], fname="", total=0, current=0, doneN=0, delN=0;
const COLORS=['#ff3232','#3232ff','#00ff00','#ffff00'];
const NAMES=['枢轴','针尖','零刻度','最大刻度'];
async function load(){
  const r=await fetch("/state"); const s=await r.json();
  total=s.total; current=s.current; doneN=s.done_n; delN=s.deleted_n; fname=s.file;
  if(!fname){document.getElementById("img").src="";document.getElementById("info").innerHTML="<b style='color:#00e676'>全部完成!</b> 已标注:"+doneN+" 已删除:"+delN;return;}
  document.getElementById("img").src="/img/"+fname;
  document.getElementById("counter").textContent="["+(current+1)+"/"+total+"] "+fname+" | 已标注:"+doneN+" 已删除:"+delN;
  points=[]; document.getElementById("info").textContent="点击标第 1 点(枢轴)"; draw();
}
document.getElementById("img").onload=function(){
  imgW=this.naturalWidth; imgH=this.naturalHeight;
  const c=document.getElementById("overlay"); c.width=this.clientWidth; c.height=this.clientHeight;
  c.style.width=this.clientWidth+"px"; c.style.height=this.clientHeight+"px"; draw();
};
function canvasPos(e){const c=document.getElementById("overlay");const r=c.getBoundingClientRect();return{x:e.clientX-r.left,y:e.clientY-r.top};}
document.getElementById("container").addEventListener("click",function(e){
  if(points.length>=4)return;
  points.push(canvasPos(e));
  document.getElementById("info").textContent=points.length<4?("点击标第 "+(points.length+1)+" 点("+NAMES[points.length]+")"):"✓ 4点标完，可保存";
  draw();
});
document.getElementById("container").addEventListener("contextmenu",function(e){
  e.preventDefault(); points.pop();
  document.getElementById("info").textContent=points.length==0?"点击标第 1 点(枢轴)":("点击标第 "+(points.length+1)+" 点("+NAMES[points.length]+")");
  draw();
});
function draw(){
  const c=document.getElementById("overlay"); const ctx=c.getContext("2d"); ctx.clearRect(0,0,c.width,c.height);
  for(let i=0;i<points.length;i++){
    ctx.beginPath(); ctx.arc(points[i].x,points[i].y,7,0,Math.PI*2);
    ctx.fillStyle=COLORS[i]; ctx.fill(); ctx.strokeStyle="#fff"; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle=COLORS[i]; ctx.font="12px monospace"; ctx.fillText(NAMES[i],points[i].x+12,points[i].y-6);
  }
  if(points.length>=2){ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);ctx.lineTo(points[1].x,points[1].y);ctx.strokeStyle="#fff";ctx.lineWidth=1;ctx.stroke();}
}
async function save(){
  if(points.length<4){alert("请标完 4 个点");return;}
  const sX=imgW/document.getElementById("overlay").clientWidth;
  const sY=imgH/document.getElementById("overlay").clientHeight;
  const px=points.map(p=>[Math.round(p.x*sX),Math.round(p.y*sY)]);
  await fetch("/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({file:fname,points:px,img_w:imgW,img_h:imgH})});
  load();
}
async function skip(){await fetch("/skip/"+fname);load();}
async function del(){if(!confirm("删除这张图片？"))return;await fetch("/delete/"+fname);load();}
document.addEventListener("keydown",function(e){
  if(e.key=="Enter"){e.preventDefault();save();}
  else if(e.key=="Escape"||e.key=="s"){e.preventDefault();skip();}
  else if(e.key=="d"){e.preventDefault();del();}
});
load();
</script></body></html>'''


@app.route("/state")
def get_state():
    imgs = state["imgs"]
    while state["idx"] < len(imgs) and (imgs[state["idx"]] in state["done"] or imgs[state["idx"]] in state["deleted"]):
        state["idx"] += 1
    if state["idx"] >= len(imgs):
        return jsonify({"file": None, "total": len(imgs), "current": state["idx"],
                        "done_n": len(state["done"]), "deleted_n": len(state["deleted"])})
    return jsonify({"file": imgs[state["idx"]], "total": len(imgs), "current": state["idx"],
                    "done_n": len(state["done"]), "deleted_n": len(state["deleted"])})


@app.route("/img/<fname>")
def serve_img(fname):
    return send_file(str(SRC_DIR / fname), mimetype="image/jpeg")


def make_pointer_label(pivot, tip, w, h):
    """指针关键点标签 (pivot + tip)"""
    xmin, xmax = min(pivot[0], tip[0]), max(pivot[0], tip[0])
    ymin, ymax = min(pivot[1], tip[1]), max(pivot[1], tip[1])
    pad = 20
    xmin, ymin = max(0, xmin - pad), max(0, ymin - pad)
    xmax, ymax = min(w, xmax + pad), min(h, ymax + pad)
    bcx, bcy = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
    bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
    return f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f} {pivot[0]/w:.6f} {pivot[1]/h:.6f} 2 {tip[0]/w:.6f} {tip[1]/h:.6f} 2\n"


def make_scale_label(min_pt, max_pt, w, h):
    """刻度检测标签 (min + max)"""
    lines = []
    for cls, pt in [(0, min_pt), (1, max_pt)]:
        diag = (w * w + h * h) ** 0.5
        box = int(diag * 0.06)
        lines.append(f"{cls} {pt[0]/w:.6f} {pt[1]/h:.6f} {max(box/w, 0.01):.6f} {max(box/h, 0.01):.6f}")
    return "\n".join(lines) + "\n"


@app.route("/save", methods=["POST"])
def save():
    data = request.json
    fname = data["file"]
    pts = data["points"]  # [pivot, tip, min_scale, max_scale]
    w, h = data["img_w"], data["img_h"]

    pivot, tip, min_pt, max_pt = pts[0], pts[1], pts[2], pts[3]

    # 保存指针标签
    ptr_dir = OUT_DIR / "pointer"
    ptr_dir.mkdir(exist_ok=True)
    with open(ptr_dir / f"{Path(fname).stem}.txt", "w") as f:
        f.write(make_pointer_label(pivot, tip, w, h))

    # 保存刻度标签
    scale_dir = OUT_DIR / "scale"
    scale_dir.mkdir(exist_ok=True)
    with open(scale_dir / f"{Path(fname).stem}.txt", "w") as f:
        f.write(make_scale_label(min_pt, max_pt, w, h))

    state["done"].append(fname)
    state["idx"] += 1
    save_progress()
    print(f"  ✓ {fname}")
    return jsonify({"ok": True})


@app.route("/skip/<fname>")
def skip(fname):
    state["idx"] += 1
    return jsonify({"ok": True})


@app.route("/delete/<fname>")
def delete_img(fname):
    src = SRC_DIR / fname
    if src.exists():
        shutil.move(str(src), str(TRASH_DIR / fname))
    state["deleted"].append(fname)
    save_progress()
    print(f"  🗑 {fname}")
    return jsonify({"ok": True})


if __name__ == "__main__":
    load_progress()
    remaining = len([f for f in state["imgs"] if f not in state["done"] and f not in state["deleted"]])
    print(f"待标注: {remaining} 张 (已标注 {len(state['done'])}, 已删除 {len(state['deleted'])})")
    print(f"输出: {OUT_DIR}/pointer (指针) + {OUT_DIR}/scale (刻度)")
    print(f"\n打开浏览器: http://localhost:8765")
    app.run(host="0.0.0.0", port=8765, debug=False)
