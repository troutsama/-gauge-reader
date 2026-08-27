"""
表盘刻度数字标注工具 (浏览器)
打开表盘图, 拖拽框选每个数字, 弹出输入框填数字值(如 1.6/0.4/25)
输出: YOLO 检测格式, class=0 number, 标签含值
用法: python annotate_digits.py [图片目录] [输出目录]
访问 http://localhost:8765
"""
import sys, json, io, shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_file

SRC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\揭榜挂帅\指针仪表数据集\关键点检测(YoloV8Pose)\data_pose\images\train")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"D:\揭榜挂帅\pointer_annotation\digits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
state = {"imgs": [], "idx": 0, "done": [], "deleted": []}

def load_progress():
    p = OUT_DIR / "progress.json"
    if p.exists():
        state["done"] = json.load(open(p))
    d = OUT_DIR / "deleted.json"
    if d.exists():
        state["deleted"] = json.load(open(d))
    state["imgs"] = sorted([f.name for f in SRC_DIR.glob("*.jpg")])

def save_progress():
    json.dump(state["done"], open(OUT_DIR / "progress.json", "w"))
    json.dump(state["deleted"], open(OUT_DIR / "deleted.json", "w"))

@app.route("/")
def index():
    return '''
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>刻度数字标注</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:"Segoe UI",sans-serif;display:flex;flex-direction:column;align-items:center;padding:16px}
h1{color:#00e676;font-size:18px;margin-bottom:4px}
#status{color:#888;font-size:13px;margin-bottom:10px}
#container{position:relative;display:inline-block;cursor:crosshair;border:2px solid #0f3460;border-radius:4px;max-width:95vw;max-height:70vh;user-select:none}
#container img{display:block;max-width:95vw;max-height:70vh;pointer-events:none;user-select:none;-webkit-user-drag:none}
#overlay{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
#info{font-size:14px;color:#ffab40;margin:8px 0}
.btns{display:flex;gap:10px;margin-top:8px;flex-wrap:wrap;justify-content:center}
.btn{padding:10px 24px;border:none;border-radius:4px;font-size:14px;cursor:pointer;font-weight:bold}
.btn-save{background:#1b5e20;color:#69f0ae}.btn-skip{background:#444;color:#ccc}.btn-del{background:#b71c1c;color:#ef9a9a}
#counter{font-size:13px;color:#aaa;margin-top:4px}
.key{display:inline-block;padding:2px 8px;border-radius:3px;background:#333;font-family:monospace;margin:0 2px}
</style></head><body>
<h1>刻度数字标注</h1>
<div id="status">拖拽框选数字 → 弹窗填数字值 → 继续框下一个</div>
<div id="counter"></div>
<div id="container"><img id="img" src=""><canvas id="overlay"></canvas></div>
<div id="info">拖拽框选数字 (按住左键拉框)</div>
<div class="btns">
  <button class="btn btn-skip" onclick="prev()">◀ 上一张</button>
  <button class="btn btn-del" onclick="deleteCur()">🗑 删除这张</button>
  <button class="btn btn-del" onclick="undo()">撤销 (Z)</button>
  <button class="btn btn-save" onclick="finish()">✓ 保存 (Enter)</button>
  <button class="btn btn-skip" onclick="skip()">跳过 (Esc)</button>
</div>
<script>
let imgW=0, imgH=0, boxes=[], fname="", total=0, current=0, doneN=0;
let drawing=false, sx=0, sy=0;
async function load(){
  const r=await fetch("/state"); const s=await r.json();
  total=s.total; current=s.current; doneN=s.done_n; fname=s.file;
  if(!fname){document.getElementById("info").innerHTML="<b style='color:#00e676'>全部完成!</b>";return;}
  const img=document.getElementById("img");
  img.src="/img/"+fname;
  img.draggable=false;
  img.addEventListener('mousedown',function(e){e.preventDefault();});
  document.getElementById("counter").textContent="["+(current+1)+"/"+total+"] "+fname+" | 已标:"+doneN;
  // 读取已标注框(返回上一张时显示)
  boxes=[];
  if(fname){
    const lb=await fetch("/getlabel/"+fname);
    const lj=await lb.json();
    for(const bb of lj.boxes){
      boxes.push({x:bb.x*imgW, y:bb.y*imgH, w:bb.w*imgW, h:bb.h*imgH, text:bb.text});
    }
  }
  document.getElementById("info").textContent=boxes.length?("已标"+boxes.length+"个, 继续框或保存"):"拖拽框选数字 → 填值";
  draw();
}
document.getElementById("img").onload=function(){
  imgW=this.naturalWidth; imgH=this.naturalHeight;
  const c=document.getElementById("overlay"); c.width=this.clientWidth; c.height=this.clientHeight;
  c.style.width=this.clientWidth+"px"; c.style.height=this.clientHeight+"px"; draw();
};
function pos(e){const c=document.getElementById("overlay");const r=c.getBoundingClientRect();return{x:e.clientX-r.left,y:e.clientY-r.top};}
document.getElementById("container").addEventListener("mousedown",function(e){drawing=true;const p=pos(e);sx=p.x;sy=p.y;});
document.getElementById("container").addEventListener("mousemove",function(e){
  if(!drawing)return; const p=pos(e);
  const c=document.getElementById("overlay");const ctx=c.getContext("2d");
  ctx.clearRect(0,0,c.width,c.height);
  ctx.strokeStyle="#00ff00";ctx.lineWidth=2;
  ctx.strokeRect(Math.min(sx,p.x),Math.min(sy,p.y),Math.abs(p.x-sx),Math.abs(p.y-sy));
});
document.getElementById("container").addEventListener("mouseup",function(e){
  if(!drawing)return; drawing=false; const p=pos(e);
  const x1=Math.min(sx,p.x),y1=Math.min(sy,p.y),w=Math.abs(p.x-sx),h=Math.abs(p.y-sy);
  if(w<10||h<10)return;
  const sX=imgW/document.getElementById("overlay").clientWidth;
  const sY=imgH/document.getElementById("overlay").clientHeight;
  const val=prompt("输入该刻度数字值 (如 1.6 / 0.4 / 25):");
  if(val!==null) boxes.push({x:x1*sX,y:y1*sY,w:w*sX,h:h*sY,text:val.trim()});
  draw();
});
function draw(){
  const c=document.getElementById("overlay");const ctx=c.getContext("2d");ctx.clearRect(0,0,c.width,c.height);
  const sX=document.getElementById("overlay").clientWidth/imgW;
  const sY=document.getElementById("overlay").clientHeight/imgH;
  ctx.lineWidth=2;
  for(const b of boxes){
    ctx.strokeStyle="#00ff00"; ctx.strokeRect(b.x*sX,b.y*sY,b.w*sX,b.h*sY);
    ctx.fillStyle="#00ff00";ctx.font="14px monospace";
    ctx.fillText(b.text,b.x*sX,b.y*sY-6);
  }
}
async function skip(){boxes=[];await fetch("/skip/"+fname);load();}
function undo(){if(boxes.length)boxes.pop();draw();}
async function prev(){await fetch("/prev");fname="";load();}
async function deleteCur(){
  if(!confirm("删除这张标注？"))return;
  await fetch("/delete/"+fname);
  fname=""; boxes=[]; load();
}
async function finish(){
  if(!boxes.length){skip();return;}
  const sX=imgW/document.getElementById("overlay").clientWidth;
  const sY=imgH/document.getElementById("overlay").clientHeight;
  const pxs=boxes.map(b=>({x:b.x*sX,y:b.y*sY,w:b.w*sX,h:b.h*sY,text:b.text}));
  await fetch("/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({file:fname,boxes:pxs,img_w:imgW,img_h:imgH})});
  load();
}
document.addEventListener("keydown",function(e){
  if(e.key=="Enter"){e.preventDefault();finish();}
  else if(e.key=="Escape"){e.preventDefault();skip();}
  else if(e.key=="z"){e.preventDefault();undo();}
  else if(e.key=="ArrowLeft"){e.preventDefault();prev();}
});
load();
</script></body></html>'''

@app.route("/state")
def get_state():
    imgs = state["imgs"]
    # 若当前 idx 指向的图已标/已删, 前进跳过
    if state["idx"] < len(imgs) and (imgs[state["idx"]] in state["done"] or imgs[state["idx"]] in state["deleted"]):
        while state["idx"] < len(imgs) and (imgs[state["idx"]] in state["done"] or imgs[state["idx"]] in state["deleted"]):
            state["idx"] += 1
    done_n = len(state["done"]) - len(state["deleted"])
    if state["idx"] >= len(imgs):
        return jsonify({"file": None, "total": len(imgs), "current": state["idx"], "done_n": done_n})
    return jsonify({"file": imgs[state["idx"]], "total": len(imgs), "current": state["idx"], "done_n": done_n})

@app.route("/img/<fname>")
def serve_img(fname):
    return send_file(str(SRC_DIR / fname), mimetype="image/jpeg")

@app.route("/getlabel/<fname>")
def get_label(fname):
    """读取已标注的框(返回上一张时显示)"""
    lbl = OUT_DIR / f"{Path(fname).stem}.txt"
    if not lbl.exists():
        return jsonify({"boxes": []})
    boxes = []
    for line in lbl.read_text(encoding='utf-8').strip().split('\n'):
        parts = line.split()
        if len(parts) >= 6:
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append({"x": cx, "y": cy, "w": bw, "h": bh, "text": parts[5]})
    return jsonify({"boxes": boxes})

@app.route("/save", methods=["POST"])
def save():
    d = request.json
    fname = d["file"]
    boxes = d["boxes"]
    w, h = d["img_w"], d["img_h"]
    lines = []
    for b in boxes:
        cx = (b["x"] + b["w"]/2) / w
        cy = (b["y"] + b["h"]/2) / h
        bw = b["w"] / w
        bh = b["h"] / h
        text = b["text"]
        lines.append(f"0 {cx:.6f} {cy:.6f} {max(bw,0.02):.6f} {max(bh,0.02):.6f} {text}")
    with open(OUT_DIR / f"{Path(fname).stem}.txt", "w") as f:
        f.write("\n".join(lines))
    state["done"].append(fname)
    state["idx"] += 1
    save_progress()
    print(f"  ✓ {fname}: {len(boxes)}个数字")
    return jsonify({"ok": True})

@app.route("/skip/<fname>")
def skip(fname):
    state["idx"] += 1
    return jsonify({"ok": True})

@app.route("/prev")
def prev():
    """返回上一张: idx 精确后退一张, 显示该图(含已标的可修改)"""
    imgs = state["imgs"]
    state["idx"] = max(0, state["idx"] - 1)
    return jsonify({"ok": True, "file": imgs[state["idx"]] if state["idx"] < len(imgs) else None})

@app.route("/delete/<fname>")
def delete_img(fname):
    """删除当前标注(标坏了): 从done移除, 记入deleted, 回到上一张"""
    if fname in state["done"]:
        state["done"].remove(fname)
    if fname not in state["deleted"]:
        state["deleted"].append(fname)
    # 删除标注文件
    lbl = OUT_DIR / f"{Path(fname).stem}.txt"
    if lbl.exists():
        lbl.unlink()
    save_progress()
    # 回到上一张
    state["idx"] = max(0, state["idx"] - 1)
    print(f"  🗑 {fname}")
    return jsonify({"ok": True})

if __name__ == "__main__":
    load_progress()
    remaining = len([f for f in state["imgs"] if f not in state["done"]])
    print(f"待标注: {remaining} 张 (已标 {len(state['done'])})")
    print(f"输出: {OUT_DIR}")
    print(f"浏览器: http://localhost:8765")
    app.run(host="0.0.0.0", port=8765, debug=False)
