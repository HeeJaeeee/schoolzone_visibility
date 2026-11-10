"""Detector, blind-spot zone geometry, traffic-light colour, annotation and photo export."""
import csv
import hashlib
import json
import math
import os
from io import BytesIO
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"demo" if (ROOT/"demo").is_dir() else ROOT.parent/"docs/research/schoolzone_visibility_2026-10-07"  # demo/ = portable package
DEMO = DATA/"demo_photos"
LABELS = {"candidate": "사각지대 안에 차량 있음", "no_candidate": "사각지대 차량 없음", "region_required": "사각지대 구역을 먼저 지정하세요", "waiting": "신호대기 차로 차량만 있음"}
LIMIT = "사진 한 장은 차량 위치만 판정합니다. 정차 시간과 알림은 ▶ 영상 감시에서 계산합니다."


def polygon(region):
    """Zone as [[x,y],...] in 0..1 image coordinates. Accepts old [x1,y1,x2,y2] rectangles."""
    if region is not None and len(region)==4 and all(isinstance(v,(int,float)) for v in region):
        x1,y1,x2,y2 = region
        if x2<=x1 or y2<=y1:
            raise ValueError("구역의 넓이가 0입니다.")
        region = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    points = [[float(x),float(y)] for x,y in region]
    if len(points)<3 or not all(math.isfinite(v) and 0<=v<=1 for p in points for v in p):
        raise ValueError("구역 좌표가 올바르지 않습니다. 꼭짓점 3개 이상, 0~1 범위가 필요합니다.")
    if abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1])))<1e-6:
        raise ValueError("구역의 넓이가 0입니다.")
    return points


def inside(point,points):
    """Ray casting point-in-polygon."""
    x,y = point
    hit = False
    for (x1,y1),(x2,y2) in zip(points,points[1:]+points[:1]):
        if (y1>y)!=(y2>y) and x<(x2-x1)*(y-y1)/(y2-y1)+x1:
            hit = not hit
    return hit


def ground_point(box):
    """Where the vehicle touches the road: bottom-centre of the box (kept just inside a zone drawn to the frame edge)."""
    return ((box[0]+box[2])/2,min(box[3],.999))


def assess(detections, region, waiting_region=None):
    """Zone of each vehicle by its ground point. The signal-waiting lane wins where the two zones overlap."""
    waiting = polygon(waiting_region) if waiting_region is not None else None
    if region is None:
        return {"status":"region_required","candidate_count":0,"zones":[],"waiting_count":0,"waiting_configured":waiting is not None}
    zone_points = polygon(region)
    zones = []
    for detection in detections:
        point = ground_point(detection["box"])
        zone = "outside" if detection["kind"]!="vehicle" else "waiting" if waiting and inside(point,waiting) else "entry" if inside(point,zone_points) else "outside"
        zones.append({"zone":zone,"candidate":zone=="entry","point":list(point)})
    count = sum(z["candidate"] for z in zones)
    waiting_count = sum(z["zone"]=="waiting" for z in zones)
    return {"status":"candidate" if count else "waiting" if waiting_count else "no_candidate","candidate_count":count,"zones":zones,
            "waiting_count":waiting_count,"waiting_configured":waiting is not None}


def signal_state(image, roi):
    """Traffic-light colour inside a user-drawn box by HSV: 'red' / 'green' / 'unknown'.
    A lit lamp is saturated and bright, so only those pixels vote; hue decides the colour."""
    if roi is None:
        return None
    import cv2
    width,height = image.size
    x1,y1,x2,y2 = roi
    crop = np.asarray(image.convert("RGB").crop((int(x1*width),int(y1*height),max(int(x2*width),int(x1*width)+1),max(int(y2*height),int(y1*height)+1))))
    hsv = cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    h,s,v = hsv[...,0],hsv[...,1],hsv[...,2]
    lit = (s>=90)&(v>=150)
    red = int((lit&((h<=10)|(h>=165))).sum())
    green = int((lit&(h>=45)&(h<=100)).sum())  # Korean lights are blue-green
    if max(red,green) < max(3,.01*h.size):
        return "unknown"
    return "red" if red>green else "green"


def canonical_detection(cls, confidence, box, width, height, mapping=None):
    mapping = mapping or {"0":0,"2":1,"5":1,"7":1}
    kind = mapping.get(str(int(cls)))
    if kind is None:
        return None
    return {"kind":"person" if kind==0 else "vehicle", "model_class":int(cls), "confidence":float(confidence), "box":[max(0,min(1,float(v)/size)) for v,size in zip(box,(width,height,width,height))]}


def validate_model_names(choice,names):
    for cls,label in choice.get("expected_model_names",{}).items():
        if names.get(int(cls)) != label:
            raise ValueError("모델의 클래스 구성이 검수된 설정과 다릅니다.")
    if not choice["model_class_to_canonical"] or any(kind not in (0,1) for kind in choice["model_class_to_canonical"].values()):
        raise ValueError("사람과 차량 클래스 연결 설정이 올바르지 않습니다.")


class Detector:
    def __init__(self,choice=None):
        self.model = None
        self.choice = choice or json.loads((ROOT/"model_choice.json").read_text(encoding="utf-8"))

    def predict(self, path, snapshot=None):
        if self.model is None:
            weights = ROOT/self.choice["selected_weights"]
            if hashlib.sha256(weights.read_bytes()).hexdigest() != self.choice["weights_sha256"]:
                raise ValueError("검수된 모델 파일과 다릅니다.")
            if weights.suffix==".onnx":
                candidate = OnnxYolo(weights)
            else:
                (ROOT/"runtime").mkdir(exist_ok=True)
                os.environ["YOLO_CONFIG_DIR"] = str(ROOT/"runtime")
                import torch
                from ultralytics import YOLO
                self.device = 0 if torch.cuda.is_available() else "cpu"
                candidate = YOLO(str(weights))
            validate_model_names(self.choice,candidate.names)
            self.model = candidate
        if snapshot is None:
            with Image.open(path) as source:
                snapshot = source.convert("RGB")
        mapping = self.choice["model_class_to_canonical"]
        classes = [int(cls) for cls in mapping]
        if isinstance(self.model,OnnxYolo):
            rows = self.model.predict(snapshot,self.choice["confidence"],classes)
        else:
            boxes = self.model.predict(source=snapshot, conf=self.choice["confidence"], imgsz=self.choice["image_size"], device=self.device, classes=classes, verbose=False)[0].boxes
            rows = zip(boxes.xyxy.tolist(),boxes.conf.tolist(),boxes.cls.tolist())
        width, height = snapshot.size
        found = sorted((canonical_detection(int(cls),score,box,width,height,mapping=mapping) for box,score,cls in rows),key=lambda d:-d["confidence"])
        return dedupe(found)


class OnnxYolo:
    """Torch-free inference for the exe: ultralytics ONNX export with NMS built in, output (1,300,[x1,y1,x2,y2,score,cls])."""
    def __init__(self, weights):
        import ast
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.enable_mem_pattern = False  # required by DirectML
        providers = [p for p in ("DmlExecutionProvider","CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(weights),options,providers=providers)
        self.names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map["names"])
        self.input = self.session.get_inputs()[0]

    def predict(self, image, conf, classes):
        # ponytail: fixed 736x1280 input = ultralytics' own letterbox for 16:9 video; other aspect ratios get scaled to fit it
        import cv2
        _,_,H,W = self.input.shape
        w,h = image.size
        r = min(H/h,W/w)
        nw,nh = round(w*r),round(h*r)
        left,top = round((W-nw)/2-.1),round((H-nh)/2-.1)
        pixels = np.asarray(image)
        if (nw,nh)!=(w,h):
            pixels = cv2.resize(pixels,(nw,nh),interpolation=cv2.INTER_LINEAR)
        canvas = np.full((H,W,3),114,np.uint8)
        canvas[top:top+nh,left:left+nw] = pixels
        out = self.session.run(None,{self.input.name:canvas.transpose(2,0,1)[None].astype(np.float32)/255})[0][0]
        out = out[(out[:,4]>=conf)&np.isin(out[:,5],classes)]
        boxes = ((out[:,:4]-[left,top,left,top])/r).clip(0,[w,h,w,h])
        return zip(boxes.tolist(),out[:,4].tolist(),out[:,5].tolist())


def dedupe(detections, threshold=.7):
    """car/bus/truck all map to 'vehicle'; the model's per-class output can put two boxes on one vehicle."""
    from temporal import box_iou
    kept = []
    for d in detections:
        if not any(k["kind"]==d["kind"] and box_iou(k["box"],d["box"])>threshold for k in kept):
            kept.append(d)
    return kept


ZONE_COLOR = {"entry":"#ff705e","waiting":"#ab8cde","outside":"#71b9e5"}


def annotated(image, region, detections, waiting_region=None, tracks=None, signal_roi=None, signal=None, banner=None):
    """Draw zones and detections. With `tracks` (from StopMonitor) vehicles carry id, stop timer and alert state."""
    image = image.convert("RGB").copy()
    width,height = image.size
    draw = ImageDraw.Draw(image,"RGBA")
    size = max(16, width//85)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size)
    line = max(3,width//550)
    def xy(points):
        return [(x*width,y*height) for x,y in points]
    for zone,color,fill,text in ((region,"#20d9c2",(32,217,194,40),"사각지대 구역"),(waiting_region,"#ab8cde",(171,140,222,40),"신호대기 차로")):
        if zone:
            points = xy(polygon(zone))
            draw.polygon(points,outline=color,fill=fill,width=line)
            draw.text(points[0],text,font=font,fill=color,stroke_width=2,stroke_fill="#152a36")
    if signal_roi:
        box = (signal_roi[0]*width,signal_roi[1]*height,signal_roi[2]*width,signal_roi[3]*height)
        color = {"red":"#ff4040","green":"#30e090"}.get(signal,"#cccccc")
        draw.rectangle(box,outline=color,width=line)
        draw.text((box[0],box[3]+4),{"red":"빨간불","green":"초록불"}.get(signal,"신호 판독 불가"),font=font,fill=color,stroke_width=2,stroke_fill="#152a36")
    if tracks is None:
        analysis = assess(detections,region,waiting_region)
        tracks = [{"box":d["box"],"zone":z["zone"]} for d,z in zip(detections,analysis["zones"]) if d["kind"]=="vehicle"] if region else []
    people = [d for d in detections if d["kind"]=="person"]
    for item in people:
        box = (item["box"][0]*width,item["box"][1]*height,item["box"][2]*width,item["box"][3]*height)
        draw.rectangle(box,outline="#e6d287",width=max(2,width//750))
    for track in tracks:
        box = (track["box"][0]*width,track["box"][1]*height,track["box"][2]*width,track["box"][3]*height)
        color = "#ff2d2d" if track.get("alert") else ZONE_COLOR[track["zone"]]
        draw.rectangle(box,outline=color,width=line*2 if track.get("alert") else max(2,width//750))
        label = f'#{track["id"]} ' if "id" in track else ""
        label += {"entry":"사각지대","waiting":"신호대기","outside":"차량"}[track["zone"]]
        if track.get("stopped",0)>=1:
            label += f' 정차 {track["stopped"]:.0f}초'
        if track["zone"]=="entry" and track.get("dwell",0)>=1:
            label += f'/{track["dwell"]:.0f}초'
        draw.text((box[0],max(0,box[1]-size*1.3)),label,font=font,fill=color,stroke_width=2,stroke_fill="#152a36")
    if banner:  # bottom edge: the top of a CCTV frame is where far vehicles are
        draw.rectangle((0,height-size*2.4,width,height),fill=(200,20,20,220))
        draw.text((size,height-size*1.9),banner,font=font,fill="white")
    return image


def export_result(path, region, detections, metadata, folder, expected_source_sha=None, model_choice=None, waiting_region=None):
    analysis = assess(detections,region,waiting_region)
    if region is None:
        raise ValueError("관심구역 지정 후 분석 결과를 저장하세요.")
    path,folder = Path(path),Path(folder)
    source_bytes = path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("사진 파일이 변경되었습니다. 사진을 다시 열고 분석하세요.")
    folder.mkdir(parents=True,exist_ok=True)
    record_id = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid4().hex[:8]
    stem = folder/record_id
    choice = model_choice or json.loads((ROOT/"model_choice.json").read_text(encoding="utf-8"))
    with Image.open(BytesIO(source_bytes)) as image:
        size = list(image.size)
        output_image = annotated(image,region,detections,waiting_region)
    data = {"record_id":record_id, "source":str(path.resolve()), "source_sha256":source_sha, "image_size":size,
            "model":choice["selected_model"], "model_sha256":choice["weights_sha256"], "confidence":choice["confidence"], "image_size_for_inference":choice["image_size"],
            "region":region,"waiting_region":waiting_region, "rule":"vehicle ground point (bottom-centre of box) inside zone polygon; waiting lane wins",
            "detections":detections,"analysis":analysis,"metadata":metadata,"limitation":LIMIT}
    journal = folder/"사진_분석_기록.csv"
    exists = journal.exists()
    try:
        with journal.open("a",encoding="utf-8-sig",newline="") as file:
            output_image.save(stem.with_suffix(".png"))
            stem.with_suffix(".json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
            writer = csv.writer(file)
            if not exists:
                writer.writerow(["기록ID","사진","카메라","판정","후보차량수","구역좌표","비고"])
            writer.writerow([record_id,path.name,metadata.get("camera","직접 지정"),LABELS[analysis["status"]],analysis["candidate_count"],json.dumps(region),"사진은 위치만 판정"])
    except OSError:
        stem.with_suffix(".png").unlink(missing_ok=True)
        stem.with_suffix(".json").unlink(missing_ok=True)
        raise
    return stem.with_suffix(".png"),stem.with_suffix(".json"),journal
