"""v4: all 34 labeled AI-Hub 169 clips, event-level split, full fine-tuning.

python train_v4.py prepare   # extract frames + YOLO labels into data_v4/, then relabel
python train_v4.py relabel   # keep person/vehicle only; vehicle -> COCO car/bus/truck (see relabel())
python train_v4.py train     # fine-tune yolo26n -> runs/v4b/weights/best.pt
"""
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/"docs/research/schoolzone_visibility_2026-10-07"
DATA = ROOT/"data_v4"
os.environ["YOLO_CONFIG_DIR"] = str(ROOT/"runtime")

# Every event shown in the GUI demo is held out, so demo results are on unseen events.
TEST = {"2020_11_13_18_30","2020_11_10_12_02","2020_11_10_12_41","2020_11_10_20_05","2020_11_09_20_22","2020_11_12_12_30"}
VAL = {"2020_11_13_19_00","2020_11_10_13_10","2020_11_10_14_11"}  # 1 night + 2 day events
CLASSES = {"person":0,"cycle":1,"vehicle":2,"motorbike":3,"traffic light":9}  # COCO indices, 'car' renamed to vehicle
STEP = {"train":2,"val":3,"test":3}  # labels are 10 fps; adjacent frames are near-duplicates
WIDTH = 1920


def videos():
    for folder in ("train1","train4","validation"):
        for video in sorted((SOURCE/f"aihub169_suddenlyappear_{folder}").glob("*.mp4")):
            event = video.name[:16]
            yield video,SOURCE/f"aihub169_suddenlyappear_{folder}_labels","test" if event in TEST else "val" if event in VAL else "train"


def extract(job):
    video,labels,split = job
    wanted = {math.ceil(1.5*(i-1)):i for i in range(1,301,STEP[split])}  # label i <-> frame ceil(1.5(i-1)) at 15 fps
    capture,frame_index,rows = cv2.VideoCapture(str(video)),0,0
    assert capture.get(cv2.CAP_PROP_FPS)==15,video
    while capture.grab():
        if frame_index in wanted:
            ok,frame = capture.retrieve()
            assert ok,video
            stem = f"{video.stem}_{wanted[frame_index]:05d}"
            root = ET.parse(labels/f"{stem}.xml").getroot()
            w,h = int(root.findtext("size/width")),int(root.findtext("size/height"))
            assert frame.shape[1::-1]==(w,h),stem
            lines = []
            for obj in root.iter("object"):
                cls = CLASSES.get(obj.findtext("name"))
                if cls is None:
                    continue
                x1,y1,x2,y2 = [min(max(int(float(obj.findtext("bndbox/"+k))),0),lim) for k,lim in (("xmin",w),("ymin",h),("xmax",w),("ymax",h))]
                if x2>x1 and y2>y1:
                    lines.append(f"{cls} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            small = cv2.resize(frame,(WIDTH,round(h*WIDTH/w)),interpolation=cv2.INTER_AREA)
            cv2.imencode(".jpg",small,[cv2.IMWRITE_JPEG_QUALITY,92])[1].tofile(str(DATA/split/"images"/f"{stem}.jpg"))
            (DATA/split/"labels"/f"{stem}.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
            rows += 1
        frame_index += 1
    capture.release()
    assert frame_index==450 and rows==len(wanted),(video,frame_index,rows)
    return split,video.name,rows


def prepare():
    for split in STEP:
        for kind in ("images","labels"):
            (DATA/split/kind).mkdir(parents=True,exist_ok=True)
    jobs = list(videos())
    events = {split:sorted({v.name[:16] for v,_,s in jobs if s==split}) for split in STEP}
    assert not (set(events["train"])&set(events["val"]) or set(events["train"])&set(events["test"]) or set(events["val"])&set(events["test"]))
    with ProcessPoolExecutor(6) as pool:
        done = list(pool.map(extract,jobs))
    for row in done:
        print(*row,flush=True)
    from ultralytics import YOLO
    names = dict(YOLO(str(ROOT.parent/"yolo26n.pt")).names)
    names[2] = "vehicle"
    (DATA/"dataset.yaml").write_text(yaml.safe_dump({"path":str(DATA).replace("\\","/"),"train":"train/images","val":"val/images","test":"test/images","names":names},allow_unicode=True,sort_keys=False),encoding="utf-8")
    counts = {split:sum(n for s,_,n in done if s==split) for split in STEP}
    (DATA/"manifest.json").write_text(json.dumps({"counts":counts,"events":events,"videos":[{"split":s,"video":v,"images":n} for s,v,n in done],
        "class_map":CLASSES,"note":"Event-level split (A/B views of one event stay together). Every GUI demo event is in test."},ensure_ascii=False,indent=2),encoding="utf-8")
    print(counts)
    relabel()


def relabel():
    """AI-Hub labels every car/bus/truck as 'vehicle'. Training all of them as COCO 'car' (2) suppresses the
    pretrained bus/truck outputs before the few training trucks can teach 'car' -> the parked dump truck was
    missed (run v4: large-vehicle recall 0.80 -> 0.70). Keep the COCO class the pretrained model assigns to
    each labeled vehicle box; the app maps 2/5/7 back to 'vehicle'. Cycle/motorbike/traffic light are labeled
    in some clips only, so they are dropped. Idempotent: re-deciding from {2,5,7} each time."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import box_iou
    model = YOLO(str(ROOT.parent/"yolo26n.pt"))
    images = sorted(DATA.glob("*/images/*.jpg"))
    changed = 0
    for start in range(0,len(images),16):
        batch = images[start:start+16]
        for path,result in zip(batch,model.predict([str(p) for p in batch],conf=.25,imgsz=1280,device=0,classes=[2,5,7],verbose=False)):
            label = path.parent.parent/"labels"/(path.stem+".txt")
            rows = [line.split() for line in label.read_text(encoding="utf-8").split("\n") if line.strip() and int(line.split()[0]) in (0,2,5,7)]
            pred = result.boxes
            for row in rows:
                if row[0]=="0" or not len(pred):
                    continue
                x,y,w,h = map(float,row[1:])
                iou = box_iou(torch.tensor([[x-w/2,y-h/2,x+w/2,y+h/2]]),pred.xyxyn.cpu())[0]
                best = int(iou.argmax())
                cls = str(int(pred.cls[best])) if iou[best]>=.5 else "2"
                changed += cls!=row[0]
                row[0] = cls
            label.write_text("".join(" ".join(row)+"\n" for row in rows),encoding="utf-8")
    print("relabel changed",changed,flush=True)


def train(*overrides):
    """python train_v4.py train name=v4c freeze=10 epochs=12 ...  (key=value pairs override the defaults below)"""
    from ultralytics import YOLO
    model = YOLO(str(ROOT.parent/"yolo26n.pt"))
    model.model.names[2] = "vehicle"
    # v3 used lr0=2e-5 with a frozen backbone for 10 epochs: steps far too small to move the weights.
    # Defaults: auto optimizer/lr, mosaic on, larger input for small distant cars.
    # v4  (80 ep, all classes): val peaked at epoch 4-8, then overfit the 11 training events.
    # v4b (30 ep, COCO-preserving labels): val best at epoch 1 -> full fine-tuning memorises scenes.
    args = dict(data=str(DATA/"dataset.yaml"),epochs=30,patience=10,imgsz=1280,batch=8,device=0,workers=6,seed=42,
                project=str(ROOT/"runs"),name="v4b",exist_ok=False,fliplr=0.5,close_mosaic=5,cos_lr=True,plots=True)
    for item in overrides:
        key,value = item.split("=",1)
        args[key] = yaml.safe_load(value)
    model.train(**args)


if __name__ == "__main__":
    {"prepare":prepare,"relabel":relabel,"train":train}[sys.argv[1]](*sys.argv[2:])
