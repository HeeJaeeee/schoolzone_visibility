"""Same held-out images, same matcher, for every model: person/vehicle AP and P/R/F1.

python evaluate_v4.py test pretrained=../yolo26n.pt v3=weights/detector.pt v4=runs/v4/weights/best.pt
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ["YOLO_CONFIG_DIR"] = str(ROOT/"runtime")
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ap_per_class, box_iou

CANON = {0:0,2:1,5:1,7:1}  # person / vehicle (COCO car, bus, truck)
IOUV = torch.linspace(.5,.95,10)
# Blind-spot zone of each demo video: the vehicles the alert actually depends on.
from photo_analysis import ground_point, inside, polygon
ZONES = {p["video"][:-4]:polygon(p["region"]) for p in json.loads((ROOT/"demo_regions.json").read_text(encoding="utf-8")).values()}


def matched(a,b):
    """Boolean per row of a: does it have a one-to-one IoU>=.5 partner in b (both [cls,x1,y1,x2,y2] vehicle rows)."""
    if not len(a) or not len(b):
        return torch.zeros(len(a),dtype=torch.bool)
    hit = DetectionValidator.match_predictions(SimpleNamespace(iouv=IOUV[:1]),a[:,0],b[:,0],box_iou(b[:,1:],a[:,1:]))
    return torch.as_tensor(hit)[:,0]


def labels(path,w,h):
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip() and int(line.split()[0]) in CANON:
            c,x,y,bw,bh = map(float,line.split())
            rows.append([CANON[int(c)],(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
    return torch.tensor(rows).reshape(-1,5)


def evaluate(weights,split,conf=.25,imgsz=1280):
    model = YOLO(str(weights))
    images = sorted((ROOT/"data_v4"/split/"images").glob("*.jpg"))
    stats = {"tp":[],"conf":[],"cls":[],"target":[]}
    at = {(scene,c):{"tp":0,"fp":0,"fn":0} for scene in ("all","day","night") for c in (0,1)}
    zc = {"gt":0,"found":0,"pred":0,"correct":0}
    for start in range(0,len(images),16):
        batch = images[start:start+16]
        for path,result in zip(batch,model.predict([str(p) for p in batch],conf=.001,imgsz=imgsz,device=0,classes=list(CANON),verbose=False)):
            h,w = result.orig_shape
            true = labels(path.parent.parent/"labels"/(path.stem+".txt"),w,h)
            b = result.boxes
            pred = torch.cat([torch.tensor([CANON[int(c)] for c in b.cls.tolist()]).reshape(-1,1).float(),b.conf.cpu().reshape(-1,1),b.xyxy.cpu()],1)
            correct = DetectionValidator.match_predictions(SimpleNamespace(iouv=IOUV),pred[:,0],true[:,0],box_iou(true[:,1:],pred[:,2:])) if len(pred) and len(true) else torch.zeros(len(pred),10,dtype=torch.bool)
            stats["tp"].append(np.asarray(correct)); stats["conf"].append(pred[:,1].numpy()); stats["cls"].append(pred[:,0].numpy()); stats["target"].append(true[:,0].numpy())
            kept = pred[pred[:,1]>=conf]  # re-match after filtering: low-conf boxes must not steal matches
            hit = DetectionValidator.match_predictions(SimpleNamespace(iouv=IOUV[:1]),kept[:,0],true[:,0],box_iou(true[:,1:],kept[:,2:]))[:,0] if len(kept) and len(true) else torch.zeros(len(kept),dtype=torch.bool)
            scene = "night" if "_night_" in path.name else "day"
            for c in (0,1):
                p,t = kept[:,0]==c,true[:,0]==c
                tp = int(torch.as_tensor(hit)[p].sum())
                for s in ("all",scene):
                    at[(s,c)]["tp"] += tp; at[(s,c)]["fp"] += int(p.sum())-tp; at[(s,c)]["fn"] += int(t.sum())-tp
            zone = ZONES.get(path.stem[:-6])
            if zone:
                w_, h_ = w,h
                rows = lambda boxes:boxes[[inside(ground_point([b[1]/w_,b[2]/h_,b[3]/w_,b[4]/h_]),zone) for b in boxes.tolist()]] if len(boxes) else boxes
                gt_v,pr_v = true[true[:,0]==1],kept[kept[:,0]==1][:,[0,2,3,4,5]]
                gt_z,pr_z = rows(gt_v),rows(pr_v)
                zc["gt"] += len(gt_z); zc["found"] += int(matched(gt_z,pr_v).sum())
                zc["pred"] += len(pr_z); zc["correct"] += int(matched(pr_z,gt_v).sum())
    tp,confs,cls,target = (np.concatenate(stats[k]) for k in ("tp","conf","cls","target"))
    out = ap_per_class(tp,confs,cls,target,names={0:"person",1:"vehicle"})
    ap = out[5]
    result = {"weights":str(weights),"split":split,"images":len(images),"imgsz":imgsz,"conf":conf,"classes":{}}
    for i,c in enumerate(out[6]):
        name = {0:"person",1:"vehicle"}[int(c)]
        row = {"AP50":float(ap[i,0]),"AP50_95":float(ap[i].mean())}
        for s in ("all","day","night"):
            k = at[(s,int(c))]
            p = k["tp"]/(k["tp"]+k["fp"]) if k["tp"]+k["fp"] else 0
            r = k["tp"]/(k["tp"]+k["fn"]) if k["tp"]+k["fn"] else 0
            row[s] = {**k,"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0}
        result["classes"][name] = row
    result["zone_vehicle"] = {**zc,"recall":zc["found"]/zc["gt"] if zc["gt"] else 0,"precision":zc["correct"]/zc["pred"] if zc["pred"] else 0}
    return result


if __name__ == "__main__":
    split,models = sys.argv[1],dict(arg.split("=",1) for arg in sys.argv[2:])
    (ROOT/"evidence/v4_eval").mkdir(parents=True,exist_ok=True)
    # name=weights[@imgsz]
    results = {name:evaluate(ROOT/path.split("@")[0],split,imgsz=int(path.split("@")[1]) if "@" in path else 1280) for name,path in models.items()}
    (ROOT/"evidence/v4_eval"/f"{split}_{'_'.join(models)}.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(f"{'model':12}{'class':9}{'AP50':>7}{'AP50-95':>9}{'P@.25':>8}{'R@.25':>8}{'F1@.25':>8}{'R day':>8}{'R night':>9}")
    for name,r in results.items():
        for c,row in r["classes"].items():
            a = row["all"]
            print(f"{name:12}{c:9}{row['AP50']:7.3f}{row['AP50_95']:9.3f}{a['precision']:8.3f}{a['recall']:8.3f}{a['f1']:8.3f}{row['day']['recall']:8.3f}{row['night']['recall']:9.3f}")
        z = r["zone_vehicle"]
        print(f"{name:12}{'zone-veh':9}  P {z['precision']:.3f} ({z['correct']}/{z['pred']})  R {z['recall']:.3f} ({z['found']}/{z['gt']})")
