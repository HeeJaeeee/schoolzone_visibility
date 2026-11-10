"""Development selection first; separate episode test is never a tuning input."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image
from photo_analysis import canonical_detection
from metrics import match_counts
from temporal import box_iou

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"data_v2"
os.environ["YOLO_CONFIG_DIR"] = str(ROOT/"runtime")
from ultralytics import YOLO


def evaluate(weights,name,splits):
    model = YOLO(str(weights))
    output = ROOT/"evidence/v2_evaluation"/name
    output.mkdir(parents=True,exist_ok=True)
    records,totals = [],{}
    for split in splits:
        paths = [Path(line) for line in (DATA/f"{split}.txt").read_text(encoding="utf-8").splitlines()]
        counts = {kind:{"tp":0,"fp":0,"fn":0} for kind in (0,1)}
        truck = 0
        for start in range(0,len(paths),4):
            for path,result in zip(paths[start:start+4],model.predict(source=[str(p) for p in paths[start:start+4]],stream=True,imgsz=960,conf=.25,device="cpu",classes=[0,2,5,7],verbose=False)):
                height,width = result.orig_shape
                detections = [canonical_detection(cls,score,box,width,height) for box,score,cls in zip(result.boxes.xyxy.tolist(),result.boxes.conf.tolist(),result.boxes.cls.tolist())]
                predicted = [[0 if d["kind"]=="person" else 1,d["confidence"],*(x*size for x,size in zip(d["box"],(width,height,width,height)))] for d in detections]
                true = []
                for line in (path.parent.parent/"labels"/(path.stem+".txt")).read_text(encoding="utf-8").splitlines():
                    cls,x,y,w,h = map(float,line.split())
                    true.append([0 if cls==0 else 1,(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])
                per_class = {}
                for kind in (0,1):
                    c = match_counts([p for p in predicted if p[0]==kind],[t for t in true if t[0]==kind])
                    per_class[kind] = c
                    for key in c:
                        counts[kind][key] += c[key]
                truck_detected = None
                if split == "crossview":
                    reference = [1584/3840,215/2160,2153/3840,728/2160]
                    assert any(box_iou(reference,[t[1]/width,t[2]/height,t[3]/width,t[4]/height])>.99 for t in true if t[0]==1)
                    truck_detected = any(d["kind"]=="vehicle" and box_iou(reference,d["box"])>=.5 for d in detections)
                    truck += int(truck_detected)
                records.append({"split":split,"image":str(path),"detections":detections,"counts":per_class,"truck_detected":truck_detected})
        metrics = {}
        for kind,c in counts.items():
            precision = c["tp"]/(c["tp"]+c["fp"]) if c["tp"]+c["fp"] else 0
            recall = c["tp"]/(c["tp"]+c["fn"]) if c["tp"]+c["fn"] else 0
            metrics[kind] = {**c,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0}
        totals[split] = {"images":len(paths),"metrics":metrics,"truck_localized":truck if split=="crossview" else None}
        print(name,split,json.dumps(totals[split]),flush=True)
    (output/"per_image.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    summary = {"model":str(weights),"sha256":hashlib.sha256(Path(weights).read_bytes()).hexdigest(),"splits":totals,"confidence":.25,"imgsz":960,"matching_iou":.5}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=["development","test"])
    parser.add_argument("--weights")
    parser.add_argument("--run",default="v2_retained_head")
    args = parser.parse_args()
    if args.stage == "development":
        summaries = []
        for name in ("best","last"):
            path = ROOT/"runs"/args.run/"weights"/(name+".pt")
            label = name if args.run=="v2_retained_head" else args.run+"_"+name
            summary = evaluate(path,label,["development","crossview"])
            accepted = summary["splits"]["development"]["metrics"][1]["f1"]>=.75 and summary["splits"]["crossview"]["truck_localized"]>=28
            summaries.append({"name":name,"accepted":accepted,"summary":summary})
        accepted = [row for row in summaries if row["accepted"]]
        selected = max(accepted,key=lambda row:row["summary"]["splits"]["development"]["metrics"][1]["f1"]) if accepted else None
        filename = "v2_selection.json" if args.run=="v2_retained_head" else args.run+"_selection.json"
        (ROOT/"evidence"/filename).write_text(json.dumps({"candidates":summaries,"selected":selected,"test_not_used_for_selection":True},indent=2),encoding="utf-8")
    else:
        assert args.weights
        evaluate(Path(args.weights),"selected_test",["test"])
