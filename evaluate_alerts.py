"""System-level check on the 10 held-out demo videos: alerts from the model vs alerts from the human labels.

The same StopMonitor runs twice per video at 5 fps: once on AI-Hub XML boxes (reference), once on detector output.
python evaluate_alerts.py <weights> [alert_seconds]
"""
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from photo_analysis import ROOT, Detector
from temporal import StopMonitor, box_iou
from video_analysis import analyze_video

SOURCE = ROOT.parent/"docs/research/schoolzone_visibility_2026-10-07"


def reference_alerts(video,profile,alert_seconds):
    labels = next(SOURCE.glob(f"aihub169_suddenlyappear_*_labels/{video.stem}_00001.xml")).parent
    monitor = StopMonitor(profile["region"],profile.get("waiting_region"),alert_seconds)
    for i in range(1,301,2):  # label i <-> frame ceil(1.5(i-1)); odd i = every 3rd frame = 5 fps
        root = ET.parse(labels/f"{video.stem}_{i:05d}.xml").getroot()
        w,h = int(root.findtext("size/width")),int(root.findtext("size/height"))
        boxes = [{"kind":"vehicle","confidence":1,"box":[int(o.findtext("bndbox/"+k))/s for k,s in (("xmin",w),("ymin",h),("xmax",w),("ymax",h))]}
                 for o in root.iter("object") if o.findtext("name")=="vehicle"]
        monitor.update(math.ceil(1.5*(i-1))/15,boxes)
    return monitor.finish()["alerts"]


def match(reference,predicted):
    """Same vehicle = box IoU >= .3. Each reference alert is met by the EARLIEST model alert on that vehicle;
    later model alerts on an already-met vehicle are duplicates, model alerts on no reference vehicle are false."""
    out,duplicates,false = [],[],[]
    met = {}
    for p in sorted(predicted,key=lambda a:a["alert_seconds"]):
        best = max(range(len(reference)),key=lambda i:box_iou(reference[i]["box"],p["box"]),default=None)
        if best is None or box_iou(reference[best]["box"],p["box"])<.3:
            false.append(p["alert"])
        elif best in met:
            duplicates.append(p["alert"])
        else:
            met[best] = p
            out.append({"reference":reference[best]["alert"],"model":p["alert"],"delay_seconds":p["alert_seconds"]-reference[best]["alert_seconds"]})
    return out,duplicates,false


def summarize(rows,weights,alert_seconds):
    total = {"reference":sum(len(r["reference_alerts"]) for r in rows),"model":sum(len(r["model_alerts"]) for r in rows),
             "matched":sum(len(r["matched"]) for r in rows),"duplicates":sum(len(r["duplicates"]) for r in rows),"false":sum(len(r["false"]) for r in rows)}
    return {"weights":str(weights),"alert_seconds":alert_seconds,"sample_fps":5,**total,
            "recall":total["matched"]/total["reference"] if total["reference"] else None,
            "max_delay_seconds":max((m["delay_seconds"] for r in rows for m in r["matched"]),default=None),"scenes":rows}


def rescore():
    """Re-match saved alert lists (no inference) after a change to match()."""
    for path in sorted((ROOT/"evidence/v4_eval").glob("alerts_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["scenes"]:
            row["matched"],row["duplicates"],row["false"] = match(row["reference_alerts"],row["model_alerts"])
        data = summarize(data["scenes"],data["weights"],data["alert_seconds"])
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
        print(path.name,{k:data[k] for k in ("reference","matched","duplicates","false","max_delay_seconds")})


if __name__=="__main__":
    if sys.argv[1]=="rescore":
        rescore()
        sys.exit()
    weights = ROOT/sys.argv[1]
    alert_seconds = float(sys.argv[2]) if len(sys.argv)>2 else 10
    choice = {"selected_model":weights.stem,"selected_weights":str(weights.relative_to(ROOT)),"weights_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),
              "confidence":.25,"image_size":1280,"model_class_to_canonical":{"0":0,"2":1,"5":1,"7":1}}
    detector = Detector(choice)
    rows = []
    for key,profile in json.loads((ROOT/"demo_regions.json").read_text(encoding="utf-8")).items():
        video = next(SOURCE.glob(f"aihub169_suddenlyappear_*/{profile['video']}"))
        reference = reference_alerts(video,profile,alert_seconds)
        report = analyze_video(video,profile["region"],detector,waiting_region=profile.get("waiting_region"),alert_seconds=alert_seconds)
        pairs,duplicates,false = match(reference,report["alerts"])
        rows.append({"scene":key,"name":profile["name"],"reference_alerts":reference,"model_alerts":report["alerts"],"matched":pairs,"duplicates":duplicates,"false":false})
        print(f'{key:9} ref {len(reference)} model {len(report["alerts"])} hit {len(pairs)} dup {len(duplicates)} false {len(false)} delay {[round(p["delay_seconds"],1) for p in pairs]}',flush=True)
    summary = summarize(rows,weights,alert_seconds)
    name = weights.parent.parent.name if weights.stem in ("best","last") else weights.stem
    out = ROOT/"evidence/v4_eval"/f"alerts_{name}_{alert_seconds:g}s.json"
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print({k:summary[k] for k in ("reference","matched","duplicates","false","max_delay_seconds")})
