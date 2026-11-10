"""Fresh fine-tuning with retained head dimensions; not a copied pretrained release."""
import hashlib
import json
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT/"runtime").mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ROOT/"runtime")
from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conservative",action="store_true")
    conservative = parser.parse_args().conservative
    name = "v2_conservative" if conservative else "v2_retained_head"
    data = ROOT/"data_v2"
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["train"] == 1260
    hashes = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (data/"manifest.json",data/"dataset.yaml",data/"train.txt",data/"development.txt",data/"test.txt")}
    (ROOT/"evidence/v2_before_training.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    model = YOLO(str(ROOT.parent/"yolo26n.pt"))
    model.model.names[2] = "vehicle"
    result = model.train(data=str(data/"dataset.yaml"),epochs=5 if conservative else 30,patience=8,imgsz=960,batch=8,device=0,workers=2,
                         seed=42,deterministic=True,cache=False,project=str(ROOT/"runs"),name=name,exist_ok=False,
                         freeze=11,optimizer="AdamW",lr0=.00002 if conservative else .0002,lrf=.5 if conservative else .2,warmup_epochs=1,mosaic=0,fliplr=0,
                         degrees=0,translate=.05,scale=.15,hsv_h=.005,hsv_s=.1,hsv_v=.2,plots=True)
    metrics_file = "v2_conservative_native_metrics.json" if conservative else "v2_native_metrics.json"
    (ROOT/"evidence"/metrics_file).write_text(json.dumps(result.results_dict,indent=2),encoding="utf-8")
