"""Evidence validation and parameter-change audit for the actual v2 run."""
import hashlib
import json
from pathlib import Path
from PIL import Image
from photo_analysis import ROOT, annotated


if __name__ == "__main__":
    data = ROOT/"data_v2"
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    groups = {}
    for row in manifest["rows"]:
        path = Path(row["image"])
        labels = path.parent.parent/"labels"/(path.stem+".txt")
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row["image_sha256"]
        assert hashlib.sha256(labels.read_bytes()).hexdigest()==row["label_sha256"]
        group = Path(row["video"]).stem.replace("_A_","_VIEW_").replace("_B_","_VIEW_")
        groups.setdefault(row["split"],set()).add(group)
        with Image.open(path) as image:
            assert image.size==(3840,2160)
            image.verify()
    assert not groups["train"]&groups["test"]
    assert not groups["train"]&groups["development"]
    assert len(groups["train"]&groups["crossview"])==1
    before = json.loads((ROOT/"evidence/v2_before_training.json").read_text(encoding="utf-8"))
    assert all(hashlib.sha256((data/name).read_bytes()).hexdigest()==value for name,value in before.items())
    row = next(row for row in manifest["rows"] if row["split"]=="train" and row["label_index"]==6 and "night_A_11" in row["video"])
    path = Path(row["image"])
    detections = []
    for line in (path.parent.parent/"labels"/(path.stem+".txt")).read_text().splitlines():
        cls,x,y,w,h = map(float,line.split())
        detections.append({"kind":"person" if cls==0 else "vehicle","confidence":1.,"box":[x-w/2,y-h/2,x+w/2,y+h/2]})
    with Image.open(path) as image:
        annotated(image,None,detections).save(ROOT/"evidence/v2_label_alignment_half_second.png")
    report = {"verified_images":len(manifest["rows"]),"counts":manifest["counts"],"train_test_episode_overlap":0,
              "train_development_episode_overlap":0,"crossview_same_episode_overlap":1,"before_after_manifest_sha_match":True}
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",default=str(ROOT/"runs/v2_retained_head/weights/best.pt"))
    weights = Path(parser.parse_args().weights)
    if weights.exists():
        from ultralytics import YOLO
        source = dict(YOLO(str(ROOT.parent/"yolo26n.pt")).model.named_parameters())
        trained = dict(YOLO(str(weights)).model.named_parameters())
        assert source.keys()==trained.keys()
        changed = [key for key in source if not source[key].float().equal(trained[key].float())]
        frozen_changed = [key for key in changed if int(key.split(".")[1])<11]
        assert not frozen_changed
        assert changed
        report.update({"changed_parameter_tensors":len(changed),"total_parameter_tensors":len(source),"frozen_parameter_changes":len(frozen_changed),"best_weight_sha256":hashlib.sha256(weights.read_bytes()).hexdigest()})
    (ROOT/"evidence/v2_integrity.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report,flush=True)
