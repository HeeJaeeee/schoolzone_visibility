"""Compare detection on real images; these are not visibility-risk scores."""
import argparse
import json
import os
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "docs/research/schoolzone_visibility_2026-10-07"
DATA = SOURCE / "paired_images_1fps"
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "runtime")

from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

from metrics import match_counts


def source_batches(paths):
    for start in range(0, len(paths), 4):
        yield [str(path) for path in paths[start:start + 4]]


def ground_truth(path, width, height):
    rows = []
    label = path.parent.parent / "labels" / (path.stem + ".txt")
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = map(float, line.split())
        rows.append([int(cls), (x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
    return rows


def rates(counts):
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return {**counts, "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("name", choices=["pretrained", "trained"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths = []
    split_by_name = {}
    for split, manifest in (("validation", "val.txt"), ("excluded_truck", "demo.txt")):
        for line in (DATA / manifest).read_text(encoding="utf-8").splitlines():
            path = Path(line)
            paths.append(path)
            split_by_name[path.name] = split
    assert len(paths) == 360 and len(split_by_name) == 360
    curated = {item["file"] for item in json.loads((SOURCE / "demo_photos/manifest.json").read_text())}
    output = ROOT / "evidence" / args.name
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    totals = {split: {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in (0, 1)} for split in ("validation", "excluded_truck")}
    # Manually inspected stationary-truck boxes from the first source XML.
    truck_boxes = {"A": [1318, 106, 2101, 736], "B": [1584, 215, 2153, 728]}
    trucks = {camera: {"images": 0, "detected": 0} for camera in truck_boxes}
    records = []
    baseline = args.name == "pretrained"
    model = YOLO(args.weights)
    predictions = (
        result
        for sources in source_batches(paths)
        for result in model.predict(
            source=sources, stream=True, imgsz=960, batch=4,
            device="cpu", conf=0.25, verbose=False, save=False,
            classes=[0, 2, 5, 7] if baseline else None,
        )
    )
    for index, result in enumerate(predictions):
        path = Path(result.path)
        split = split_by_name[path.name]
        height, width = result.orig_shape
        true = ground_truth(path, width, height)
        pred = []
        for box, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            canonical = (0 if int(cls) == 0 else 1) if baseline else int(cls)
            pred.append([canonical, score, *box])
        counts = {}
        for cls in (0, 1):
            counts[cls] = match_counts([p for p in pred if p[0] == cls], [t for t in true if t[0] == cls])
            for key, value in counts[cls].items():
                totals[split][cls][key] += value
        target_detected = None
        if split == "excluded_truck":
            camera = "A" if "_night_A_" in path.stem else "B"
            target = torch.tensor([truck_boxes[camera]], dtype=torch.float32)
            vehicles = [p[2:] for p in pred if p[0] == 1]
            target_detected = bool(vehicles and box_iou(target, torch.tensor(vehicles)).max() >= 0.5)
            trucks[camera]["images"] += 1
            trucks[camera]["detected"] += int(target_detected)
        records.append({"image": path.name, "split": split, "counts": counts, "predictions": pred, "truck_detected": target_detected})
        if path.name in curated:
            image = result.orig_img.copy()
            for cls, score, x1, y1, x2, y2 in pred:
                color = (70, 210, 80) if cls == 0 else (0, 180, 255)
                a, b = (round(x1), round(y1)), (round(x2), round(y2))
                cv2.rectangle(image, a, b, color, 4)
                cv2.putText(image, f'{"person" if cls == 0 else "vehicle"} {score:.2f}', (a[0], max(28, a[1]-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imencode(".jpg", image)[1].tofile(str(output / "predictions" / path.name))
        if (index + 1) % 30 == 0:
            print(f"{args.name}: {index+1}/360 actual images evaluated", flush=True)
    assert len(records) == 360
    summary = {
        "model": args.name, "weights": str(Path(args.weights).resolve()),
        "confidence": 0.25, "iou": 0.5, "image_size": 960, "inference_device": "cpu",
        "coco_vehicle_mapping": [2, 5, 7] if baseline else None,
        "by_split": {s: {"person" if c == 0 else "vehicle": rates(v) for c, v in groups.items()} for s, groups in totals.items()},
        "stationary_truck": trucks,
        "scope": "Object localization only; not driver visibility or facility-priority accuracy.",
    }
    (output / "per_image.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
