"""Train the approved real-camera object detector; no data generation."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "docs/research/schoolzone_visibility_2026-10-07/paired_images_1fps"
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "runtime")

from ultralytics import YOLO


if __name__ == "__main__":
    for name, expected in (("train", 660), ("val", 300), ("demo", 60)):
        paths = (DATA / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        assert len(paths) == expected and all(Path(p).exists() for p in paths)
    model = YOLO(str(ROOT.parent / "yolo26n.pt"))
    result = model.train(
        data=str(DATA / "dataset.yaml"), epochs=40, patience=10,
        imgsz=960, batch=8, device=0, workers=2, seed=42,
        deterministic=True, cache=False, project=str(ROOT / "runs"),
        name="yolo26n_960_seed42", exist_ok=False, plots=True,
    )
    (ROOT / "evidence/native_training_metrics.json").write_text(
        json.dumps(result.results_dict, indent=2), encoding="utf-8"
    )
