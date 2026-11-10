"""Build additional pairs exclusively from the already verified real videos/XML."""
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/"docs/research/schoolzone_visibility_2026-10-07"
OUTPUT = ROOT/"data_v2"


def split_for_video(name, original_split):
    if name.startswith("2020_11_12_12_30_"):
        return "test"
    if name == "2020_11_13_18_30_suddenlyappear_night_B_11.mp4":
        return "crossview"
    return "development" if original_split == "validation" else "train"


def label_class(name):
    return {"person":0,"vehicle":2}.get(name)


if __name__ == "__main__":
    cv2.setNumThreads(2)
    rows = []
    for original in ("train1","train4","validation"):
        for video in sorted((SOURCE/f"aihub169_suddenlyappear_{original}").glob("*.mp4")):
            split = split_for_video(video.name,original)
            for kind in ("images","labels"):
                (OUTPUT/split/kind).mkdir(parents=True,exist_ok=True)
            if split == "train":
                capture = cv2.VideoCapture(str(video))
                assert capture.get(cv2.CAP_PROP_FPS) == 15
                expected = {math.ceil(1.5*k):k for k in range(0,300,5)}
                frame_index = 0
                while capture.grab():
                    if frame_index in expected:
                        ok,frame = capture.retrieve()
                        assert ok
                        label_index = expected[frame_index]+1
                        stem = f"{video.stem}_{label_index:05d}"
                        image_path = OUTPUT/split/"images"/(stem+".jpg")
                        encoded,buffer = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,95])
                        assert encoded
                        buffer.tofile(str(image_path))
                        rows.append({"split":split,"image":str(image_path.resolve()),"video":str(video),"original":original,"label_index":label_index,"time":frame_index/15})
                    frame_index += 1
                capture.release()
                assert frame_index == 450
            else:
                for image in sorted((SOURCE/"paired_images_1fps"/original/"images").glob(video.stem+"_*.jpg")):
                    image_path = OUTPUT/split/"images"/image.name
                    shutil.copyfile(image,image_path)
                    label_index = int(image.stem.rsplit("_",1)[1])
                    rows.append({"split":split,"image":str(image_path.resolve()),"video":str(video),"original":original,"label_index":label_index,"time":(label_index-1)/10})
            print(split,video.name,flush=True)
    counts = {}
    for row in rows:
        path = Path(row["image"])
        xml = SOURCE/f'aihub169_suddenlyappear_{row["original"]}_labels'/(path.stem+".xml")
        annotation = ET.parse(xml).getroot()
        width,height = (int(annotation.findtext("size/"+axis)) for axis in ("width","height"))
        lines = []
        for obj in annotation.findall("object"):
            cls = label_class(obj.findtext("name"))
            if cls is None:
                continue
            x1,y1,x2,y2 = (int(obj.findtext("bndbox/"+key)) for key in ("xmin","ymin","xmax","ymax"))
            assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
            lines.append(f"{cls} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
        labels = path.parent.parent/"labels"/(path.stem+".txt")
        labels.write_text("\n".join(lines)+"\n",encoding="utf-8")
        row["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        row["label_sha256"] = hashlib.sha256(labels.read_bytes()).hexdigest()
        counts[row["split"]] = counts.get(row["split"],0)+1
    assert counts == {"train":1260,"test":60,"crossview":30,"development":300},counts
    for split in counts:
        paths = [row["image"].replace("\\","/") for row in rows if row["split"]==split]
        (OUTPUT/f"{split}.txt").write_text("\n".join(paths)+"\n",encoding="utf-8")
    from ultralytics import YOLO
    names = YOLO(str(ROOT.parent/"yolo26n.pt")).names
    names[2] = "vehicle"
    (OUTPUT/"dataset.yaml").write_text(yaml.safe_dump({"path":str(OUTPUT).replace("\\","/"),"train":"train.txt","val":"development.txt","names":names},sort_keys=False),encoding="utf-8")
    (OUTPUT/"manifest.json").write_text(json.dumps({"counts":counts,"rows":rows,"test_episode":"2020_11_12_12_30","warning":"2fps increases temporal sampling, not independent incidents; crossview same truck episode A trained/B development"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(counts,flush=True)
