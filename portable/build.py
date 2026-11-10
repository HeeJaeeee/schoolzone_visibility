"""Build portable/dist/매의눈.zip: extract anywhere, double-click 매의눈.exe. No torch, demo photos/videos embedded.

No exe of our own: an unsigned exe is new to Smart App Control on every build and gets blocked, and SAC blocks .cmd files that came
from the web. 매의눈.exe is the signed pythonw.exe renamed; everything else is signed (python, tcl/tk, onnxruntime) or widely used
(numpy, opencv, Pillow), byte-identical to the binaries already running on this PC.

..\\.venv-exe\\Scripts\\python.exe portable\\build.py   (venv: pillow==12.3.0 numpy==2.5.2 opencv-python-headless onnxruntime-directml; ffmpeg on PATH)
Pins = versions SAC already trusts; Pillow 13.0.0's _imagingtk was blocked. portable/detector_v4.onnx comes from .venv-battery (ultralytics):
YOLO('weights/detector_v4.pt').export(format='onnx',imgsz=(736,1280),nms=True,conf=0.25,iou=0.7,opset=17,simplify=True)
"""
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
APP = HERE.parent
DATA = APP.parent/"docs/research/schoolzone_visibility_2026-10-07"
BUNDLE = HERE/"bundle"
DIST = HERE/"dist/매의눈"
CRF = 32  # alerts checked against evidence/v4_eval/alerts_v4c_10s.json
CODE = ("app.py","video_window.py","photo_analysis.py","video_analysis.py","temporal.py")
PACKAGES = ("numpy","numpy.libs","PIL","cv2","onnxruntime")
# never imported by the app: tests, tools, AVIF, ssl (hashlib falls back to its built-in sha256), sqlite, IDLE, tix
SKIP = shutil.ignore_patterns("__pycache__","tests","site-packages","idlelib","ensurepip","tix8.4.3","nmake","*.lib","_test*","_avif*",
                              "transformers","quantization","tools","libcrypto*","libssl*","_ssl.pyd","_hashlib.pyd","sqlite3.dll","_sqlite3.pyd")


def prepare():
    shutil.rmtree(BUNDLE,ignore_errors=True)
    (BUNDLE/"weights").mkdir(parents=True)
    onnx = BUNDLE/"weights/detector_v4.onnx"
    shutil.copy(HERE/"detector_v4.onnx",onnx)
    choice = json.loads((APP/"model_choice.json").read_text(encoding="utf-8"))
    choice.update(selected_weights="weights/detector_v4.onnx",weights_sha256=hashlib.sha256(onnx.read_bytes()).hexdigest(),
                  exported_from={"weights":"weights/detector_v4.pt","sha256":choice["weights_sha256"],"format":"onnx 736x1280, NMS iou 0.7 inside"})
    (BUNDLE/"model_choice.json").write_text(json.dumps(choice,ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.copy(APP/"demo_regions.json",BUNDLE)

    photos = BUNDLE/"demo/demo_photos"
    photos.mkdir(parents=True)
    manifest = json.loads((DATA/"demo_photos/manifest.json").read_text(encoding="utf-8"))
    shutil.copy(DATA/"demo_photos/manifest.json",photos)
    for entry in manifest:
        with Image.open(DATA/"demo_photos"/entry["file"]) as image:
            image.convert("RGB").resize((1280,round(1280*image.height/image.width)),Image.Resampling.LANCZOS).save(photos/entry["file"],quality=88)

    for split,video in sorted({(e["source_split"],e["source_video"]) for e in manifest}):
        source = DATA/f"aihub169_suddenlyappear_{split}"/video
        target = BUNDLE/f"demo/aihub169_suddenlyappear_{split}"/video
        target.parent.mkdir(exist_ok=True)
        capture = cv2.VideoCapture(str(source))
        assert capture.get(cv2.CAP_PROP_FPS)==15, source
        capture.release()
        # Keep every 3rd frame = exactly the frames the app analyses and shows (5 fps); timestamps stay index/fps.
        # neighbor: 4K->1280 is exactly 3x, where cv2 INTER_LINEAR (the app's letterbox) picks every 3rd pixel; same pixels as the 4K run.
        subprocess.run(["ffmpeg","-v","error","-y","-i",str(source),"-vf","select='not(mod(n\\,3))',setpts=N/5/TB,scale=1280:-2:flags=neighbor","-r","5",
                        "-c:v","libx264","-preset","slow","-crf",str(CRF),"-g","15","-pix_fmt","yuv420p","-an",str(target)],check=True)
        print(f"{video}: {target.stat().st_size/2**20:.1f} MB",flush=True)


def package():
    shutil.rmtree(DIST.parent,ignore_errors=True)
    DIST.mkdir(parents=True)
    base = Path(sys.base_prefix)
    for item in [*base.glob("*.dll"),base/"LICENSE.txt"]:
        shutil.copy(item,DIST)
    shutil.copy(base/"pythonw.exe",DIST/"매의눈.exe")  # renaming keeps the signature valid; Python finds Lib/ next to any exe name
    for folder in ("DLLs","Lib","tcl"):
        shutil.copytree(base/folder,DIST/folder,ignore=SKIP)
    for name in PACKAGES:
        shutil.copytree(Path(sys.prefix)/"Lib/site-packages"/name,DIST/"Lib/site-packages"/name,ignore=SKIP)
    (DIST/"Lib/site-packages/maeui.pth").write_text('import os, runpy, sys; runpy.run_path(os.path.join(sys.prefix, "app", "launch.py"))\n',encoding="ascii")
    shutil.copytree(BUNDLE,DIST/"app")
    for path in [*(APP/name for name in CODE),HERE/"launch.py"]:
        shutil.copy(path,DIST/"app")
    archive = DIST.parent/"매의눈.zip"  # deflate zip: the only archive every Windows Explorer extracts (no .7z in the classroom)
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(DIST.rglob("*")):
            z.write(path,path.relative_to(DIST.parent))
    size = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(f"{archive}: {archive.stat().st_size/2**20:.1f} MB (unpacked {size/2**20:.1f} MB)")


if __name__=="__main__":
    if "package" not in sys.argv[1:]:
        prepare()
    if "prepare" not in sys.argv[1:]:
        package()
