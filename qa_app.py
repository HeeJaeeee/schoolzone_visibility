"""Acceptance check using the actual Tk app and fresh neural-network inference."""
import ctypes
import hashlib
import json
import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageGrab
from app import PhotoApp
from photo_analysis import ROOT, DEMO, assess

OUT = ROOT/"evidence/app"
OUT.mkdir(parents=True,exist_ok=True)


def settle(root,seconds=.2):
    end = time.monotonic()+seconds
    while time.monotonic()<end:
        root.update()
        time.sleep(.015)


def capture(root,name):
    root.lift()
    settle(root,.3)
    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    if not hwnd:
        hwnd = root.winfo_id()
    shot = ImageGrab.grab(window=hwnd)
    assert shot.width>1000 and shot.height>700
    shot.save(OUT/name)


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoApp(root)
    root.geometry("1380x900+40+40")
    settle(root)
    records = []
    try:
        for index,entry in enumerate(app.catalog):
            app.open_path(DEMO/entry["file"])
            app.buttons[2].invoke()
            assert app.busy
            assert str(app.save_button["state"]) == "disabled"
            deadline = time.monotonic()+120
            while app.busy and time.monotonic()<deadline:
                root.update()
                time.sleep(.025)
            assert not app.busy and app.detections is not None, app.detail.get()
            analysis = assess(app.detections,app.region)
            paths = app.save_result(OUT/"exports")
            saved = json.loads(paths[1].read_text(encoding="utf-8"))
            assert saved["analysis"] == analysis
            assert saved["source_sha256"] == hashlib.sha256(app.path.read_bytes()).hexdigest()
            with Image.open(paths[0]) as image:
                assert image.size == app.image.size
                image.verify()
            records.append({"file":entry["file"],"analysis":analysis,"output_json":str(paths[1]),"detections":len(app.detections)})
            if index in (0,3,6,9,10):
                capture(root,f"screen_{index:02d}.png")
            print(index+1,entry["file"],analysis["status"],analysis["candidate_count"],flush=True)

        # Exercise real pointer events, including reverse direction and letterbox.
        app.start_edit()
        ox,oy,width,height = app.display_box
        def event(kind,x,y):
            app.canvas.event_generate(kind,x=int(ox+x*width),y=int(oy+y*height))
            root.update()
        event("<ButtonPress-1>",.6,.6)
        event("<B1-Motion>",.2,.2)
        event("<ButtonRelease-1>",.2,.2)
        assert all(abs(a-b)<.005 for a,b in zip(app.region,[.2,.2,.6,.6])),app.region
        assert app.detections is None
        assert str(app.save_button["state"]) == "disabled"
        external = OUT/"external_same_pixels.jpg"
        external.write_bytes(app.path.read_bytes())
        app.open_path(external)
        assert app.region is None, "Same pixels/path-independent uploads must not inherit ROI"
        app.run_analysis()
        assert not app.busy and app.detections is None
        capture(root,"screen_region_required.png")
        report = {"real_photos":len(records),"fresh_inference":True,"source":"AI-Hub 169; acted real recordings included", "records":records,
                  "gui_pointer_drag_verified":True,"external_photo_region_reset":True,"stale_save_disabled":True,
                  "limits":"12 selected correlated photos; overlap candidates only, not actual visibility/duration/independent accuracy"}
        (OUT/"acceptance.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print("Acceptance complete",flush=True)
    finally:
        root.destroy()
