"""Real selected-model inference through the actual photo and video windows."""
import ctypes
import hashlib
import json
import time
import tkinter as tk
import xml.etree.ElementTree as ET

from PIL import Image, ImageGrab
from app import PhotoApp
from photo_analysis import ROOT, DEMO, assess
from temporal import ObservationLog
from qa_app import settle

OUT = ROOT/"evidence/app_v2_trained"
OUT.mkdir(parents=True,exist_ok=True)


def screenshot(window,name):
    window.lift()
    settle(window,.3)
    hwnd = ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
    image = ImageGrab.grab(window=hwnd)
    assert image.width>=960 and image.height>=400
    image.save(OUT/name)


def wait_video(root,dialog):
    deadline = time.monotonic()+180
    while dialog.running and time.monotonic()<deadline:
        settle(root,.05)
    assert not dialog.running, "Video worker timed out"


def reference_report(data):
    source = ROOT.parent/"docs/research/schoolzone_visibility_2026-10-07"
    metadata = data["metadata"]
    log = ObservationLog(data["region"])
    for second in range(30):
        label = source/f'aihub169_suddenlyappear_{metadata["source_split"]}_labels'/f'{data["metadata"]["source_video"][:-4]}_{10*second+1:05d}.xml'
        annotation = ET.parse(label).getroot()
        width,height = (int(annotation.findtext("size/"+axis)) for axis in ("width","height"))
        detections = []
        for obj in annotation.findall("object"):
            kind = obj.findtext("name")
            if kind not in ("person","vehicle"):
                continue
            box = [float(obj.findtext("bndbox/"+key))/size for key,size in zip(("xmin","ymin","xmax","ymax"),(width,height,width,height))]
            detections.append({"kind":kind,"confidence":1.,"box":box})
        log.observe(second,detections)
    return log.finish()


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoApp(root)
    root.geometry("1380x900+40+40")
    assert app.detector.choice["selected_model"]=="AI-Hub trained YOLO26 nano (v2)"
    assert app.detector.choice["expected_model_names"]["2"]=="vehicle"
    photos,videos = [],[]
    try:
        for index,entry in enumerate(app.catalog):
            app.open_path(DEMO/entry["file"])
            app.buttons[2].invoke()
            assert app.busy and str(app.save_button["state"])=="disabled"
            deadline = time.monotonic()+120
            while app.busy and time.monotonic()<deadline:
                settle(root,.025)
            assert not app.busy and app.detections is not None,app.detail.get()
            paths = app.save_result(OUT/"photo_exports")
            data = json.loads(paths[1].read_text(encoding="utf-8"))
            assert data["analysis"]==assess(app.detections,app.region)
            assert data["source_sha256"]==hashlib.sha256(app.path.read_bytes()).hexdigest()
            assert data["model_sha256"]==app.detector.choice["weights_sha256"]
            with Image.open(paths[0]) as image:
                assert image.size==app.image.size
                image.verify()
            photos.append({"file":entry["file"],"result":str(paths[1]),"status":data["analysis"]["status"],"candidates":data["analysis"]["candidate_count"]})
            if index in (0,3,6,9):
                screenshot(root,f"photo_{index:02d}.png")
            print("Photo",index+1,data["analysis"]["status"],flush=True)

        unique = {entry["source_video"]:entry for entry in app.catalog}
        for index,entry in enumerate(unique.values()):
            app.open_path(DEMO/entry["file"])
            app.video_button.invoke()
            dialog = app.video_window
            assert app.busy and str(app.buttons[2]["state"])=="disabled"
            wait_video(root,dialog)
            assert dialog.report_path is not None,dialog.count.get()
            data = json.loads(dialog.report_path.read_text(encoding="utf-8"))
            assert data["complete"] and data["samples"]==30 and data["decoded_frames"]==450
            assert data["source_duration_seconds"]==30 and not data["safety_verified"]
            assert data["source_sha256"]==hashlib.sha256(dialog.path.read_bytes()).hexdigest()
            assert data["inference_model"]["weights_sha256"]==app.detector.choice["weights_sha256"]
            assert [item["time"] for item in data["detections_by_sample"]]==list(range(30))
            assert len(data["episodes"])==len(data["representative_images"])
            for path in data["representative_images"]:
                with Image.open(path) as image:
                    assert image.size==(3840,2160)
                    image.verify()
            screenshot(dialog.window,f"video_{index:02d}.png")
            reference = reference_report(data)
            videos.append({"source":str(dialog.path),"result":str(dialog.report_path),"candidate_samples":data["candidate_sample_count"],
                           "persistent_samples":data["persistent_sample_count"],"longest_seconds":data["longest_seconds"],"episodes":data["episodes"],
                           "xml_geometry_reference":{"persistent_samples":reference["persistent_sample_count"],"longest_seconds":reference["longest_seconds"]}})
            dialog.close()
            settle(root)
            assert not app.busy
            print("Video",index+1,videos[-1]["persistent_samples"],videos[-1]["longest_seconds"],flush=True)

        app.show_review()
        review = next(w for w in root.winfo_children() if isinstance(w,tk.Toplevel))
        screenshot(review,"review_list.png")
        review.destroy()
        table = json.loads((ROOT/"outputs/video/현장점검_관측목록.json").read_text(encoding="utf-8"))
        assert len(table["views"])==4
        assert len({row["source_episode"] for row in table["views"]})==3

        # Real cancellation through both windows; no fake detector in acceptance.
        before = set((ROOT/"outputs/video").glob("*"))
        app.video_button.invoke()
        dialog = app.video_window
        dialog.stop.invoke()
        wait_video(root,dialog)
        assert dialog.report_path is None
        assert set((ROOT/"outputs/video").glob("*"))==before
        dialog.close()
        settle(root)
        app.video_button.invoke()
        app.close()
        deadline = time.monotonic()+30
        while time.monotonic()<deadline:
            try:
                root.update()
                exists = root.winfo_exists()
            except tk.TclError:
                exists = False
            if not exists:
                break
            time.sleep(.025)
        assert not exists,"Main close must wait for worker cleanup"
        assert set((ROOT/"outputs/video").glob("*"))==before
        report = {"model":app.detector.choice,"real_photos":photos,"real_videos":videos,"review_views":4,"independent_source_episodes":3,
                  "video_stop_verified":True,"parent_close_cleanup_verified":True,"complete":True,
                  "limitation":"XML compares same image-plane geometry, not physical driver visibility; correlated acted scenarios, not field risk accuracy."}
        (OUT/"acceptance.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print("Real trained-model acceptance complete",flush=True)
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
