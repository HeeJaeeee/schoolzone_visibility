"""End-to-end acceptance with the real Tk app and the real v4 model; also captures the screenshots used in the deck."""
import ctypes
import json
import time
import tkinter as tk

from PIL import ImageGrab
from app import PhotoApp
from photo_analysis import ROOT, DEMO

OUT = ROOT/"evidence/v4_app"
OUT.mkdir(parents=True,exist_ok=True)


def settle(root,seconds=.2):
    end = time.monotonic()+seconds
    while time.monotonic()<end:
        root.update()
        time.sleep(.015)


def capture(root,window,name):
    window.lift()
    settle(root,.4)
    hwnd = ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
    ImageGrab.grab(window=hwnd).save(OUT/name)


def press(app,label):
    next(b for b in app.buttons if b["text"]==label).invoke()


def wait(root,condition,seconds,what):
    deadline = time.monotonic()+seconds
    while not condition():
        assert time.monotonic()<deadline,"timeout: "+what
        root.update()
        time.sleep(.02)


def demo(app,video,second):
    index = next(i for i,e in enumerate(app.catalog) if e["source_video"]==video and e["source_time_seconds"]==second)
    app.listbox.selection_clear(0,"end")
    app.listbox.selection_set(index)
    app.select_demo()


def monitor(root,app,shot=None):
    press(app,"▶ 영상 감시 시작")
    window = app.video_window
    if shot:
        wait(root,lambda:window.table.get_children(),90,"first alert")
        settle(root,2.5)
        capture(root,window.window,shot)
    wait(root,lambda:not window.running,180,"monitor finished")
    settle(root,.5)
    report = json.loads(window.report_path.read_text(encoding="utf-8"))
    return window,report


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoApp(root)
    root.geometry("1380x900+20+20")
    settle(root)
    result = {}
    try:
        night = json.loads((ROOT/"demo_regions.json").read_text(encoding="utf-8"))["night_A"]["video"]
        demo(app,night,10)
        press(app,"사진 분석")
        wait(root,lambda:not app.busy,120,"photo analysis")
        assert app.detections is not None and "사각지대 안에 차량" in app.status.get(),app.status.get()
        capture(root,root,"app_photo_night_A.png")
        result["photo_status"] = app.status.get()

        window,report = monitor(root,app,"monitor_alert_night_A.png")
        assert report["alert_count"]==1,report["alerts"]
        capture(root,window.window,"monitor_done_night_A.png")
        result["night_A_alerts"] = report["alerts"]
        result["night_A_alert_image"] = report["alert_images"][0]
        window.close()
        settle(root,.5)

        control = json.loads((ROOT/"demo_regions.json").read_text(encoding="utf-8"))["pass_B"]["video"]
        demo(app,control,12)
        window,report = monitor(root,app)
        assert report["alert_count"]==0,report["alerts"]
        capture(root,window.window,"monitor_control_pass_B.png")
        result["pass_B_alerts"] = 0
        window.close()
        settle(root,.5)

        # Draw a polygon with real pointer events, then confirm the zone and the monitor button follow it.
        app.start_edit("entry")
        ox,oy,width,height = app.display_box
        for x,y in ((.2,.3),(.6,.3),(.6,.7),(.2,.7)):
            app.canvas.event_generate("<ButtonPress-1>",x=int(ox+x*width),y=int(oy+y*height))
            app.canvas.event_generate("<ButtonRelease-1>",x=int(ox+x*width),y=int(oy+y*height))
            root.update()
        app.canvas.event_generate("<Button-3>",x=int(ox+.2*width),y=int(oy+.7*height))
        root.update()
        assert len(app.region)==4 and all(abs(a-b)<.01 for p,q in zip(app.region,((.2,.3),(.6,.3),(.6,.7),(.2,.7))) for a,b in zip(p,q)),app.region
        assert str(app.video_button["state"])=="normal"
        capture(root,root,"app_polygon_drawn.png")
        result["polygon_drawn_by_pointer"] = True
        (OUT/"acceptance.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        root.destroy()
