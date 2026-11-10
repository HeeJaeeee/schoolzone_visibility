"""Live monitoring window: plays the video at real speed with tracked vehicles, stop timers and alerts."""
import json
import queue
import threading
import time
import tkinter as tk
import winsound
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk
from photo_analysis import ROOT, annotated
from video_analysis import alert_banner, analyze_video, export_video_report


class VideoWindow:
    def __init__(self,parent,detector,path,region,metadata,on_saved,on_close,waiting_region=None,alert_seconds=10,signal_roi=None):
        self.window = tk.Toplevel(parent)
        self.window.title("매의 눈 | 영상 감시")
        width = min(1280,self.window.winfo_screenwidth()-60,int((self.window.winfo_screenheight()-330)*16/9))
        self.size = (width,width*9//16)
        self.window.geometry(f"{width+40}x{self.size[1]+300}+10+10")
        self.window.configure(bg="#edf2f6")
        self.detector,self.path,self.region,self.metadata = detector,Path(path),region,dict(metadata)
        self.waiting_region,self.alert_seconds,self.signal_roi = waiting_region,alert_seconds,signal_roi
        self.on_saved,self.on_close = on_saved,on_close
        self.cancel,self.messages = threading.Event(),queue.Queue()
        self.running,self.closing,self.report_path = True,False,None
        self.status = tk.StringVar(value="감시 준비 중…")
        self.window.protocol("WM_DELETE_WINDOW",self.close)
        top = tk.Frame(self.window,bg="#152a3a")
        top.pack(fill="x")
        tk.Label(top,text="매의 눈",font=("맑은 고딕",20,"bold"),bg="#152a3a",fg="white").pack(side="left",padx=20,pady=12)
        tk.Label(top,text=f'{metadata["camera"]}   정차 기준 {alert_seconds:g}초',font=("맑은 고딕",12),bg="#152a3a",fg="#b8ccd9").pack(side="left")
        self.banner = tk.Label(self.window,text="",font=("맑은 고딕",16,"bold"),bg="#edf2f6",fg="white",pady=6)
        self.banner.pack(fill="x",padx=20,pady=(8,0))
        self.view = tk.Label(self.window,bg="#101d29")
        self.view.pack(padx=20,pady=8)
        row = tk.Frame(self.window,bg="#edf2f6")
        row.pack(fill="x",padx=20)
        tk.Label(row,textvariable=self.status,bg="#edf2f6",fg="#087f80",font=("맑은 고딕",12,"bold")).pack(side="left")
        self.folder_button = tk.Button(row,text="근거 폴더",command=self.open_folder,bg="white",bd=0,padx=14,pady=6,state="disabled")
        self.folder_button.pack(side="right")
        self.stop = tk.Button(row,text="감시 중단",command=self.cancel.set,bg="white",bd=0,padx=14,pady=6)
        self.stop.pack(side="right",padx=6)
        self.progress = ttk.Progressbar(self.window,mode="determinate",maximum=100)
        self.progress.pack(fill="x",padx=20,pady=6)
        self.table = ttk.Treeview(self.window,columns=("no","track","since","at","dwell"),show="headings",height=4)
        for key,label in zip(("no","track","since","at","dwell"),("알림","차량 ID","정차 시작(초)","알림 시각(초)","사각지대 정차(초)")):
            self.table.heading(key,text=label)
            self.table.column(key,width=200,anchor="center")
        self.table.pack(fill="x",padx=20,pady=(0,12))
        threading.Thread(target=self.worker,daemon=True).start()
        self.poll_job = self.window.after(30,self.poll)

    def worker(self):
        start = time.perf_counter()
        def on_frame(image,detections,state):
            small = image.resize(self.size,Image.Resampling.BILINEAR)
            frame = annotated(small,self.region,detections,self.waiting_region,tracks=state["tracks"],signal_roi=self.signal_roi,signal=state["signal"])
            self.messages.put(("frame",(frame,state)))
            delay = state["time"]-(time.perf_counter()-start)  # real-time playback; slower hardware just runs behind
            if delay>0 and not self.cancel.is_set():
                time.sleep(delay)
        try:
            report = analyze_video(self.path,self.region,self.detector,progress=lambda done,total:self.messages.put(("progress",done/total)),cancel=self.cancel,
                                   waiting_region=self.waiting_region,alert_seconds=self.alert_seconds,signal_roi=self.signal_roi,on_frame=on_frame)
            self.messages.put(("done",export_video_report(report,self.metadata,ROOT/"outputs/video",cancel=self.cancel)))
        except Exception as error:
            self.messages.put(("error",str(error)))

    def poll(self):
        latest = None
        while True:
            try:
                kind,data = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind=="frame":
                latest = data
                for alert in data[1]["new_alerts"]:
                    self.table.insert("","end",values=(alert["alert"],f'#{alert["track"]}',f'{alert["stopped_since_seconds"]:.1f}',f'{alert["alert_seconds"]:.1f}',f'{alert["dwell_seconds"]:.0f}'))
                    self.banner.configure(text="⚠ "+alert_banner(alert),bg="#c62828")
                    winsound.PlaySound("SystemExclamation",winsound.SND_ALIAS|winsound.SND_ASYNC)
            elif kind=="progress":
                self.progress["value"] = data*100
            else:
                self.finish(kind,data)
                if self.closing:
                    self.window.destroy()
                    self.on_close()
                    return
        if latest:
            frame,state = latest
            self.tkimage = ImageTk.PhotoImage(frame)
            self.view.configure(image=self.tkimage)
            in_zone = sum(t["zone"]=="entry" for t in state["tracks"])
            alerting = sum(t["alert"] for t in state["tracks"])
            self.status.set(f'영상 {state["time"]:.1f}초   사각지대 차량 {in_zone}대   알림 {alerting}대')
            if not alerting and self.banner["bg"]=="#c62828":
                self.banner.configure(text="알림 차량이 떠났습니다",bg="#7a8a99")
        self.poll_job = self.window.after(30,self.poll)

    def finish(self,kind,data):
        self.running = False
        self.stop.configure(state="disabled")
        if kind=="error":
            self.status.set("감시 중단: "+data)
            return
        self.report_path = Path(data)
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.status.set(f'감시 끝. 알림 {report["alert_count"]}건, 사진과 기록 저장함')
        self.folder_button.configure(state="normal")
        self.on_saved(self.report_path)

    def open_folder(self):
        import os
        os.startfile(self.report_path.parent)

    def close(self):
        if self.running:
            self.cancel.set()
            self.closing = True
            self.status.set("중단 요청 처리 중…")
        else:
            self.window.after_cancel(self.poll_job)
            self.window.destroy()
            self.on_close()
