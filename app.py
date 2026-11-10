"""매의 눈: crosswalk blind-spot zone setup, photo check, and live illegal-stop monitoring of the source video."""
import json
import hashlib
import queue
import threading
import tkinter as tk
from pathlib import Path
from io import BytesIO
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
from photo_analysis import ROOT, DATA, DEMO, LABELS, Detector, annotated, assess, export_result, polygon, signal_state

BG, NAVY, MUTED, TEAL = "#edf2f6", "#152a3a", "#65778a", "#087f80"
VIDEO_TYPES = (".mp4",".avi",".mov",".mkv")
ZONE_NAMES = {"entry":"사각지대","waiting":"신호대기 차로","signal":"신호등"}


class PhotoApp:
    def __init__(self, root):
        self.root = root
        root.title("매의 눈")
        root.geometry("1380x900")
        root.minsize(1120,760)
        root.configure(bg=BG)
        root.option_add("*Font", ("맑은 고딕",10))
        self.catalog = json.loads((DEMO/"manifest.json").read_text(encoding="utf-8"))
        self.profiles = json.loads((ROOT/"demo_regions.json").read_text(encoding="utf-8"))
        self.video_profiles = {profile["video"]:key for key,profile in self.profiles.items()}
        self.detector, self.messages = Detector(), queue.Queue()
        self.waiting_region = self.signal_roi = None
        self.edit_zone, self.points = "entry", []
        self.zone_note = tk.StringVar(value="횡단보도 앞 사각지대")
        self.alert_seconds = tk.DoubleVar(value=10)
        self.region = self.detections = self.path = self.image = None
        self.busy = self.closing = self.editing = False
        self.video_window = None
        root.protocol("WM_DELETE_WINDOW",self.close)
        self.review_paths = {}
        self.buttons = []
        self.status = tk.StringVar(value="사진이나 영상을 선택하세요")
        self.detail = tk.StringVar()
        self.camera = tk.StringVar()
        self.counts = tk.StringVar(value="—")
        self.footer = tk.StringVar(value=self.detector.choice["selected_model"])

        header = tk.Frame(root,bg=NAVY)
        header.pack(fill="x")
        tk.Label(header,text="매의 눈",font=("맑은 고딕",23,"bold"),bg=NAVY,fg="white").pack(anchor="w",padx=24,pady=14)
        toolbar = tk.Frame(root,bg=BG)
        toolbar.pack(fill="x",padx=20,pady=14)
        self.button(toolbar,"파일 열기",self.choose_photo).pack(side="left",padx=(0,8))
        self.button(toolbar,"① 사각지대 그리기",lambda:self.start_edit("entry")).pack(side="left",padx=(0,6))
        self.button(toolbar,"② 신호대기 차로",lambda:self.start_edit("waiting")).pack(side="left",padx=(0,6))
        self.button(toolbar,"③ 신호등(선택)",lambda:self.start_edit("signal")).pack(side="left",padx=(0,8))
        self.button(toolbar,"사진 분석",self.run_analysis).pack(side="left",padx=(0,6))
        self.save_button = self.button(toolbar,"결과 저장",lambda:self.handle(self.save_result,notify=True))
        self.save_button.pack(side="left")
        self.video_button = self.button(toolbar,"▶ 영상 감시 시작",self.open_video,primary=True)
        self.video_button.pack(side="left",padx=(14,6))
        tk.Label(toolbar,text="정차 기준",bg=BG,fg=NAVY).pack(side="left",padx=(8,2))
        tk.Spinbox(toolbar,from_=3,to=300,increment=1,width=4,textvariable=self.alert_seconds).pack(side="left")
        tk.Label(toolbar,text="초",bg=BG,fg=NAVY).pack(side="left",padx=(2,8))
        self.button(toolbar,"저장 폴더",self.open_outputs).pack(side="right")
        self.button(toolbar,"알림 기록",self.show_review).pack(side="right",padx=6)

        body = tk.Frame(root,bg=BG)
        body.pack(fill="both",expand=True,padx=20)
        left = tk.Frame(body,bg="white")
        left.pack(side="left",fill="both",expand=True,padx=(0,14))
        tk.Label(left,textvariable=self.camera,font=("맑은 고딕",13,"bold"),fg=NAVY,bg="white").pack(anchor="w",padx=16,pady=(13,6))
        self.canvas = tk.Canvas(left,bg="#101d29",highlightthickness=0)
        self.canvas.pack(fill="both",expand=True,padx=12,pady=(0,8))
        self.canvas.bind("<Configure>",lambda event:self.redraw())
        self.canvas.bind("<ButtonPress-1>",self.click)
        self.canvas.bind("<B1-Motion>",self.drag_move)
        self.canvas.bind("<ButtonRelease-1>",self.drag_end)
        self.canvas.bind("<Double-Button-1>",lambda event:self.finish_polygon())
        self.canvas.bind("<Button-3>",lambda event:self.finish_polygon())
        root.bind("<Escape>",lambda event:self.cancel_edit())
        tk.Label(left,text="청록 사각지대   보라 신호대기 차로",bg="white",fg=MUTED).pack(anchor="w",padx=16,pady=(0,10))

        side = tk.Frame(body,bg="white",width=320)
        side.pack(side="right",fill="y")
        side.pack_propagate(False)
        self.status_label = tk.Label(side,textvariable=self.status,font=("맑은 고딕",15,"bold"),bg="white",fg=NAVY,wraplength=282,justify="left")
        self.status_label.pack(fill="x",padx=18,pady=(18,10))
        tk.Label(side,textvariable=self.counts,font=("맑은 고딕",12,"bold"),bg="white",fg=TEAL,justify="left",anchor="w").pack(fill="x",padx=18,pady=6)
        tk.Label(side,textvariable=self.detail,bg="white",fg=MUTED,justify="left",anchor="nw",wraplength=282).pack(fill="x",padx=18,pady=8)
        tk.Label(side,text="구역 메모",bg="white",fg=MUTED).pack(anchor="w",padx=18)
        tk.Entry(side,textvariable=self.zone_note).pack(fill="x",padx=18,pady=4)
        tk.Frame(side,bg=BG,height=1).pack(fill="x",padx=18,pady=8)
        tk.Label(side,text="시연 영상",bg="white",fg=NAVY,font=("맑은 고딕",11,"bold")).pack(anchor="w",padx=18)
        self.listbox = tk.Listbox(side,height=10,bg="#f4f7f9",fg=NAVY,selectbackground=TEAL,highlightthickness=0,bd=0,exportselection=False)
        self.listbox.pack(fill="both",expand=True,padx=18,pady=8)
        for entry in self.catalog:
            profile = self.profiles[self.video_profiles[entry["source_video"]]]
            self.listbox.insert("end",f'{profile["name"]}  {entry["source_time_seconds"]:g}초')
        self.listbox.bind("<<ListboxSelect>>",self.select_demo)
        row = tk.Frame(side,bg="white")
        row.pack(fill="x",padx=18,pady=(0,10))
        self.button(row,"구역 설정 저장",self.save_profile).pack(side="left",padx=(0,6))
        self.button(row,"설정 불러오기",self.choose_profile).pack(side="left")
        tk.Label(root,textvariable=self.footer,bg=BG,fg=MUTED,anchor="w",font=("맑은 고딕",9)).pack(fill="x",padx=24,pady=(8,12))
        self.poll_job = self.root.after(80,self.poll)
        self.open_path(DEMO/self.catalog[0]["file"])

    def button(self,parent,text,command,primary=False):
        button = tk.Button(parent,text=text,command=command,bg=TEAL if primary else "white",fg="white" if primary else NAVY,activebackground="#d8eeec",bd=0,padx=12,pady=8,cursor="hand2")
        self.buttons.append(button)
        return button

    def handle(self,action,notify=False):
        try:
            result = action()
            if notify:
                self.footer.set(f"저장함: {result[0].parent}")
            return result
        except Exception as error:
            messagebox.showerror("확인 필요",str(error),parent=self.root)

    def choose_photo(self):
        path = filedialog.askopenfilename(title="사진 또는 영상 선택",filetypes=[("사진, 영상","*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv")])
        if path:
            self.handle(lambda:self.open_path(path))

    def open_path(self,path):
        if self.busy:
            raise ValueError("현재 분석이 끝난 뒤 선택하세요.")
        path = Path(path).resolve()
        source_bytes = path.read_bytes()
        if path.suffix.lower() in VIDEO_TYPES:
            import cv2
            import numpy as np
            capture = cv2.VideoCapture(str(path))
            ok,frame = capture.read()
            capture.release()
            if not ok:
                raise ValueError("영상의 첫 프레임을 읽을 수 없습니다.")
            image = Image.fromarray(cv2.cvtColor(np.asarray(frame),cv2.COLOR_BGR2RGB))
        else:
            with Image.open(BytesIO(source_bytes)) as source:
                image = source.convert("RGB")
        self.path,self.image = path,image
        self.source_sha = hashlib.sha256(source_bytes).hexdigest()
        self.detections,self.region,self.editing,self.points = None,None,False,[]
        self.waiting_region = self.signal_roi = None
        self.zone_note.set("횡단보도 앞 사각지대")
        self.metadata = {"camera":"새 영상" if path.suffix.lower() in VIDEO_TYPES else "새 사진","source":"사용자가 불러온 파일"}
        if path.suffix.lower() in VIDEO_TYPES:
            self.metadata["video_path"] = str(path)
        for entry in self.catalog:
            if path == (DEMO/entry["file"]).resolve():
                profile = self.profiles[self.video_profiles[entry["source_video"]]]
                self.region = profile["region"]
                self.waiting_region = profile.get("waiting_region")
                self.signal_roi = profile.get("signal_roi")
                self.zone_note.set(profile["note"])
                self.metadata = {**entry,"camera":profile["name"],"source":"AI-Hub 169","region_note":profile["note"]}
                break
        self.camera.set(self.metadata["camera"])
        self.footer.set(self.metadata["source"]+" | "+self.detector.choice["selected_model"])
        self.refresh()

    def select_demo(self,event=None):
        selection = self.listbox.curselection()
        if selection and not self.busy:
            self.handle(lambda:self.open_path(DEMO/self.catalog[selection[0]]["file"]))

    def apply_region(self,region,zone="entry"):
        if self.busy:
            raise ValueError("분석 중에는 구역을 변경할 수 없습니다.")
        if zone=="signal":
            x1,y1,x2,y2 = region
            if x2<=x1 or y2<=y1:
                raise ValueError("신호등 상자의 넓이가 0입니다.")
            self.signal_roi = list(region)
        else:
            points = polygon(region)
            if zone=="waiting":
                self.waiting_region = points
            else:
                self.region = points
        self.detections = None
        self.metadata.pop("region_note",None)
        self.refresh()

    def start_edit(self,zone="entry"):
        if self.busy or self.image is None:
            return
        self.edit_zone,self.editing,self.points = zone,True,[]
        if zone=="signal":
            self.status.set("신호등을 드래그해 상자로 지정하세요")
            self.detail.set("")
        else:
            self.status.set(f"{ZONE_NAMES[zone]} 꼭짓점을 차례로 클릭하세요")
            self.detail.set("더블클릭이나 우클릭으로 완성, Esc는 취소")

    def cancel_edit(self):
        self.editing,self.points = False,[]
        self.refresh()

    def to_normalized(self,x,y):
        ox,oy,width,height = self.display_box
        return max(0,min(1,(x-ox)/width)),max(0,min(1,(y-oy)/height))

    def to_canvas(self,point):
        ox,oy,width,height = self.display_box
        return ox+point[0]*width,oy+point[1]*height

    def click(self,event):
        if not self.editing or self.busy:
            return
        if self.edit_zone=="signal":
            self.anchor = self.to_normalized(event.x,event.y)
            self.drag_id = self.canvas.create_rectangle(event.x,event.y,event.x,event.y,outline="#ffd24a",width=3)
            return
        point = self.to_normalized(event.x,event.y)
        if self.points and self.points[-1]==point:
            return  # second click of a double-click
        self.points.append(point)
        self.canvas.delete("draft")
        coords = [c for p in self.points for c in self.to_canvas(p)]
        for p in self.points:
            x,y = self.to_canvas(p)
            self.canvas.create_oval(x-4,y-4,x+4,y+4,fill="#20d9c2",outline="",tags="draft")
        if len(self.points)>1:
            self.canvas.create_line(*coords,fill="#20d9c2",width=3,tags="draft")

    def finish_polygon(self):
        if self.editing and self.edit_zone!="signal":
            points,self.points,self.editing = [list(p) for p in self.points],[],False
            if len(points)<3:
                self.refresh()
                self.status.set("꼭짓점이 3개 이상 필요합니다")
                return
            self.handle(lambda:self.apply_region(points,self.edit_zone))

    def drag_move(self,event):
        if self.editing and self.edit_zone=="signal" and hasattr(self,"drag_id"):
            ox,oy,w,h = self.display_box
            x,y = self.anchor
            ex,ey = self.to_normalized(event.x,event.y)
            self.canvas.coords(self.drag_id,ox+x*w,oy+y*h,ox+ex*w,oy+ey*h)

    def drag_end(self,event):
        if self.editing and self.edit_zone=="signal" and hasattr(self,"drag_id"):
            x,y = self.anchor
            ex,ey = self.to_normalized(event.x,event.y)
            self.editing = False
            del self.drag_id
            self.handle(lambda:self.apply_region([min(x,ex),min(y,ey),max(x,ex),max(y,ey)],"signal"))

    def refresh(self):
        signal = signal_state(self.image,self.signal_roi) if self.image is not None else None
        signal_text = {"red":"신호등 빨간불","green":"신호등 초록불","unknown":"신호등 판독 불가",None:"신호등 미지정"}[signal]
        waiting_text = "신호대기 차로 지정" if self.waiting_region else "신호대기 차로 미지정"
        if self.detections is None:
            self.status.set("구역 지정 완료" if self.region else LABELS["region_required"])
            self.counts.set("사각지대 차량  —")
            self.status_label.configure(fg=NAVY)
        else:
            result = assess(self.detections,self.region,self.waiting_region)
            self.status.set(LABELS[result["status"]])
            self.status_label.configure(fg="#c64c3b" if result["candidate_count"] else TEAL)
            self.counts.set(f'사각지대 차량  {result["candidate_count"]}대\n신호대기 차로  {result["waiting_count"]}대')
        self.detail.set(waiting_text+"\n"+signal_text)
        self.save_button.configure(state="normal" if self.detections is not None and self.region else "disabled")
        self.video_button.configure(state="normal" if self.registered_video() and self.region and not self.busy else "disabled")
        self.redraw()

    def redraw(self):
        if self.image is None:
            return
        signal = signal_state(self.image,self.signal_roi)
        image = annotated(self.image,self.region,self.detections or [],self.waiting_region,signal_roi=self.signal_roi,signal=signal)
        width,height = max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        image.thumbnail((width,height),Image.Resampling.LANCZOS)
        ox,oy = (width-image.width)//2,(height-image.height)//2
        self.display_box = (ox,oy,image.width,image.height)
        self.tkimage = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(ox,oy,image=self.tkimage,anchor="nw")

    def run_analysis(self):
        if self.busy:
            return
        if self.region is None:
            self.status.set(LABELS["region_required"])
            return
        try:
            self.check_source()
        except (OSError,ValueError) as error:
            self.status.set("파일이 바뀜. 다시 여세요")
            self.detail.set(str(error))
            return
        self.busy,self.editing = True,False
        self.detections = None
        for button in self.buttons:
            button.configure(state="disabled")
        self.listbox.configure(state="disabled")
        self.status.set("사진 분석 중…")
        self.detail.set("차량과 사람의 위치를 찾고 있습니다.\n첫 실행은 모델 준비로 잠시 걸립니다.")
        path = self.path
        snapshot = self.image.copy()
        def worker():
            try:
                self.messages.put((self.detector.predict(path,snapshot=snapshot),None))
            except Exception as error:
                self.messages.put((None,str(error)))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        try:
            detections,error = self.messages.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            for button in self.buttons:
                button.configure(state="normal")
            self.listbox.configure(state="normal")
            self.detections = detections
            self.refresh()
            if error:
                self.status.set("분석 실패")
                self.detail.set(error)
        if not self.closing:
            self.poll_job = self.root.after(80,self.poll)

    def save_result(self,folder=None):
        if self.detections is None or self.region is None:
            raise ValueError("사각지대를 지정하고 사진 분석을 완료하세요.")
        self.check_source()
        self.metadata["region_note"] = self.zone_note.get()
        return export_result(self.path,self.region,self.detections,self.metadata,folder or ROOT/"outputs",expected_source_sha=self.source_sha,model_choice=self.detector.choice,waiting_region=self.waiting_region)

    def check_source(self):
        try:
            current = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if current != self.source_sha:
                raise ValueError("파일이 변경되었습니다. 다시 열고 분석하세요.")
        except (OSError,ValueError):
            self.detections = None
            self.refresh()
            raise

    def save_profile(self):
        if self.region is None:
            return
        path = filedialog.asksaveasfilename(title="카메라별 구역 저장",defaultextension=".json",filetypes=[("설정","*.json")])
        if path:
            self.handle(lambda:Path(path).write_text(json.dumps({"camera":self.camera.get(),"image_size":list(self.image.size),"region":self.region,"waiting_region":self.waiting_region,
                                                                  "signal_roi":self.signal_roi,"region_note":self.zone_note.get(),"alert_seconds":self.alert_seconds.get()},ensure_ascii=False,indent=2),encoding="utf-8"))

    def load_profile(self,path,confirmed):
        if not confirmed:
            raise ValueError("같은 카메라와 구도인지 확인해야 합니다.")
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
        if profile["image_size"] != list(self.image.size):
            raise ValueError("설정의 원본 해상도가 현재 화면과 다릅니다. 구역을 새로 지정하세요.")
        assess([],profile["region"],profile.get("waiting_region"))
        self.waiting_region = profile.get("waiting_region") and polygon(profile["waiting_region"])
        self.signal_roi = profile.get("signal_roi")
        self.zone_note.set(profile.get("region_note","저장된 구역"))
        self.alert_seconds.set(profile.get("alert_seconds",self.alert_seconds.get()))
        self.apply_region(profile["region"])
        self.metadata["camera"] = profile["camera"]
        self.camera.set(profile["camera"])

    def choose_profile(self):
        path = filedialog.askopenfilename(title="구역 설정",filetypes=[("설정","*.json")])
        if path:
            confirmed = messagebox.askyesno("카메라 확인","현재 화면이 이 설정과 같은 카메라와 촬영 구도인가요? 해상도가 같아도 다른 구도이면 적용할 수 없습니다.",parent=self.root)
            if confirmed:
                self.handle(lambda:self.load_profile(path,True))

    def open_outputs(self):
        import os
        (ROOT/"outputs").mkdir(exist_ok=True)
        os.startfile(ROOT/"outputs")

    def registered_video(self):
        if self.metadata.get("video_path"):
            return Path(self.metadata["video_path"])
        video = self.metadata.get("source_video")
        if not video:
            return None
        return DATA/f'aihub169_suddenlyappear_{self.metadata["source_split"]}'/video

    def open_video(self):
        path = self.registered_video()
        if self.busy or path is None or self.region is None:
            return
        try:
            seconds = float(self.alert_seconds.get())
            if not 0<seconds<=3600:
                raise ValueError
        except (tk.TclError,ValueError):
            self.status.set("정차 기준은 1~3600초 숫자로 입력하세요")
            return
        from video_window import VideoWindow
        self.busy = True
        for button in self.buttons:
            button.configure(state="disabled")
        self.listbox.configure(state="disabled")
        metadata = dict(self.metadata,region_note=self.zone_note.get())
        def saved(report_path):
            self.review_paths[str(path)] = report_path
        def closed():
            self.video_window = None
            if self.closing:
                self.root.destroy()
                return
            self.busy = False
            for button in self.buttons:
                button.configure(state="normal")
            self.listbox.configure(state="normal")
            self.refresh()
        self.video_window = VideoWindow(self.root,self.detector,path,self.region,metadata,saved,closed,waiting_region=self.waiting_region,
                                        alert_seconds=seconds,signal_roi=self.signal_roi)

    def close(self):
        self.closing = True
        if self.poll_job is not None:
            self.root.after_cancel(self.poll_job)
            self.poll_job = None
        if self.video_window is not None and self.video_window.window.winfo_exists():
            self.video_window.close()
        else:
            self.root.destroy()

    def show_review(self):
        if not self.review_paths:
            messagebox.showinfo("알림 기록","먼저 ▶ 영상 감시를 실행하세요. 알림 사진과 기록은 저장 폴더의 video 폴더에도 남습니다.",parent=self.root)
            return
        from video_analysis import export_review_table
        records = self.handle(lambda:export_review_table(self.review_paths.values(),ROOT/"outputs/video"))
        if records is None:
            return
        from tkinter import ttk
        window = tk.Toplevel(self.root)
        window.title("매의 눈 | 알림 기록")
        window.geometry("1000x420")
        tk.Label(window,text="영상별 불법 정차 알림",font=("맑은 고딕",18,"bold"),padx=18,pady=16).pack(anchor="w")
        columns = (("camera","카메라"),("video","영상"),("alerts","알림 수"),("first","첫 알림(초)"),("longest","최장 사각지대 정차(초)"))
        table = ttk.Treeview(window,columns=[c for c,_ in columns],show="headings",height=8)
        for key,label in columns:
            table.heading(key,text=label)
            table.column(key,width=180,anchor="center")
        table.pack(fill="both",expand=True,padx=18)
        for record in records:
            first = "—" if record["first_alert_seconds"] is None else f'{record["first_alert_seconds"]:.1f}'
            table.insert("","end",values=(record["camera"],record["video"],record["alert_count"],first,f'{record["longest_dwell_seconds"]:.0f}'))
        tk.Label(window,text="저장 위치: outputs/video",justify="left",padx=18,pady=14,fg=MUTED).pack(anchor="w")


if __name__ == "__main__":
    root = tk.Tk()
    PhotoApp(root)
    root.mainloop()
