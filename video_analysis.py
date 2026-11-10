"""Run the illegal-stop monitor over a video file and export alerts with evidence snapshots."""
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
from PIL import Image
from photo_analysis import annotated, signal_state
from temporal import StopMonitor


def analyze_video(path,region,detector,progress=None,cancel=None,waiting_region=None,alert_seconds=10,sample_fps=5,signal_roi=None,on_frame=None):
    """on_frame(image, detections, state) is called for every analysed sample (live view, beeps)."""
    path = Path(path)
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monitor = StopMonitor(region,waiting_region,alert_seconds)
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or not math.isfinite(fps) or fps<=0 or declared<=0:
        capture.release()
        raise ValueError("영상을 열거나 촬영 시각을 확인할 수 없습니다.")
    step = max(1,round(fps/sample_fps))
    index,samples = 0,[]
    try:
        while capture.grab():
            if cancel is not None and cancel.is_set():
                raise InterruptedError("영상 분석을 중단했습니다. 결과는 저장하지 않았습니다.")
            if index%step==0:
                ok,frame = capture.retrieve()
                if not ok:
                    raise ValueError("영상 프레임을 읽지 못했습니다.")
                image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                detections = detector.predict(path,snapshot=image)
                signal = signal_state(image,signal_roi)
                state = monitor.update(index/fps,detections,signal)
                samples.append({"frame":index,"time":index/fps,"detections":detections,"tracks":state["tracks"],"signal":signal})
                if on_frame:
                    on_frame(image,detections,state)
                if progress:
                    progress(index+1,declared)
            index += 1
    finally:
        capture.release()
    if index<declared*.98:
        raise ValueError(f"영상이 불완전합니다: {index}/{declared} 프레임")
    if cancel is not None and cancel.is_set():
        raise InterruptedError("영상 분석을 중단했습니다.")
    return {**monitor.finish(),"source":str(path.resolve()),"source_sha256":original_sha,"complete":True,"fps":fps,"decoded_frames":index,
            "source_duration_seconds":index/fps,"sample_fps":fps/step,"region":region,"waiting_region":waiting_region,"signal_roi":signal_roi,
            "inference_model":dict(detector.choice),"samples_detail":samples}


def alert_banner(alert):
    return f'불법 정차 의심: 차량 #{alert["track"]}, 사각지대에 {alert["alert_dwell_seconds"]:.0f}초 정차 (영상 {alert["alert_seconds"]:.1f}초)'


def export_video_report(report,metadata,folder,cancel=None):
    """JSON report + one annotated snapshot per alert (the frame where the alert fired) + rows in 알림_기록.csv."""
    if not report["complete"]:
        raise ValueError("완료된 영상 분석만 저장할 수 있습니다.")
    source = Path(report["source"])
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    record_id = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid4().hex[:8]
    path = folder/(record_id+".json")
    by_time = {s["time"]:s for s in report["samples_detail"]}
    images = []
    capture = cv2.VideoCapture(str(source))
    try:
        for alert in report["alerts"]:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("근거 저장을 중단했습니다.")
            sample = by_time[alert["alert_seconds"]]
            capture.set(cv2.CAP_PROP_POS_FRAMES,sample["frame"])
            ok,frame = capture.read()
            if not ok:
                raise ValueError("알림 시점 프레임을 읽지 못했습니다.")
            image_path = folder/f'{record_id}_alert{alert["alert"]}.png'
            annotated(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),report["region"],sample["detections"],report.get("waiting_region"),
                      tracks=sample["tracks"],signal_roi=report.get("signal_roi"),signal=sample["signal"],banner=alert_banner(alert)).save(image_path)
            images.append(str(image_path))
        data = {**report,"metadata":metadata,"alert_images":images}
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
        if cancel is not None and cancel.is_set():
            raise InterruptedError("근거 저장을 중단했습니다.")
        journal = folder/"알림_기록.csv"
        new = not journal.exists()
        with journal.open("a",encoding="utf-8-sig",newline="") as file:
            writer = csv.writer(file)
            if new:
                writer.writerow(["기록ID","카메라","영상","알림번호","차량ID","정차시작(초)","알림시각(초)","마지막관측(초)","사각지대정차_총(초)","근거사진"])
            for alert,image in zip(report["alerts"],images):
                writer.writerow([record_id,metadata.get("camera",""),source.name,alert["alert"],alert["track"],f'{alert["stopped_since_seconds"]:.1f}',
                                 f'{alert["alert_seconds"]:.1f}',f'{alert["end_seconds"]:.1f}',f'{alert["dwell_seconds"]:.1f}',image])
    except Exception:
        for image in images:
            Path(image).unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
    return path


def export_review_table(reports,folder):
    """One row per analysed video, for the 알림 기록 window. Only reports made with the same model and rules are listed together."""
    records,signature = [],None
    for path in dict.fromkeys(Path(p).resolve() for p in reports):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data["inference_model"]
        current = (data["rules"],model["weights_sha256"],model["confidence"],model["image_size"])
        if signature is not None and current!=signature:
            raise ValueError("모델과 분석 설정이 같은 결과만 함께 볼 수 있습니다.")
        signature = current
        records.append({"camera":data["metadata"]["camera"],"video":Path(data["source"]).name,"duration":data["source_duration_seconds"],
                        "alert_count":data["alert_count"],"longest_dwell_seconds":data["longest_dwell_seconds"],
                        "first_alert_seconds":min((a["alert_seconds"] for a in data["alerts"]),default=None),"waiting_configured":data["waiting_configured"],"evidence_json":str(path)})
    records.sort(key=lambda r:r["camera"])
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"영상별_알림_요약.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    return records
