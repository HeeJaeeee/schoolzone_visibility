"""Does the 'not signal-waiting' rule change decisions on real footage?

Held-out test clip 12:41 (주간 교차로 B): three cars queue at the far stop line at 8.4 / 14.4 / 18.4 s.
Zone drawn over that stop-line approach, monitored with and without the waiting lane, on human labels and on the model.
Also logs the HSV colour of the visible signal heads over time.
python evaluate_waiting.py
"""
import json

import cv2
from PIL import Image

from evaluate_alerts import SOURCE, reference_alerts
from photo_analysis import ROOT, Detector, signal_state
from video_analysis import analyze_video

VIDEO = next(SOURCE.glob("aihub169_suddenlyappear_*/2020_11_10_12_41_suddenlyappear_sun_A_2.mp4"))
APPROACH = [[0.04,0.09],[0.36,0.09],[0.36,0.24],[0.04,0.24]]  # crosswalk approach incl. the stop-line lane
LANE = [[0.04,0.10],[0.36,0.10],[0.36,0.215],[0.04,0.215]]     # the stop-line lane where the queue forms
LIGHTS = {"보행자 신호(좌)":[0.276,0.135,0.305,0.205],"차량 신호(우상)":[0.672,0.04,0.688,0.08]}


def run(waiting,detector):
    profile = {"region":APPROACH,"waiting_region":waiting}
    labels = reference_alerts(VIDEO,profile,10)
    model = analyze_video(VIDEO,APPROACH,detector,waiting_region=waiting,alert_seconds=10)["alerts"]
    brief = lambda alerts:[{"track":a["track"],"stopped_since":round(a["stopped_since_seconds"],1),"alert_at":round(a["alert_seconds"],1),"box":[round(v,3) for v in a["box"]]} for a in alerts]
    return {"label_alerts":brief(labels),"model_alerts":brief(model)}


if __name__=="__main__":
    detector = Detector()
    result = {"video":VIDEO.name,"zone":APPROACH,"waiting_lane":LANE,"without_waiting_lane":run(None,detector),"with_waiting_lane":run(LANE,detector)}
    capture,states = cv2.VideoCapture(str(VIDEO)),{name:[] for name in LIGHTS}
    for second in range(30):
        capture.set(cv2.CAP_PROP_POS_FRAMES,second*15)
        ok,frame = capture.read()
        image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        for name,roi in LIGHTS.items():
            states[name].append(signal_state(image,roi))
    capture.release()
    result["signal_states_per_second"] = states
    (ROOT/"evidence/v4_eval/waiting_rule_12_41.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for key in ("without_waiting_lane","with_waiting_lane"):
        print(key,"labels:",len(result[key]["label_alerts"]),"model:",len(result[key]["model_alerts"]),result[key])
    for name,s in states.items():
        print(name,"".join({"red":"R","green":"G","unknown":"?"}[x] for x in s))
