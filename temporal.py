"""Illegal-stop monitor: track vehicles across samples, time how long each one is stopped in the blind-spot
zone, and raise an alert when that time reaches the threshold.

Time counts toward an alert only while the vehicle is
  1. stopped     - its centre stays within `still` x (box size) of where it stopped,
  2. in the zone - its ground point (box bottom-centre) is inside the blind-spot polygon,
  3. not waiting - it is not in the signal-waiting lane, and the light (if a light box is set) is not red.
A vehicle missed by the detector for up to `max_gap` seconds keeps its track and its timer.
"""
import math

from photo_analysis import assess


def box_iou(a,b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0


def overlap_of_smaller(a,b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    smaller = min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
    return intersection/smaller if smaller else 0


class StopMonitor:
    def __init__(self,region,waiting_region=None,alert_seconds=10,still=.2,match_iou=.3,max_gap=2.0):
        if region is None:
            raise ValueError("사각지대 구역이 필요합니다.")
        assess([],region,waiting_region)
        if not (math.isfinite(alert_seconds) and alert_seconds>0):
            raise ValueError("알림 기준 시간은 0보다 커야 합니다.")
        self.region,self.waiting_region = region,waiting_region
        self.rules = {"alert_seconds":alert_seconds,"still":still,"match_iou":match_iou,"max_gap_seconds":max_gap}
        self.tracks,self.alerts,self.timeline = [],[],[]
        self.next_id = 1

    def update(self,time,detections,signal=None):
        if not math.isfinite(time) or time<0 or (self.timeline and time<=self.timeline[-1]["time"]):
            raise ValueError("영상 시각은 유한하고 증가해야 합니다.")
        vehicles = [d for d in detections if d["kind"]=="vehicle"]
        zones = [z["zone"] for z in assess(vehicles,self.region,self.waiting_region)["zones"]]
        for track in [t for t in self.tracks if time-t["seen"]>self.rules["max_gap_seconds"]]:
            self.tracks.remove(track)
        pairs = sorted(((box_iou(t["box"],v["box"]),i,j) for i,t in enumerate(self.tracks) for j,v in enumerate(vehicles)),reverse=True)
        used_tracks,used_vehicles,new_alerts = set(),set(),[]
        for overlap,i,j in pairs:
            if overlap<self.rules["match_iou"]:
                break
            if i in used_tracks or j in used_vehicles:
                continue
            used_tracks.add(i)
            used_vehicles.add(j)
            alert = self.step(self.tracks[i],time,vehicles[j]["box"],zones[j],signal)
            if alert:
                new_alerts.append(alert)
        for j,vehicle in enumerate(vehicles):
            if j not in used_vehicles:
                self.tracks.append({"id":self.next_id,"box":list(vehicle["box"]),"anchor":list(vehicle["box"]),"zone":zones[j],
                                    "seen":time,"stop_start":time,"dwell":0.0,"alert":None})
                self.next_id += 1
        tracks = [{"id":t["id"],"box":t["box"],"zone":t["zone"],"stopped":time-t["stop_start"] if t["seen"]==time else 0,
                   "dwell":t["dwell"],"alert":t["alert"] is not None} for t in self.tracks if t["seen"]==time]
        self.timeline.append({"time":time,"signal":signal,"in_zone":sum(t["zone"]=="entry" for t in tracks),
                              "alerting":sum(t["alert"] for t in tracks)})
        return {"time":time,"tracks":tracks,"new_alerts":new_alerts,"signal":signal}

    def step(self,track,time,box,zone,signal):
        dt = time-track["seen"]
        anchor = track["anchor"]
        size = max(anchor[2]-anchor[0],anchor[3]-anchor[1])
        moved = math.hypot((box[0]+box[2]-anchor[0]-anchor[2])/2,(box[1]+box[3]-anchor[1]-anchor[3])/2)
        track["box"],track["zone"],track["seen"] = list(box),zone,time
        if moved>self.rules["still"]*size:
            # Drove off (or crept a car length): the stop is over, a new one starts here.
            track.update(anchor=list(box),stop_start=time,dwell=0.0,alert=None)
            return None
        if zone=="entry" and signal!="red":
            track["dwell"] += dt
        if track["alert"]:
            track["alert"].update(end_seconds=time,dwell_seconds=max(track["dwell"],track["alert"]["dwell_seconds"]))
        elif track["dwell"]>=self.rules["alert_seconds"]:
            # A second, partial box on a vehicle that is already alerting is the same vehicle: join its alert.
            same = next((t for t in self.tracks if t is not track and t["alert"] and overlap_of_smaller(t["box"],box)>=.6),None)
            if same:
                track["alert"] = same["alert"]
                return None
            # alert_dwell_seconds: counted time when the alert fired (fixed); dwell_seconds: total, grows until the vehicle leaves.
            track["alert"] = {"alert":len(self.alerts)+1,"track":track["id"],"stopped_since_seconds":track["stop_start"],"alert_seconds":time,
                              "end_seconds":time,"alert_dwell_seconds":track["dwell"],"dwell_seconds":track["dwell"],"box":list(box)}
            self.alerts.append(track["alert"])
            return track["alert"]
        return None

    def finish(self):
        return {"alerts":[dict(a) for a in self.alerts],"alert_count":len(self.alerts),
                "longest_dwell_seconds":max((a["dwell_seconds"] for a in self.alerts),default=0),
                "samples":len(self.timeline),"timeline":list(self.timeline),"rules":dict(self.rules),
                "waiting_configured":self.waiting_region is not None}
