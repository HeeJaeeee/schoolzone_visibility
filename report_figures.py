"""Figures for the v4 report/PPT from saved evidence (no inference here).

python report_figures.py <final_run> <alerts_json>
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"evidence/v4_report"
plt.rcParams.update({"font.family":"Malgun Gothic","font.size":12,"axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":False,
                     "axes.titlesize":12.5,"legend.fontsize":10.5,"xtick.labelsize":11,"ytick.labelsize":11})
INK,MUTED,ACCENT,GOOD,PALE = "#1F2A44","#7A8599","#E4572E","#2855DE","#C9D3E8"


def results(run):
    with (ROOT/"runs"/run/"results.csv").open(encoding="utf-8") as file:
        return [{k.strip():float(v) for k,v in row.items()} for row in csv.DictReader(file)]


def training_curves(final):
    """Left: v3's step size was so small its loss barely moved. Right: on unseen validation events the
    unfrozen run peaks early and then memorises training scenes; the frozen-backbone run stays flat -> pick by val."""
    fig,(left,right) = plt.subplots(1,2,figsize=(7.4,3.5),dpi=250)
    # Runs use different data/classes, so losses are shown relative to their own epoch 1: the slope is the point.
    for run,label,color,marker in (("v3_selected","처음 설정, 학습률 0.00002",MUTED,"o"),("v4","고친 설정, 학습률 0.000119",ACCENT,"s")):
        rows = results(run)
        first = rows[0]["train/box_loss"]+rows[0]["train/cls_loss"]
        left.plot([r["epoch"] for r in rows],[100*(r["train/box_loss"]+r["train/cls_loss"])/first for r in rows],"-",marker=marker,ms=3.5,lw=2.2,color=color,label=label)
    left.set_title("학습 손실 (1 epoch = 100%)",color=INK)
    left.set_xlabel("epoch")
    left.legend(frameon=False)
    # Same relabelled data and validation set for both: only the frozen backbone differs.
    for run,label,color,marker in (("v4b","백본까지 전부 학습",ACCENT,"s"),(final,"백본 고정 (최종)",GOOD,"D")):
        rows = results(run)
        values = [r["metrics/mAP50-95(B)"] for r in rows]
        right.plot([r["epoch"] for r in rows],values,"-",marker=marker,ms=3.5,lw=2.2,color=color,label=label)
    right.set_title("검증 mAP50-95 (새 사건 3개)",color=INK)
    right.set_xlabel("epoch")
    right.legend(frameon=False,loc="lower left")
    for ax in (left,right):
        ax.grid(axis="y",color="#E6E9F0")
    fig.tight_layout()
    fig.savefig(OUT/"training_curves.png")


def alert_timeline(alerts_json):
    """Per held-out video: stop interval that produced an alert (bar) and the alert moment (marker),
    from human labels (grey, upper) and from the model (red, lower). Empty rows = correctly no alert."""
    data = json.loads(Path(alerts_json).read_text(encoding="utf-8"))
    scenes = [r for r in data["scenes"] if r["reference_alerts"] or r["model_alerts"]]  # only scenes where something should happen
    fig,ax = plt.subplots(figsize=(8.4,2.5),dpi=230)
    for y,row in enumerate(reversed(scenes)):
        for source,offset,color in (("reference_alerts",.18,MUTED),("model_alerts",-.18,ACCENT)):
            for a in row[source]:
                ax.barh(y+offset,a["end_seconds"]-a["stopped_since_seconds"],left=a["stopped_since_seconds"],height=.3,color=color,alpha=.85)
                ax.plot(a["alert_seconds"],y+offset,marker="v",color=INK,ms=7)
        if not row["reference_alerts"] and not row["model_alerts"]:
            ax.text(.4,y,"알림 없음, 정답과 같음",va="center",fontsize=11.5,color=MUTED)
    ax.set_yticks(range(len(scenes)),[r["name"].replace(" · "," ") for r in reversed(scenes)],fontsize=11.5)
    ax.set_ylim(-.6,len(scenes)-.4)
    ax.set_xlim(0,30)
    ax.set_xlabel("영상 시각 (초)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MUTED,label="사람이 표시한 정차 구간"),Patch(color=ACCENT,label="모델 감지 정차 구간"),
                       Line2D([],[],marker="v",color=INK,ls="",label=f'알림 ({data["alert_seconds"]:g}초 도달)')],
              frameon=False,fontsize=11,loc="upper center",bbox_to_anchor=(.42,-.32),ncol=3)
    ax.grid(axis="x",color="#E6E9F0")
    fig.tight_layout()
    fig.savefig(OUT/"alert_timeline.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    final = sys.argv[1]
    training_curves(final)
    alert_timeline(sys.argv[2])
    print("saved",sorted(p.name for p in OUT.glob("*.png")))
