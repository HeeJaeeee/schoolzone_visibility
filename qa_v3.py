"""Fresh trained weights, actual GUI screenshots, all selected source videos."""
import ctypes
import hashlib
import json
import time
import tkinter as tk
from pathlib import Path
from PIL import ImageGrab
from app import PhotoApp
from photo_analysis import ROOT, DEMO, assess
from video_analysis import analyze_video, export_video_report

OUT=ROOT/'evidence/app_v3_trained'
OUT.mkdir(parents=True,exist_ok=True)

def settle(root,seconds=.2):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        root.update();time.sleep(.015)

def shot(root,name):
    root.lift();settle(root,.3)
    hwnd=ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()
    ImageGrab.grab(window=hwnd).save(OUT/name)

if __name__=='__main__':
    root=tk.Tk();app=PhotoApp(root);root.geometry('1520x960+20+20');settle(root)
    photos=[];videos=[]
    try:
        for index,entry in enumerate(app.catalog):
            app.open_path(DEMO/entry['file']);app.run_analysis()
            deadline=time.monotonic()+120
            while app.busy and time.monotonic()<deadline:root.update();time.sleep(.02)
            assert not app.busy and app.detections is not None
            analysis=assess(app.detections,app.region,app.waiting_region)
            files=app.save_result(OUT/'photo_exports')
            saved=json.loads(files[1].read_text(encoding='utf-8'));assert saved['analysis']==analysis
            photos.append({'file':entry['file'],'analysis':analysis,'vehicles':sum(d['kind']=='vehicle' for d in app.detections),'export':str(files[1])})
            if index in (0,3,9,12,13):shot(root,f'photo_{index:02d}.png')
            print('photo',index,analysis['candidate_count'],analysis['waiting_count'],analysis['boundary_count'],flush=True)
        # Real-image boundary check: intentionally overlapping setup; not an extra incident.
        app.open_path(DEMO/app.catalog[0]['file']);app.waiting_region=[.50,.18,.70,.35]
        app.zone_note.set('구역 경계 검수 · 같은 원본 사진의 설정 변경, 별도 사건 아님')
        app.run_analysis()
        while app.busy:root.update();time.sleep(.02)
        boundary=assess(app.detections,app.region,app.waiting_region)
        assert boundary['boundary_count']>0
        app.save_result(OUT/'boundary_exports');shot(root,'boundary_check.png')
        for key,profile in app.profiles.items():
            entry=next(x for x in app.catalog if x['source_video']==profile['video'])
            source=ROOT.parent/'docs/research/schoolzone_visibility_2026-10-07'/f"aihub169_suddenlyappear_{entry['source_split']}"/profile['video']
            report=analyze_video(source,profile['region'],app.detector,waiting_region=profile.get('waiting_region'))
            meta={'camera':profile['name'],'source_episode':profile['video'].replace('_A_','_VIEW_').replace('_B_','_VIEW_'),'region_note':profile['note']}
            output=export_video_report(report,meta,OUT/'video_exports')
            videos.append({'profile':key,'source':str(source),'result':str(output),'entry':report['by_zone']['entry'],
                           'waiting':report['by_zone']['waiting'],'boundary':report['by_zone']['boundary'],
                           'waiting_configured':report['waiting_configured'],'samples':report['samples']})
            (OUT/'progress.json').write_text(json.dumps(videos,ensure_ascii=False,indent=2),encoding='utf-8')
            print('video',key,[(z,report['by_zone'][z]['persistent_sample_count'],report['by_zone'][z]['longest_seconds']) for z in ('entry','waiting','boundary')],flush=True)
        # Exercise actual asynchronous video window and table with latest model.
        app.open_path(DEMO/app.catalog[0]['file']);app.open_video()
        child=app.video_window;deadline=time.monotonic()+120
        while child.running and time.monotonic()<deadline:root.update();time.sleep(.02)
        assert child.report_path is not None
        shot(child.window,'video_window.png');child.close();settle(root)
        report={'complete':True,'model':app.detector.choice,'real_photos':photos,'real_videos':videos,
                'boundary_photo_check':boundary,'actual_gui_video_verified':True,
                'limitations':'Curated acted real recordings; zone observations not physical visibility, signal state or legality.'}
        (OUT/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('V3 APPLICATION ACCEPTANCE COMPLETE',flush=True)
    finally:
        app.close();settle(root,.1)
