"""Selected real events, fresh pretrained initialization, event-level split."""
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import yaml

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'docs/research/schoolzone_visibility_2026-10-07'
DATA=ROOT/'data_v3'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'runtime')
from ultralytics import YOLO


if __name__=='__main__':
    cv2.setNumThreads(2)
    selected=json.loads((ROOT/'design/선정자료_후보.json').read_text(encoding='utf-8'))
    rows=[]
    splits={'training_candidate':'train','development_candidate':'development','training_excluded_reproduction':'test'}
    for item in selected['rows']:
        if item['role'] not in splits:continue
        split=splits[item['role']]
        for kind in ('images','labels'):(DATA/split/kind).mkdir(parents=True,exist_ok=True)
        capture=cv2.VideoCapture(item['source'])
        assert capture.get(cv2.CAP_PROP_FPS)==15
        wanted={math.ceil(1.5*(i-1)):i for i in item['label_indices']}
        frame_index=0
        while capture.grab():
            if frame_index in wanted:
                ok,frame=capture.retrieve();assert ok
                index=wanted[frame_index];stem=Path(item['source']).stem+f'_{index:05d}'
                xml=SOURCE/f"aihub169_suddenlyappear_{item['original']}_labels"/(stem+'.xml')
                tree=ET.parse(xml).getroot();h,w=frame.shape[:2]
                assert (w,h)==(int(tree.findtext('size/width')),int(tree.findtext('size/height')))
                lines=[]
                for obj in tree.findall('object'):
                    cls={'person':0,'vehicle':2}.get(obj.findtext('name'))
                    if cls is None:continue
                    x1,y1,x2,y2=[int(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
                    assert 0<=x1<x2<=w and 0<=y1<y2<=h
                    lines.append(f'{cls} {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
                image=DATA/split/'images'/(stem+'.jpg');label=DATA/split/'labels'/(stem+'.txt')
                success,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,95]);assert success
                encoded.tofile(str(image));label.write_text('\n'.join(lines)+'\n',encoding='utf-8')
                rows.append({'split':split,'episode':item['episode'],'source':item['source'],'image':str(image),
                             'time':frame_index/15,'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                             'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest()})
            frame_index+=1
        capture.release();assert frame_index==450
        print('prepared',split,Path(item['source']).name,flush=True)
    counts={s:sum(r['split']==s for r in rows) for s in splits.values()}
    assert counts=={'train':720,'development':180,'test':180},counts
    event_sets={s:{r['episode'] for r in rows if r['split']==s} for s in counts}
    assert not (event_sets['train']&event_sets['development'] or event_sets['train']&event_sets['test'] or event_sets['development']&event_sets['test'])
    for split in counts:
        (DATA/f'{split}.txt').write_text('\n'.join(r['image'].replace('\\','/') for r in rows if r['split']==split)+'\n',encoding='utf-8')
    model=YOLO(str(ROOT.parent/'yolo26n.pt'))
    names=dict(model.names);names[2]='vehicle'
    config={'path':str(DATA).replace('\\','/'),'train':'train.txt','val':'development.txt','names':names}
    (DATA/'dataset.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    (DATA/'manifest.json').write_text(json.dumps({'counts':counts,'rows':rows,'event_overlap':0,
        'initial_weights':str(ROOT.parent/'yolo26n.pt'),'initial_sha256':hashlib.sha256((ROOT.parent/'yolo26n.pt').read_bytes()).hexdigest(),
        'warning':'Curated previously inspected footage, not a blind field benchmark.'},ensure_ascii=False,indent=2),encoding='utf-8')
    model.model.names[2]='vehicle'
    metrics=model.train(data=str(DATA/'dataset.yaml'),epochs=10,patience=10,imgsz=960,batch=8,device=0,workers=2,
        seed=42,deterministic=True,cache=False,project=str(ROOT/'runs'),name='v3_selected',exist_ok=False,
        freeze=11,optimizer='AdamW',lr0=.00002,lrf=.5,warmup_epochs=1,mosaic=0,fliplr=0,
        degrees=0,translate=.05,scale=.15,hsv_h=.005,hsv_s=.1,hsv_v=.2,plots=True)
    (ROOT/'evidence/v3_native_metrics.json').write_text(json.dumps(metrics.results_dict,indent=2),encoding='utf-8')
    print('V3 TRAINING COMPLETE',counts,flush=True)
