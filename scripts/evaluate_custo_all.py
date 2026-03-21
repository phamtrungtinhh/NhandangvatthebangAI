#!/usr/bin/env python3
from pathlib import Path
import json
from ultralytics import YOLO
from PIL import Image

SAMPLE_JSON = Path('S:/workspace/tmp_test_outputs/dumbbell_infer_best.json')
OUT_JSON = Path('S:/workspace/tmp_test_outputs/dumbbell_custo_all_eval.json')
MODEL = Path('S:/workspace/model_custom/weights/custo_all.pt')
IOU_THR = 0.5
CONF_THR = 0.25

def xywhn_to_xyxy(x,y,w,h,iw,ih):
    bw=w*iw; bh=h*ih; cx=x*iw; cy=y*ih
    x1=cx-bw/2; y1=cy-bh/2; x2=cx+bw/2; y2=cy+bh/2
    return (x1,y1,x2,y2)

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
    inter=iw*ih
    if inter==0: return 0.0
    area_a=max(0,ax2-ax1)*max(0,ay2-ay1)
    area_b=max(0,bx2-bx1)*max(0,by2-by1)
    union=area_a+area_b-inter
    if union==0: return 0.0
    return inter/union

def load_sample_images():
    payload=json.loads(SAMPLE_JSON.read_text(encoding='utf-8'))
    return [Path(item['image']) for item in payload.get('samples',[])]

def load_gt(img_path):
    label = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
    if not label.exists():
        return []
    with Image.open(img_path) as im:
        iw,ih = im.size
    out=[]
    for line in label.read_text(encoding='utf-8',errors='ignore').splitlines():
        parts=line.strip().split()
        if len(parts) < 5: continue
        try:
            cls=int(float(parts[0])); x,y,w,h = map(float,parts[1:5])
        except:
            continue
        if cls!=0: continue
        out.append(xywhn_to_xyxy(x,y,w,h,iw,ih))
    return out

def predict(img_path, model):
    r = model.predict(source=str(img_path), imgsz=640, conf=CONF_THR, max_det=100, verbose=False)[0]
    out=[]
    if getattr(r,'boxes',None) is None or getattr(r.boxes,'cls',None) is None:
        return out
    cls_list = r.boxes.cls.tolist(); conf_list = r.boxes.conf.tolist(); xyxy_list = r.boxes.xyxy.tolist()
    for i,c in enumerate(cls_list):
        if int(c)!=0: continue
        x1,y1,x2,y2 = xyxy_list[i]
        out.append((x1,y1,x2,y2,float(conf_list[i])))
    return out

def match_counts(gt,preds):
    pairs=[]
    for pi,p in enumerate(preds):
        pb=(p[0],p[1],p[2],p[3])
        for gi,g in enumerate(gt):
            pairs.append((iou(pb,g),pi,gi))
    pairs.sort(reverse=True,key=lambda x:x[0])
    used_p=set(); used_g=set(); tp=0
    for score,pi,gi in pairs:
        if score < IOU_THR: break
        if pi in used_p or gi in used_g: continue
        used_p.add(pi); used_g.add(gi); tp+=1
    fp=len(preds)-tp
    fn=len(gt)-tp
    return tp,fp,fn

def main():
    images = load_sample_images()[:20]
    model = YOLO(str(MODEL))
    total_tp=total_fp=total_fn=0
    images_with_pred=0
    per_image=[]
    fails_miss=[]
    fails_fp=[]
    for img in images:
        gt = load_gt(img)
        preds = predict(img, model)
        tp,fp,fn = match_counts(gt,preds)
        total_tp+=tp; total_fp+=fp; total_fn+=fn
        if len(preds)>0: images_with_pred+=1
        per_image.append({'image':str(img),'gt':len(gt),'pred':len(preds),'tp':tp,'fp':fp,'fn':fn})
        if fn>0:
            fails_miss.append(str(img))
        if fp>0:
            fails_fp.append(str(img))
    precision = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0.0
    recall = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0.0
    out = {
        'model':str(MODEL),
        'images':len(images),
        'images_with_pred_class0':images_with_pred,
        'total_detections_class0': total_tp + total_fp,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision':precision, 'recall':recall,
        'fail_miss': fails_miss[:3], 'fail_fp': fails_fp[:3],
        'per_image': per_image
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('Wrote', OUT_JSON)
    print('images',out['images'])
    print('images_with_pred_class0',out['images_with_pred_class0'])
    print('total_detections_class0',out['total_detections_class0'])
    print('precision',round(out['precision'],4),'recall',round(out['recall'],4))
    print('fail_miss',out['fail_miss'])
    print('fail_fp',out['fail_fp'])

if __name__ == '__main__':
    main()
