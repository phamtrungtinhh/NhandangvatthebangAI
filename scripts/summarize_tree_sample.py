import json
from pathlib import Path
p=Path(r'S:/workspace/tmp_test_outputs/tree_sample_batch.json')
if not p.exists():
    print('{"error":"file not found","path":"%s"}'%p)
    raise SystemExit(1)

d=json.loads(p.read_text(encoding='utf-8'))

n=len(d)
raw_tree_imgs=[]
misses=[]
goods=[]
drops=[]
for item in d:
    img=item.get('image')
    rep=item.get('report',{})
    raw=rep.get('raw_custom_counts',{})
    raw_tree=raw.get('tree',0)
    final=rep.get('num_final',0)
    if raw_tree>0:
        raw_tree_imgs.append(img)
        if final==0:
            misses.append(img)
        else:
            goods.append({"image":img,"raw_tree":raw_tree,"final":final})
        drops.append({"image":img,"raw_tree":raw_tree,"final":final,"drop":raw_tree-final})

# sort drops by decrease descending
drops_sorted=sorted(drops,key=lambda x: x['drop'], reverse=True)

out={
    'total_images': n,
    'images_with_raw_tree': len(raw_tree_imgs),
    'misses_count': len(misses),
    'misses': misses[:5],
    'top_drops': drops_sorted[:5],
    'examples_good': goods[:5]
}
print(json.dumps(out, indent=2))
