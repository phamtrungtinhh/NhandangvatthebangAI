import json
from pathlib import Path
from statistics import mean

LOG = Path('tmp_test_outputs') / 'infer_log.json'
if not LOG.exists():
    print('infer_log.json not found')
    raise SystemExit(1)

data = json.loads(LOG.read_text(encoding='utf-8'))
items = data.get('items', [])

stages = ['after_nms', 'after_resolve_cross_class', 'after_flower_pruning', 'after_scene_rules', 'final']

# Collect per-stage totals per image
per_image = []
for it in items:
    stats = it.get('stage_stats') or {}
    counts = {s: stats.get(s, 0) or 0 for s in stages}
    per_image.append({'image': it.get('image'), 'counts': counts})

# Compute summary
summary = {}
for s in stages:
    vals = [p['counts'][s] for p in per_image]
    if vals:
        summary[s] = {'mean': mean(vals), 'min': min(vals), 'max': max(vals)}
    else:
        summary[s] = {'mean': 0, 'min': 0, 'max': 0}

print('Stage summary (mean/min/max):')
for s in stages:
    v = summary[s]
    print(f"- {s}: mean={v['mean']:.2f}, min={v['min']}, max={v['max']}")

# Compute top-5 drops between consecutive stages
drops = []
for p in per_image:
    c = p['counts']
    # measure largest drop among consecutive pairs
    stage_order = stages
    prev = None
    max_drop = 0
    max_pair = None
    prev_stage = None
    for s in stage_order:
        if prev is not None:
            drop = prev - c[s]
            if drop > max_drop:
                max_drop = drop
                max_pair = (prev_stage, s)
        prev = c[s]
        prev_stage = s
    drops.append({'image': p['image'], 'max_drop': max_drop, 'pair': max_pair, 'counts': c})

# sort and print top-5
sorted_drops = sorted(drops, key=lambda x: x['max_drop'], reverse=True)
print('\nTop-5 images with largest single-stage drops:')
for d in sorted_drops[:5]:
    print(f"- {d['image']}: drop={d['max_drop']} at {d['pair']}, counts={d['counts']}")
