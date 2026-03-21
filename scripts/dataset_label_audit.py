#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path('S:/workspace/model_custom/dataset')
SPLITS = ['train', 'val']
ALLOWED_CLASSES = {0,1,2,3}
TINY_THRESH = 0.01

report = {
    'summary': {},
    'class_distribution': {},
    'suspicious_files': []
}

summary = defaultdict(int)
class_counts = Counter()
suspicious = {}

def image_extensions():
    return {'.jpg','.jpeg','.png','.bmp','.gif','.webp','.tif','.tiff'}

for split in SPLITS:
    images_dir = ROOT / split / 'images'
    labels_dir = ROOT / split / 'labels'
    imgs = {}
    lbls = {}
    # collect images
    if images_dir.exists():
        for p in images_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in image_extensions():
                imgs[p.stem] = p
    # collect labels
    if labels_dir.exists():
        for p in labels_dir.rglob('*.txt'):
            if p.is_file():
                lbls[p.stem] = p

    # images and labels counts
    summary['total_images'] += len(imgs)
    summary['total_label_files'] += len(lbls)

    # images without labels
    for stem, imgp in imgs.items():
        if stem not in lbls:
            key = str(imgp)
            suspicious.setdefault(key, []).append('image_has_no_label')
            summary['images_without_label'] += 1

    # labels without images
    for stem, lblp in lbls.items():
        if stem not in imgs:
            key = str(lblp)
            suspicious.setdefault(key, []).append('label_without_image')
            summary['labels_without_image'] += 1

    # inspect label files
    for stem, lblp in lbls.items():
        issues = []
        try:
            txt = lblp.read_text(encoding='utf-8').strip()
        except Exception as e:
            issues.append(f'read_error:{e}')
            suspicious.setdefault(str(lblp), []).extend(issues)
            continue
        if txt == '':
            issues.append('empty_label_file')
            summary['empty_label_files'] += 1
            suspicious.setdefault(str(lblp), []).extend(issues)
            continue
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        for i, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) < 5:
                issues.append(f'bad_format_line_{i}')
                summary['bbox_errors'] += 1
                continue
            cls_s, x_s, y_s, w_s, h_s = parts[:5]
            try:
                cls_id = int(float(cls_s))
            except Exception:
                issues.append(f'invalid_class_line_{i}')
                summary['invalid_class'] += 1
                continue
            if cls_id not in ALLOWED_CLASSES:
                issues.append(f'invalid_class_{cls_id}_line_{i}')
                summary['invalid_class'] += 1
            try:
                x = float(x_s); y = float(y_s); w = float(w_s); h = float(h_s)
            except Exception:
                issues.append(f'non_numeric_bbox_line_{i}')
                summary['bbox_errors'] += 1
                continue
            # bounds
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                issues.append(f'bbox_out_of_range_line_{i}')
                summary['bbox_errors'] += 1
            # tiny
            if w < TINY_THRESH or h < TINY_THRESH:
                issues.append(f'tiny_bbox_w{w:.6f}_h{h:.6f}_line_{i}')
                summary['tiny_bboxes'] += 1
            # all zero bbox
            if x == 0 and y == 0 and w == 0 and h == 0:
                issues.append(f'zero_bbox_line_{i}')
                summary['zero_bboxes'] += 1
            # count class
            class_counts[cls_id] += 1
            summary['total_objects'] += 1
        if issues:
            suspicious.setdefault(str(lblp), []).extend(issues)

# class distribution check
report['class_distribution'] = {str(k): int(v) for k,v in class_counts.items()}

# prepare summary fields defaults
for k in ['total_images','total_label_files','empty_label_files','bbox_errors','tiny_bboxes','invalid_class','images_without_label','labels_without_image','total_objects','zero_bboxes']:
    summary.setdefault(k, 0)

report['summary'] = dict(summary)

# suspicious files: sort by number of issues desc
suspicious_list = []
for f, issues in suspicious.items():
    suspicious_list.append({'file': f, 'issues': issues, 'count': len(issues)})
suspicious_list.sort(key=lambda x: (-x['count'], x['file']))
report['suspicious_files'] = suspicious_list[:20]

# imbalance warning
total_objs = report['summary'].get('total_objects', 0)
if total_objs > 0:
    dist = {k: v for k,v in report['class_distribution'].items()}
    report['class_distribution'] = dist
    max_class, max_count = None, 0
    for k,v in class_counts.items():
        if v > max_count:
            max_class, max_count = k, v
    if max_count / total_objs > 0.7:
        report['warning'] = f'class_distribution_highly_imbalanced: class={max_class}, ratio={max_count/total_objs:.2f}'

# write json
OUT = Path('S:/workspace/model_custom/dataset_label_audit.json')
OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')

# print terminal summary
print('Dataset audit written to', OUT)
print('--- Summary ---')
print('Total images:', report['summary'].get('total_images',0))
print('Total label files:', report['summary'].get('total_label_files',0))
print('Total objects:', report['summary'].get('total_objects',0))
print('Empty label files:', report['summary'].get('empty_label_files',0))
print('BBox errors:', report['summary'].get('bbox_errors',0))
print('Tiny bboxes:', report['summary'].get('tiny_bboxes',0))
print('Invalid class:', report['summary'].get('invalid_class',0))
print('Images without labels:', report['summary'].get('images_without_label',0))
print('Labels without images:', report['summary'].get('labels_without_image',0))
if 'warning' in report:
    print('WARNING:', report['warning'])

print('\nTop suspicious files:')
for s in report['suspicious_files']:
    print(s['file'], s['count'], s['issues'])
