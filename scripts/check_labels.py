#!/usr/bin/env python3
"""Scan YOLO-format label files and report invalid lines/boxes.
Usage: python scripts/check_labels.py --labels-root <path> [--report report.json]
"""
import argparse
import json
from pathlib import Path


def check_label_file(p: Path):
    bad = []
    try:
        for i, ln in enumerate(p.read_text(encoding='utf-8', errors='ignore').splitlines(), start=1):
            parts = ln.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                bad.append({'line': i, 'reason': 'too_few_fields', 'text': ln.strip()})
                continue
            try:
                cid = int(float(parts[0]))
                x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
            except Exception as e:
                bad.append({'line': i, 'reason': 'parse_error', 'text': ln.strip()})
                continue
            if w <= 0 or h <= 0:
                bad.append({'line': i, 'reason': 'nonpositive_wh', 'w': w, 'h': h, 'text': ln.strip()})
            if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                bad.append({'line': i, 'reason': 'center_out_of_range', 'x': x, 'y': y, 'text': ln.strip()})
            if w > 1.0 or h > 1.0:
                bad.append({'line': i, 'reason': 'wh_gt_1', 'w': w, 'h': h, 'text': ln.strip()})
    except Exception as e:
        return [{'line': None, 'reason': 'file_read_error', 'error': str(e)}]
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels-root', default='.', help='Path to search for labels (searches **/labels/*.txt)')
    ap.add_argument('--report', default='tmp_test_outputs/label_check_report.json')
    args = ap.parse_args()

    root = Path(args.labels_root)
    files = list(root.rglob('labels/**/*.txt'))
    report = {'checked_files': len(files), 'bad_files': []}
    for f in files:
        bad = check_label_file(f)
        if bad:
            report['bad_files'].append({'file': str(f).replace('\\','/'), 'issues': bad})
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Checked', report['checked_files'], 'label files; bad files:', len(report['bad_files']))
    print('Report saved to', args.report)

if __name__ == '__main__':
    main()
