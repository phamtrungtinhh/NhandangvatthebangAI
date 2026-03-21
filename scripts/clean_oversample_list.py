import os
from pathlib import Path

INPUT = Path(r"S:/workspace/model_custom/dataset/oversample_db_all_train_x8_hardneg_plus_extdb.txt")
OUTPUT = INPUT.with_suffix('.clean.txt')
REPORT = Path(r"S:/workspace/tmp_test_outputs/oversample_clean_report.json")

if not INPUT.exists():
    print(f"Input not found: {INPUT}")
    raise SystemExit(1)

lines = [l.rstrip('\n') for l in INPUT.read_text(encoding='utf-8').splitlines() if l.strip()]
before = len(lines)
valid = []
missing = []
seen = set()
for p in lines:
    if p in seen:
        # keep duplicates; if you want to dedupe, skip this check
        pass
    seen.add(p)
    if os.path.exists(p):
        valid.append(p)
    else:
        missing.append(p)

OUTPUT.write_text('\n'.join(valid), encoding='utf-8')

report = {
    'input': str(INPUT),
    'output': str(OUTPUT),
    'before_lines': before,
    'after_lines': len(valid),
    'missing_count': len(missing),
    'missing_sample': missing[:20]
}

import json
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
