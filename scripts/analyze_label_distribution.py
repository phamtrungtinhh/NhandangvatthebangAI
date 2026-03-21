#!/usr/bin/env python3
import json
from pathlib import Path

AUDIT = Path('S:/workspace/model_custom/dataset_label_audit.json')
INFER_LOG = Path('S:/workspace/tmp_test_outputs/infer_log.json')

CLASS_MAP = {'0': 'dumbbell', '1': 'flower', '2': 'fruit', '3': 'tree'}

def load_audit():
    if not AUDIT.exists():
        raise FileNotFoundError(f'{AUDIT} not found')
    return json.loads(AUDIT.read_text(encoding='utf-8'))

def load_infer():
    if not INFER_LOG.exists():
        return None
    return json.loads(INFER_LOG.read_text(encoding='utf-8'))


def main():
    audit = load_audit()
    class_dist = audit.get('class_distribution', {})
    total_objs = audit.get('summary', {}).get('total_objects', 0)

    print('Class | Objects | Percentage')
    print('-----------------------------')
    most = (None, -1)
    least = (None, 10**18)
    for k in sorted(CLASS_MAP.keys(), key=int):
        objs = int(class_dist.get(k, 0))
        pct = (objs / total_objs * 100) if total_objs > 0 else 0.0
        print(f"{k} ({CLASS_MAP[k]}) | {objs} | {pct:.2f}%")
        if objs > most[1]:
            most = (k, objs)
        if objs < least[1]:
            least = (k, objs)

    print('\nTotal objects:', total_objs)
    if most[0] is not None:
        print('Most frequent class:', most[0], CLASS_MAP[most[0]], most[1])
        print('Least frequent class:', least[0], CLASS_MAP[least[0]], least[1])

    # imbalance check
    if total_objs > 0 and most[1] / total_objs > 0.6:
        print('\nWARNING: dataset class imbalance detected')

    # compare with inference sample
    infer = load_infer()
    if infer is None:
        print('\nNo inference sample log found at', INFER_LOG)
        return

    sample_images = infer.get('summary', {}).get('total_images', 0)
    sample_dets = infer.get('summary', {}).get('total_detections', 0)
    by_class = infer.get('summary', {}).get('by_class', {})

    print('\nSample inference summary:')
    print('Sample images:', sample_images)
    print('Sample detections:', sample_dets)
    # normalize by_class keys to strings
    # find if all detections are flower (class 1)
    total_sample_objs = sum(int(v) for v in by_class.values()) if by_class else 0
    print('By class (sample):')
    for k,v in sorted(by_class.items(), key=lambda x:int(float(x[0])) if x[0].replace('.','',1).isdigit() else x[0]):
        # convert key to int index where possible
        try:
            ki = str(int(float(k)))
        except Exception:
            ki = k
        name = CLASS_MAP.get(ki, ki)
        print(f"{ki} ({name}) : {v}")

    # check all detected are flower
    flower_count = 0
    # by_class keys might be '1' or '1.0'
    for key, v in by_class.items():
        try:
            if int(float(key)) == 1:
                flower_count += int(v)
        except Exception:
            pass
    if total_sample_objs > 0 and flower_count == total_sample_objs:
        print('\nNote: all sample detections are class "flower"')
        # if dataset also majority flower
        if total_objs > 0 and int(class_dist.get('1',0)) / total_objs > 0.6:
            print('Possible model bias toward class "flower" (dataset is flower-dominant)')
        else:
            print('Possible model bias toward class "flower" (sample shows all detections are flower)')

if __name__ == "__main__":
    main()
