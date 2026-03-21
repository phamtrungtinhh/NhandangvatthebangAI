import sys
from pathlib import Path

def is_label_nonempty(img_path: Path) -> bool:
    # check same-name .txt label next to image
    label = img_path.with_suffix('.txt')
    if label.exists() and label.stat().st_size > 0:
        return True
    # also check a common dataset layout: replace '/images/' with '/labels/'
    try:
        p = img_path.as_posix()
        if '/images/' in p:
            p2 = p.replace('/images/', '/labels/')
            label2 = Path(p2).with_suffix('.txt')
            if label2.exists() and label2.stat().st_size > 0:
                return True
    except Exception:
        pass
    return False


def filter_manifest(manifest_path: Path, out_path: Path):
    kept = 0
    total = 0
    with manifest_path.open('r', encoding='utf-8') as f_in, out_path.open('w', encoding='utf-8') as f_out:
        for line in f_in:
            total += 1
            line = line.strip()
            if not line:
                continue
            img_path = Path(line)
            if is_label_nonempty(img_path):
                f_out.write(line + '\n')
                kept += 1
    print(f"Filtered manifest: {total} entries -> {kept} kept. Output: {out_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: filter_manifest_remove_background.py <manifest_in.txt> <manifest_out.txt>')
        sys.exit(2)
    manifest_in = Path(sys.argv[1])
    manifest_out = Path(sys.argv[2])
    if not manifest_in.exists():
        print('Manifest not found:', manifest_in)
        sys.exit(1)
    filter_manifest(manifest_in, manifest_out)
