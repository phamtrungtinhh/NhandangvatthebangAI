"""
Watch a training run directory for checkpoint arrival and run evaluations.
Usage:
  python watch_and_eval_on_checkpoint.py --run-dir <run_dir> [--prefer-best] [--interval 60]

By default prefers `best.pt`. When a checkpoint appears, runs:
  - scripts/run_raw_targeted_conf005_check.py
  - scripts/run_targeted_dumbbell_tree_checks.py

Outputs are written according to the called scripts' behavior.
"""
import argparse
import time
from pathlib import Path
import subprocess
import sys


def find_checkpoint(run_dir: Path, prefer_best: bool):
    best = run_dir / 'weights' / 'best.pt'
    last = run_dir / 'weights' / 'last.pt'
    if prefer_best:
        if best.exists():
            return best
        if last.exists():
            return last
        return None
    else:
        if last.exists():
            return last
        if best.exists():
            return best
        return None


def run_eval(script_path: Path, extra_args=None):
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd += extra_args
    print('Running:', ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=Path(__file__).parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, help='Path to training run (folder containing weights/)')
    ap.add_argument('--prefer-best', action='store_true', help='Prefer best.pt over last.pt')
    ap.add_argument('--interval', type=int, default=60, help='Poll interval seconds')
    ap.add_argument('--once', action='store_true', help='Exit after first evaluation')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print('Run dir not found:', run_dir)
        sys.exit(2)

    print('Watching', run_dir, 'prefer_best=', args.prefer_best)
    checked = False
    while True:
        ckpt = find_checkpoint(run_dir, args.prefer_best)
        if ckpt:
            print('Found checkpoint:', ckpt)
            # Run raw detect (conf=0.05)
            run_eval(Path('scripts/run_raw_targeted_conf005_check.py'), [str(ckpt)])
            # Run full pipeline targeted eval
            run_eval(Path('scripts/run_targeted_dumbbell_tree_checks.py'), [str(ckpt)])
            checked = True
            if args.once:
                print('Done (once)')
                return
            # wait for a new checkpoint modification
            last_mtime = ckpt.stat().st_mtime
            print('Waiting for checkpoint update...')
            while True:
                time.sleep(args.interval)
                ckpt2 = find_checkpoint(run_dir, args.prefer_best)
                if ckpt2 and ckpt2.exists() and ckpt2.stat().st_mtime > last_mtime:
                    print('Checkpoint updated:', ckpt2)
                    break
        else:
            print('No checkpoint yet. Sleeping', args.interval, 's')
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
