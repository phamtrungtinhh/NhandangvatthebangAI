#!/usr/bin/env python3
import argparse
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune custo_all.pt on a flower-focused 4-class dataset.')
    parser.add_argument('--model', default='S:/workspace/model_custom/weights/custo_all.pt', help='Base model path.')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint path.')
    parser.add_argument('--data', default='S:/workspace/model_custom/dataset_flower_focused/data.yaml', help='Flower-focused data yaml.')
    parser.add_argument('--manifest', default='S:/workspace/model_custom/dataset_flower_focused/manifest.json', help='Focused dataset manifest path.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--freeze', type=int, default=6)
    parser.add_argument('--project', default='S:/workspace/model_custom/_flower_runs', help='Project directory for focused runs.')
    parser.add_argument('--name', default='flower_focused_ft', help='Run name.')
    parser.add_argument('--save-period', type=int, default=1)
    parser.add_argument('--cls-loss-gain', type=float, default=1.5, help='Classification loss gain.')
    parser.add_argument('--box-loss-gain', type=float, default=7.5, help='Box regression loss gain.')
    parser.add_argument('--dfl-loss-gain', type=float, default=1.5, help='DFL loss gain.')
    parser.add_argument('--mosaic', type=float, default=0.6, help='Ultralytics mosaic strength for the focused dataset.')
    parser.add_argument('--mixup', type=float, default=0.15, help='Ultralytics mixup strength for the focused dataset.')
    parser.add_argument('--copy-paste', type=float, default=0.0, help='Ultralytics copy-paste strength.')
    parser.add_argument('--deploy', action='store_true', help='Deploy best.pt back to custo_all.pt after training.')
    parser.add_argument('--deploy-target', default='S:/workspace/model_custom/weights/custo_all.pt', help='Target path for deployed weights.')
    parser.add_argument('--quiet', action='store_true', help='Reduce Ultralytics console verbosity.')
    return parser.parse_args()


def backup_and_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.copy2(dst, dst.with_suffix(dst.suffix + '.bak'))
    shutil.copy2(src, dst)


def write_deploy_metadata(metadata_path: Path, run_dir: Path, best_path: Path, deploy_target: Path, args):
    payload = {
        'deployed_at': datetime.now(timezone.utc).isoformat(),
        'source_run': str(run_dir),
        'source_best': str(best_path),
        'deploy_target': str(deploy_target),
        'data_yaml': str(args.data),
        'manifest': str(args.manifest),
        'mode': 'flower_focused_finetune',
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'cls_loss_gain': args.cls_loss_gain,
        'box_loss_gain': args.box_loss_gain,
        'dfl_loss_gain': args.dfl_loss_gain,
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def write_error_log(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'train_error.log').write_text(traceback.format_exc(), encoding='utf-8')


def print_dataset_summary(data_yaml_path: Path, manifest_path: Path):
    data_payload = yaml.safe_load(data_yaml_path.read_text(encoding='utf-8')) or {}
    print('Focused data yaml:', data_yaml_path)
    print('Train split:', data_payload.get('train'))
    print('Val split:', data_payload.get('val'))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        counts = manifest.get('counts', {})
        print('Train records:', counts.get('train_records', 0))
        print('Sparse augments:', len(manifest.get('items', {}).get('train_sparse_augmented', [])))
        print('Dense tiles:', len(manifest.get('items', {}).get('train_dense_tiles', [])))
        print('Focused val images:', counts.get('focused_val_unique', 0))


def main():
    args = parse_args()
    model_path = args.resume or args.model
    model = YOLO(model_path)
    project_path = Path(args.project)
    run_dir = project_path / args.name

    print_dataset_summary(Path(args.data), Path(args.manifest))
    print('Run directory:', run_dir)
    if args.resume:
        print('Resuming from:', args.resume)

    try:
        model.train(
            resume=args.resume if args.resume else False,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            freeze=args.freeze,
            project=str(project_path),
            name=args.name,
            exist_ok=True,
            save=True,
            save_period=args.save_period,
            val=True,
            cache=False,
            deterministic=False,
            cos_lr=True,
            close_mosaic=8,
            plots=True,
            amp=True,
            cls=args.cls_loss_gain,
            box=args.box_loss_gain,
            dfl=args.dfl_loss_gain,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            verbose=not args.quiet,
        )

        if args.deploy:
            save_dir = Path(getattr(model.trainer, 'save_dir', run_dir))
            best_path = save_dir / 'weights' / 'best.pt'
            if not best_path.exists():
                raise FileNotFoundError(f'Best checkpoint not found: {best_path}')
            deploy_target = Path(args.deploy_target)
            backup_and_copy(best_path, deploy_target)
            write_deploy_metadata(deploy_target.with_suffix('.flower_focused.deployment.json'), save_dir, best_path, deploy_target, args)
            print('Deployed best checkpoint to:', deploy_target)
    except Exception:
        write_error_log(run_dir)
        raise


if __name__ == '__main__':
    main()