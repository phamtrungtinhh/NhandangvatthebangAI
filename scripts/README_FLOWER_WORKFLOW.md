# Flower-Focused Workflow

This workflow keeps the model in the existing 4-class format and concentrates fine-tuning on flower-heavy failure cases.

## 1. Normalize labels

```powershell
python S:/workspace/scripts/normalize_flower_labels.py --overwrite --dense-mode keep
```

Use `--dense-mode split` only when you explicitly want heuristic splitting of large flower cluster boxes into a grid.

## 2. Prepare focused dataset

```powershell
python S:/workspace/scripts/prepare_flower_dataset.py --overwrite
```

What it does:
- keeps all train images that contain flower labels
- oversamples sparse flower scenes with flip and rotation
- tiles dense flower scenes into crops for counting stability
- builds a focused validation set from hard bias-check cases and dense flower scenes

## 3. Fine-tune from `custo_all.pt`

```powershell
python S:/workspace/scripts/train_flower_focused.py --deploy
```

Recommended low-VRAM start:

```powershell
python S:/workspace/scripts/train_flower_focused.py --imgsz 640 --batch 4 --epochs 40 --workers 0 --deploy
```

## 4. Compare before/after

```powershell
python S:/workspace/scripts/eval_flower_improvement.py \
  --baseline-model S:/workspace/model_custom/weights/custo_all.pt.bak \
  --candidate-model S:/workspace/model_custom/weights/custo_all.pt
```

The evaluation report is written to `S:/workspace/model_custom/weights/flower_eval_improvement.json`.

## Notes

- This workflow does not collapse the classifier head into a single class, so the fine-tuned checkpoint can be deployed back as the main `custo_all.pt` model.
- If you want per-class loss weighting beyond flower-focused sampling, that requires a custom training loop. The provided script uses dataset focus plus increased global classification loss gain.