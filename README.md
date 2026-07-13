# C/G Two-Stream ViT Classification

Data is read from `data_content/IN_original` by default. File names must follow `id-label-mode.png`, where `mode` is `C` or `G`. The dataset code pairs C/G images with the same `id`.

Default backbone: `deit_tiny_patch16_224`. This is a small ViT-style backbone, which is more suitable for the current 846 paired 224x224 samples than a larger ViT-B model.

## Install

```bash
pip install -r requirements.txt
```

If pretrained weights cannot be downloaded, add `--no-pretrained`.

## Train And Test

Run both ViT methods:

```bash
python train_vit.py --method both --epochs 100 --batch-size 16
```

Choose GPU 0 or 1:

```bash
python train_vit.py --method both --gpu 0
python train_vit.py --method both --gpu 1
```

Run direct three-class classification:

```bash
python train_vit.py --method vit_direct
```

Run two-stage three-class classification:

```bash
python train_vit.py --method vit_two_stage
```

Two-stage logic:

- stage 1: `0` vs `1/2`
- stage 2: `1` vs `2`
- test: predict stage 1 first; non-zero samples go through stage 2

## Output

Each run creates timestamped result folders:

```text
outputs/
  vit_direct/
    20260711_153000/
      fold_1/
        best.pt
        test_metrics.json
        test_predictions.csv
      ...
  vit_two_stage/
    20260711_153000/
      fold_1/
        stage1_0_vs_12.pt
        stage2_1_vs_2.pt
        test_metrics.json
        test_predictions.csv
      ...
  summary_20260711_153000.json
```

`test_metrics.json` contains:

- `acc`
- `bacc`
- `precision`
- `f1`
- `recall`
- `specificity`
- `sensitivity`
- `confusion_matrix`
- `per_class`

## Standalone Test

After training, you can test checkpoints again without retraining:

```bash
python test_vit.py --method vit_direct --checkpoint-dir outputs/vit_direct/20260711_153000 --fold all --gpu 0
```

Test one fold only:

```bash
python test_vit.py --method vit_direct --checkpoint-dir outputs/vit_direct/20260711_153000 --fold 1 --gpu 0
```

Two-stage test:

```bash
python test_vit.py --method vit_two_stage --checkpoint-dir outputs/vit_two_stage/20260711_153000 --fold all --gpu 0
```

Standalone test outputs are saved under:

```text
outputs/vit_direct/20260711_153000/standalone_test_YYYYMMDD_HHMMSS/
```

`test_predictions.csv` includes logits and probabilities, which are useful for later sample selection and heatmap visualization.

## Mamba Training And Test

For the current small paired 224x224 dataset, the Mamba scripts default to `vim_tiny_patch16_224`, a tiny Vision Mamba-style backbone. If your installed `timm` does not include this model, upgrade `timm` or pass another installed Mamba/Vim model name with `--backbone`.

Train both Mamba methods:

```bash
python train_mamba.py --method both --gpu 0
```

Train direct Mamba classification:

```bash
python train_mamba.py --method mamba_direct --gpu 0
```

Train two-stage Mamba classification:

```bash
python train_mamba.py --method mamba_two_stage --gpu 0
```

Standalone Mamba test:

```bash
python test_mamba.py --method mamba_direct --checkpoint-dir outputs/mamba_direct/20260713_153000 --fold all --gpu 0
```

Two-stage standalone Mamba test:

```bash
python test_mamba.py --method mamba_two_stage --checkpoint-dir outputs/mamba_two_stage/20260713_153000 --fold all --gpu 0
```
