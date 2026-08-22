# Cross-Calibration between Tversky Indices in Segmentation

This repository contains the public reproduction code for the simulations and
semantic-segmentation experiments in *Cross-Calibration between Tversky Indices
in Segmentation*. The authoritative experiment settings are in
[`configs/paper.yaml`](configs/paper.yaml).

The real-data experiments train U-Net and FCN8 probability estimators, then
apply Dice- and IoU-specific RankSEG inference to the same probability outputs.
You can either reproduce the reported tables from the released checkpoints or
train all ten models from scratch.

## 1. Environment

The reported runs used Python 3.9, PyTorch 1.10, torchvision 0.11, CUDA 11.3,
RankSEG 0.0.5, and FP32 on one NVIDIA GeForce RTX 4080 Super (16 GB). Full
training and original-resolution RankSEG evaluation are intended for a CUDA
GPU. CPU execution is useful for tests and small smoke runs but is much slower.

Create the environment:

```bash
conda create -n tversky-cc python=3.9 -y
conda activate tversky-cc

# For the paper's CUDA 11.3 stack:
conda install pytorch=1.10.0 torchvision=0.11.0 cudatoolkit=11.3 -c pytorch -y

pip install -r requirements.txt
pip install -e .
python -m unittest discover -s tests -v
```

For a CPU-only installation, omit the `conda install` line and let
`requirements.txt` install PyTorch. Confirm the important versions with:

```bash
python - <<'PY'
import importlib.metadata as metadata
import sys
import torch
import torchvision

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("RankSEG:", metadata.version("RankSEG"))
PY
```

All paper defaults use seed 42, deterministic FP32 execution, Adam with
learning rate and weight decay `1e-4`, at most 200 epochs, and the checkpoint
with the lowest validation loss.

## 2. Download the datasets

The repository does not download datasets automatically. Download each
official archive into the exact staging directory below. The preparation code
only extracts, normalizes, splits, and validates local files.

### Oxford-IIIT Pet

```bash
mkdir -p datasets/downloads/oxford_pet
curl -L --fail --retry 5 \
  -o datasets/downloads/oxford_pet/images.tar.gz \
  https://thor.robots.ox.ac.uk/pets/images.tar.gz
curl -L --fail --retry 5 \
  -o datasets/downloads/oxford_pet/annotations.tar.gz \
  https://thor.robots.ox.ac.uk/pets/annotations.tar.gz
```

### ISIC 2017 Task 1

```bash
mkdir -p datasets/downloads/isic2017
curl -L --fail --retry 5 -o datasets/downloads/isic2017/ISIC-2017_Training_Data.zip \
  https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Data.zip
curl -L --fail --retry 5 -o datasets/downloads/isic2017/ISIC-2017_Training_Part1_GroundTruth.zip \
  https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Part1_GroundTruth.zip
curl -L --fail --retry 5 -o datasets/downloads/isic2017/ISIC-2017_Validation_Data.zip \
  https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Validation_Data.zip
curl -L --fail --retry 5 -o datasets/downloads/isic2017/ISIC-2017_Validation_Part1_GroundTruth.zip \
  https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Validation_Part1_GroundTruth.zip
curl -L --fail --retry 5 -o datasets/downloads/isic2017/ISIC-2017_Test_v2_Data.zip \
  https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Test_v2_Data.zip
curl -L --fail --retry 5 -o datasets/downloads/isic2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip \
  https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip
```

### Kvasir-SEG

```bash
mkdir -p datasets/downloads/kvasir_seg
curl -L --fail --retry 5 \
  -o datasets/downloads/kvasir_seg/kvasir-seg.zip \
  https://datasets.simula.no/downloads/kvasir-seg.zip
```

If the Simula download host reports a local TLS certificate error, download the
same archive from the [official Kvasir-SEG page](https://datasets.simula.no/kvasir-seg/)
in a browser and keep the filename and destination shown above. Do not disable
TLS verification.

### Pascal VOC 2012

```bash
mkdir -p datasets/downloads/voc2012
curl -L --fail --retry 5 \
  -o datasets/downloads/voc2012/VOCtrainval_11-May-2012.tar \
  https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

### Cityscapes

Cityscapes requires registration and acceptance of its license. Sign in at the
[official download page](https://www.cityscapes-dataset.com/downloads/), then
place these two files under `datasets/downloads/cityscapes/`:

```text
datasets/downloads/cityscapes/leftImg8bit_trainvaltest.zip
datasets/downloads/cityscapes/gtFine_trainvaltest.zip
```

Dataset licenses remain with their original providers. Do not redistribute the
archives.

## 3. Extract, process, and validate the datasets

Prepare each downloaded dataset separately:

```bash
python -m application.prepare_datasets --dataset oxford_pet
python -m application.prepare_datasets --dataset isic2017
python -m application.prepare_datasets --dataset kvasir_seg
python -m application.prepare_datasets --dataset voc2012
python -m application.prepare_datasets --dataset cityscapes
```

Kvasir-SEG is deterministically split with seed 42 into 720 training, 80
validation, and 200 evaluation images. The other datasets use the official
partitions plus the deterministic development splits described below. Safe
archive extraction rejects members that escape the destination directory.

The resulting layout is:

```text
datasets/
├── downloads/
├── OxfordIIITPet/{images,annotations}/
├── ISIC-2017/ISIC-2017_{Training,Validation,Test_v2}_*/
├── Kvasir-SEG/paper_split/{train,val,test}/{images,masks}/
├── VOCtrainval_11-May-2012/VOCdevkit/VOC2012/
└── Cityscapes/{leftImg8bit,gtFine}/
```

Validate one dataset or all five without extracting anything:

```bash
python -m application.prepare_datasets --dataset isic2017 --check
python -m application.prepare_datasets --all --check
```

Expected partitions:

| Dataset | Training | Validation | Evaluation |
|---|---:|---:|---:|
| Oxford-IIIT Pet | 3,312 | 368 | 3,669 official test |
| ISIC 2017 | 2,000 | 150 | 600 official test |
| Kvasir-SEG | 720 | 80 | 200 fixed test |
| Pascal VOC 2012 | 1,317 | 147 | 1,449 official val |
| Cityscapes | 2,677 | 298 | 500 official val |

Use `--download-dir /path/to/archives` when the archives are stored elsewhere,
and `--data-root /path/to/data` to override the final dataset root.

## 4. Download released checkpoints

The ten inference-only state dictionaries are hosted in the
[Hugging Face checkpoint repository](https://huggingface.co/youqiangao/tversky-cross-calibration-checkpoints).
Install the `hf` CLI through `requirements.txt`, then download only the models
you want to reproduce. Each command preserves the required directory layout.

Binary segmentation:

```bash
hf download youqiangao/tversky-cross-calibration-checkpoints binary/oxford_pet/unet/best.pt --local-dir outputs/application
hf download youqiangao/tversky-cross-calibration-checkpoints binary/oxford_pet/fcn8/best.pt --local-dir outputs/application
hf download youqiangao/tversky-cross-calibration-checkpoints binary/isic2017/unet/best.pt --local-dir outputs/application
hf download youqiangao/tversky-cross-calibration-checkpoints binary/isic2017/fcn8/best.pt --local-dir outputs/application
hf download youqiangao/tversky-cross-calibration-checkpoints binary/kvasir_seg/unet/best.pt --local-dir outputs/application
hf download youqiangao/tversky-cross-calibration-checkpoints binary/kvasir_seg/fcn8/best.pt --local-dir outputs/application
```

Multilabel semantic segmentation:

```bash
hf download youqiangao/tversky-cross-calibration-checkpoints multilabel_segmentation/voc2012/unet/best.pt --local-dir outputs/application
hf download youqiangao/tversky-cross-calibration-checkpoints multilabel_segmentation/voc2012/fcn8/best.pt --local-dir outputs/application
hf download youqiangao/tversky-cross-calibration-checkpoints multilabel_segmentation/cityscapes/unet/best.pt --local-dir outputs/application
hf download youqiangao/tversky-cross-calibration-checkpoints multilabel_segmentation/cityscapes/fcn8/best.pt --local-dir outputs/application
```

The final paths must follow this pattern:

```text
outputs/application/binary/<dataset>/<model>/best.pt
outputs/application/multilabel_segmentation/<dataset>/<model>/best.pt
```

Check that a downloaded file is a nonempty PyTorch state dictionary:

```bash
python - <<'PY'
from pathlib import Path
import torch
from application.checkpoints import is_model_state_dict

path = Path("outputs/application/binary/isic2017/unet/best.pt")
payload = torch.load(path, map_location="cpu")
assert path.stat().st_size > 0 and is_model_state_dict(payload)
print("checkpoint is loadable:", path)
PY
```

## 5. Fast checkpoint-based reproduction

Start with a small smoke run. Binary rank evaluation always restores logits to
the original mask size:

```bash
python -m application.binary.rank_eval \
  --dataset isic2017 --model unet --split test \
  --device cuda --max-samples 2 --refit
python -m application.binary.delta \
  --dataset isic2017 --checkpoint outputs/application/binary/isic2017/unet/best.pt \
  --model-label unet --split test --device cuda --max-samples 2 --refit
```

For Pascal VOC 2012 or Cityscapes, request original-resolution evaluation
explicitly:

```bash
python -m application.multilabel_segmentation.rank_eval \
  --dataset voc2012 --model unet --split test --full-resolution \
  --device cuda --max-samples 2 --refit
python -m application.multilabel_segmentation.delta \
  --dataset voc2012 --model unet --split test \
  --device cuda --max-samples 2 --refit
```

Once all five datasets and all ten checkpoints are present, reproduce the four
real-data tables:

```bash
python scripts/verify_release_checkpoints.py --device cuda --refit
```

This writes detailed results, four summary CSV files, four LaTeX tables, and
`paper_result_check.txt` under `tmp/release_verification/`. A full run compares
the results at the precision displayed in the paper. `--max-samples N` is
available for integration tests, but deliberately skips the paper-value check.

## 6. Train all models from scratch

The CLI defaults are read from `configs/paper.yaml`; no paper hyperparameters
need to be repeated on the command line.

Binary datasets:

```bash
for dataset in oxford_pet isic2017 kvasir_seg; do
  for model in unet fcn8; do
    python -m application.binary.train --dataset "$dataset" --model "$model" --device cuda
  done
done
```

Pascal VOC 2012 and Cityscapes with independent class channels:

```bash
for dataset in voc2012 cityscapes; do
  for model in unet fcn8; do
    python -m application.multilabel_segmentation.train --dataset "$dataset" --model "$model" --device cuda
  done
done
```

Training writes `last.pt`, the lowest-validation-loss `best.pt`, and the metric
history below `outputs/application/<task>/<dataset>/<model>/`. Resume an
interrupted run with `--resume-checkpoint /path/to/last.pt`. Use
`--max-train-samples`, `--max-val-samples`, and a small `--epochs` value only
for smoke testing; those overrides do not reproduce the paper models.

After training, run the same rank-evaluation and delta commands from Section 5.
The table builder can also consume four independently produced summary files:

```bash
python scripts/build_paper_tables.py --help
```

## 7. Simulations

Every entry point resets Python, NumPy, and PyTorch to seed 42. Numeric results
are cached under `tmp/data/`; each cache has a metadata sidecar containing its
parameters, package versions, and Git revision. Without
`--refit`, a compatible cache is reused and the figure is regenerated.

```bash
python simulation/sim-1-1.py --refit  # Example 1: sparse binary probabilities
python simulation/sim-1-2.py --refit  # Example 1: dense binary probabilities
python simulation/sim-3.py --refit    # Example 2: imbalanced MLS
python simulation/sim-2.py --refit    # Additional Example 3
python simulation/sim-4.py --refit    # Additional Example 4
python simulation/sim-5.py --refit    # Additional Example 5
```

The full settings use 10,000 repetitions, dimensions
`[10, 20, 50, 100, 200, 500]`, and dimension 50,000 for the additional
excess-risk studies. The large studies are computationally expensive. Exact
rank inference is used up to the configured 10,000-entry threshold; larger
problems use RankDice-BA (`eps=1e-4`) or RankIoU-RMA with `smooth=0` and
`pruning_prob=0`.

Run the tractable exact-versus-approximate sensitivity check separately:

```bash
python scripts/check_rank_approximation.py
```

## 8. Reproduction checklist

Before comparing numbers, confirm all of the following:

- `python -m application.prepare_datasets --all --check` reports the expected
  counts.
- Evaluation rows record `split=test`. For VOC 2012 and Cityscapes, the code's
  `test` alias selects the official validation partition with public labels.
- Binary results record `evaluation_resolution=original_mask`; multilabel paper
  results record `evaluation_resolution=original`.
- Ignored labels are excluded, and macro metrics average only classes present
  in each ground-truth image.
- Full checkpoint verification creates `paper_result_check.txt` with `PASS`.
- Simulations were run with `--refit` when the configuration or code changed.

The paper values are compared at their displayed precision rather than by raw
floating-point equality. Small lower-order differences can result from GPU and
library implementations without changing a displayed table entry.

## 9. Troubleshooting

- **Archive not found:** read the exception literally; it prints the expected
  filename and staging directory. `--download-dir` changes that directory.
- **Archive extracts into an unexpected extra folder:** remove the incomplete
  final dataset directory and rerun the preparation command with the untouched
  official archive.
- **Cityscapes cannot be downloaded:** log in to the official website and
  accept its license; anonymous scripted download is not supported.
- **Checkpoint is missing:** the batch verifier prints the exact `hf download`
  command for the missing model.
- **Checkpoint cannot be loaded:** download the individual file again and make
  sure it is stored under its complete task/dataset/model path.
- **CUDA out of memory:** do not increase evaluation batch sizes. Binary and
  full-resolution multilabel rank evaluation use batch size 1. Close other GPU
  jobs before a full run.
- **Rank evaluation appears stalled:** original-resolution ranking is much more
  expensive than ordinary thresholding, especially on Cityscapes.
- **Old cached results are reused:** pass `--refit`; simulation metadata rejects
  caches produced by a different configuration.

## License and citation

Original repository code is released under the MIT License. Third-party code
and separately installed packages retain their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Dataset licenses remain
with their respective providers. Citation metadata is provided in
[`CITATION.cff`](CITATION.cff).
