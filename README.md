# Tversky Cross-Calibration Experiments

This repository contains the public reproduction code for the simulation and
semantic-segmentation experiments in *Cross-Calibration between Tversky Indices in Segmentation*.
The authoritative settings are recorded in [`configs/paper.yaml`](configs/paper.yaml).

## Environment

The paper environment uses Python 3.9, PyTorch 1.10, torchvision 0.11,
RankSEG 0.0.5, and SciPy 1.11.4. A CUDA 11.3 PyTorch build was used for the
reported GPU experiments.

```bash
conda create -n tversky-cc python=3.9
conda activate tversky-cc
pip install -r requirements-repro.txt
pip install -e .
python -m unittest discover -s tests -v
```

`predict_rank` is the only rank-inference entry point. Problems with at most
10,000 valid entries use the exact Poisson-binomial implementation. Larger
problems call the official RankSEG package: Dice uses BA with `eps=1e-4`, and
IoU uses RMA. RankSEG 0.0.5 does not expose an exact solver and supports BA for
Dice but not IoU. Both branches use `smooth=0` and `pruning_prob=0`.

## Simulations

Each script stores numeric results under `tmp/data/`, reuses them by default,
and regenerates figures from the CSV. Each cached result CSV has a JSON sidecar
recording seed, parameters, package versions, configuration hash, and Git
commit. Pass `--refit` to overwrite the cache. Every simulation entry point
resets Python, NumPy, and PyTorch to seed 42.

```bash
python simulation/sim-1-1.py --refit  # Example 1, sparse
python simulation/sim-1-2.py --refit  # Example 1, dense
python simulation/sim-3.py --refit    # Example 2
python simulation/sim-2.py --refit    # Appendix Example 3
python simulation/sim-4.py --refit    # Appendix Example 4
python simulation/sim-5.py --refit    # Appendix Example 5
```

The full settings use 10,000 repetitions, dimensions
`[10, 20, 50, 100, 200, 500]`, and dimension 50,000 for the appendix
excess-risk experiments. A quick simulation check is:

```bash
python -c "import importlib.util; s=importlib.util.spec_from_file_location('x','simulation/sim-1-1.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.sim(sample_size=8,width=10,height=1)[4].shape)"
```

The scripts for Figures 4 and 5 plot every raw excess-risk pair and write
`sim-4-bound-diagnostics.csv` and `sim-5-bound-diagnostics.csv`, containing
violation counts and maximum violations at tolerance `1e-6`. Appendix Examples
3--5 use independent component-specific flip probabilities `(rho_1, rho_2)`
on the full `51 x 51` grid.

For a tractable exact-versus-BA/RMA sensitivity check:

```bash
python scripts/check_rank_approximation.py
```

## Real applications

The experiments cover Oxford-IIIT Pet, ISIC 2017, Kvasir-SEG, Pascal VOC
2012, and Cityscapes. Dataset files and benchmark licenses prevent bundling
the data. Put them below `datasets/` using the roots in `configs/paper.yaml`.
Kvasir can be downloaded and split deterministically with:

```bash
python -m application.prepare_kvasir
```

Train both paper architectures from scratch, for example:

```bash
python -m application.binary.train --dataset isic2017 --model unet
python -m application.binary.train --dataset isic2017 --model fcn8
python -m application.multilabel_segmentation.train --dataset voc2012 --model unet
python -m application.multilabel_segmentation.train --dataset voc2012 --model fcn8
```

Repeat for the datasets listed by each command's `--help`. Defaults exactly
match the paper: Adam with learning rate and weight decay `1e-4`, at most 200
epochs, deterministic FP32 execution, seed 42, and the checkpoint with lowest validation loss. Run training
inside a persistent terminal session for the full experiments.

Evaluate rank predictions at original resolution:

```bash
python -m application.binary.rank_eval --dataset isic2017 --model unet --split test
python -m application.multilabel_segmentation.rank_eval --dataset voc2012 --model unet --split test
```

Multilabel rank evaluation always runs at original image resolution, as
required by the paper protocol and generated paper tables.

Compute the transfer-error summaries on the evaluation partition:

```bash
python -m application.binary.delta --dataset isic2017 --split test
python -m application.multilabel_segmentation.delta --dataset voc2012 --split test
```

The four summary CSVs can be converted into auditable LaTeX tables with
`scripts/build_paper_tables.py --help`. The generator rejects rows whose
recorded split is not `test`.

## License

Original code in this repository is released under the MIT License. Third-party
components retain their respective license terms; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Dataset licenses remain
with their respective providers.
