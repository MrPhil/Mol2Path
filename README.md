# Chemical Vision to Synergistic Mechanisms: Structure-Driven Inference for Drug Combination Prediction

This is a drug synergy predictor. Before the article is published, this project only contains all data, and the reproduction code is shown in the submission document.

Mol2PathVC predicts drug synergy scores from molecular images, perturbation profiles, pathway annotations, and cell embeddings. Keep `Mol2PathVC` and `Mol2PathVC-dataset` as sibling directories.

## System requirements

Installation was tested on Ubuntu 16.04, CentOS 7, and Windows 10 with Python 3.7 on one NVIDIA RTX 4080 Ti GPU.

## Data sources

All drug synergy and individual drug sensitivity information is collected from Kuru [91] and DrugComb [92]. Drug property data are derived from MoleculeNet [38]. Gene expression profiles for different cell lines are collected from the GDSC database [93]. Gene-pathway relationships are taken from the study by Deng et al. [37], and drug-gene associations are obtained from STITCH [94]. Molecular image data are sourced from PubChem [26]. Pretrained weights for the cell large language model are downloaded from [C2S-Pythia-410m-cell-type-prediction](https://huggingface.co/vandijklab/C2S-Pythia-410m-cell-type-prediction).

## Installation

After downloading the code and data, execute the following command to install all dependencies. This may take some time.

```bash
pip install -r requirements.txt
```

## Usage

Quick training and evaluation:

```bash
python train.py --toy
python test.py --toy
```

Training and evaluation:

```bash
python train.py
python test.py
```

Use `--data-root` when the dataset directory is stored elsewhere.
