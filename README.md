# Inferring gene associations from molecular structure for drug synergy prediction with limited annotation

Mol2Path predicts drug combination synergy from molecular images, drug-gene profiles, pathway annotations, and cell embeddings.

## Project contents

Keep `Mol2Path` and `Mol2Path-dataset` as sibling directories.

- `Mol2Path` contains the model, VQ-VAE, data loaders, training and testing scripts.
- `Mol2Path-dataset` contains processed DrugComb tables, drug and cell features, pathway annotations, molecular images, cell embeddings, and trained weights.
- `results/bliss` and `results/loewe` contain the supplied models and fold indices for S1, S2, and S3.

## System requirements

The scripts use Python 3.8, PyTorch 2.3, and torchvision 0.18. CUDA is used automatically when available. Verification was performed with CUDA 12.1 on an NVIDIA GeForce RTX 4060 Laptop GPU.

## Installation

Run the following command from `Mol2Path` to install the dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train a Bliss model using the included DrugComb split:

```bash
python train.py
```

Run a short training and validation cycle, followed by testing:

```bash
python train.py --toy
```

Select a score and fold, or train all five folds:

```bash
python train.py --metric loewe --fold 0
python train.py --metric bliss --fold all
```

Training uses the pretrained visual model, joint optimization, and synergy-model training. New weights are saved under `Mol2Path-dataset/results/trained`; toy weights are saved under `results/toy`. Use `--output-dir` to choose another destination.

## Testing

Evaluate the supplied Bliss and Loewe models across the three scenarios and five folds:

```bash
python test.py
```

Select a model or run a quick test:

```bash
python test.py --metric bliss --scenario S1 --fold 0
python test.py --metric bliss --scenario S1 --fold 0 --toy
```

The script reports MSE, Spearman correlation, Pearson correlation, and the mean across the evaluated folds. Use `--output scores.csv` to save the scores. The visual features are reused for repeated drugs during testing; `--direct-images` evaluates images batch by batch.

Use `--data-root` with either script when the dataset directory is stored elsewhere.

## Data sources

Drug synergy and individual drug sensitivity information is collected from Kuru [91] and DrugComb [92]. Drug property data are derived from MoleculeNet [38]. Gene expression profiles for different cell lines are collected from the GDSC database [93]. Gene-pathway relationships are taken from the study by Deng et al. [37], and drug-gene associations are obtained from STITCH [94]. Molecular image data are sourced from PubChem [26]. Pretrained weights for the cell large language model are downloaded from [C2S-Pythia-410m-cell-type-prediction](https://huggingface.co/vandijklab/C2S-Pythia-410m-cell-type-prediction).
