import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SynergyEncoderDataset, load_imgdataset
from model_torch import DrugCombinationModel
from vq_vae import Model
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default=None)
parser.add_argument('--metric', choices=['bliss', 'loewe', 'all'], default='all')
parser.add_argument('--scenario', choices=['S1', 'S2', 'S3', 'all'], default='all')
parser.add_argument('--fold', choices=['0', '1', '2', '3', '4', 'all'], default='all')
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--vq-checkpoint', default=None)
parser.add_argument('--index-file', default=None)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--max-samples', type=int, default=None)
parser.add_argument('--toy', action='store_true')
parser.add_argument('--device', default='auto')
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--direct-images', action='store_true')
parser.add_argument('--output', default=None)
args = parser.parse_args()
if args.batch_size < 1 or (args.max_samples is not None and args.max_samples < 2):
    parser.error('Use a positive batch size and at least two test samples.')
if (args.checkpoint or args.vq_checkpoint or args.index_file) and (
        args.metric == 'all' or args.scenario == 'all' or args.fold == 'all'):
    parser.error('For custom files, select one metric, scenario, and fold.')

code_path = Path(__file__).resolve().parent
data_root = Path(args.data_root).resolve() if args.data_root else code_path.parent / 'Mol2PathVC-dataset'
dataset_path = data_root / 'DrugComb'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
torch.set_num_threads(args.threads)
batchsize = 8 if args.toy else args.batch_size
random_state = 20241014
metrics = ['bliss', 'loewe'] if args.metric == 'all' else [args.metric]
scenarios = ['S1', 'S2', 'S3'] if args.scenario == 'all' else [args.scenario]
folds = range(5) if args.fold == 'all' else [int(args.fold)]
max_samples = 64 if args.toy and args.max_samples is None else args.max_samples

matrix_cellExp = pd.read_csv(dataset_path / '0matrix_all_forCell_cellExp_landmark_ENSG.csv', index_col=0)
matrix_cellExp.iloc[:, :] = utils.normalize(matrix_cellExp.values)
matrix_cellExp.index = matrix_cellExp.index.str[5:].astype(int)
matrix_drugExp = pd.read_csv(dataset_path / '0matrix_all_drugExp_landmark_ENSG.csv', index_col=0)
matrix_all_pathway_ENSG = pd.read_csv(dataset_path / '0matrix_all_pathway_ENSG.csv', index_col=0)
matrix_all_pathway_ENSG[matrix_all_pathway_ENSG > 0] = 1
Gene_pathway = matrix_all_pathway_ENSG.T
if any(s != 'S1' for s in scenarios):
    for name in ['DrugComb_novel_old', 'DrugComb_novel_novel']:
        extra = pd.read_csv(data_root / name / '0matrix_all_drugExp_landmark_ENSG.csv', index_col=0)
        matrix_drugExp = pd.concat([matrix_drugExp, extra])
    matrix_drugExp = matrix_drugExp[~matrix_drugExp.index.duplicated(keep='first')]

rows = []
dic_img_dic = {}
for scenario in scenarios:
    table = {'S1': 'DrugComb.csv', 'S2': 'DrugComb_novel_old_test.csv',
             'S3': 'DrugComb_novel_novel_test.csv'}[scenario]
    haveGene_drugCom2 = pd.read_csv(data_root / 'all_dataset' / table)
    for metric in metrics:
        synergy_metric = 'synergy_' + metric
        for split in folds:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)
            result_path = data_root / 'results' / metric / scenario / str(split)
            model_name = Path(args.checkpoint).resolve() if args.checkpoint else result_path / 'model_weights_final.pth'
            vae_name = Path(args.vq_checkpoint).resolve() if args.vq_checkpoint else result_path / 'vae_weights.pth'
            if args.index_file:
                test_index = np.load(Path(args.index_file))
            elif scenario == 'S1':
                test_index = np.load(result_path / 'test_index.npy')
            else:
                test_index = np.arange(len(haveGene_drugCom2))
            if len(test_index) == 0 or test_index.min() < 0 or test_index.max() >= len(haveGene_drugCom2):
                raise ValueError('Test indices do not match the selected table.')
            if max_samples is not None and max_samples < len(test_index):
                rng = np.random.default_rng(random_state)
                test_index = rng.choice(test_index, size=max_samples, replace=False)
            test_data = haveGene_drugCom2.iloc[test_index]
            test_dataset = SynergyEncoderDataset(
                test_data['cid_row'].values, test_data['cid_col'].values,
                test_data['COSMIC identifier'].values, test_data['ic50_row'].values,
                test_data['ic50_col'].values, test_data[synergy_metric].values,
                matrix_drugExp, matrix_cellExp, device,
                embedding_file=data_root / 'cell_LLM' / 'cell_embeddings.npz')
            testLoader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

            layers = {'encoder_layers': [512, 256, 128, 64],
                      'ic50_layers': [256, 128, 64], 'synergy_layers': [256, 128, 64]}
            model = DrugCombinationModel(len(matrix_all_pathway_ENSG), 64, 32, 2,
                                         layers, 2, 0, Gene_pathway.values, 0.3,
                                         512, fine_tune=False).to(device)
            model.load_state_dict(torch.load(model_name, map_location=device, weights_only=True))
            vq_vae = Model(128, 2, 32, 512, 64, 0.25, 0.99).to(device)
            vq_vae.load_state_dict(torch.load(vae_name, map_location=device, weights_only=True))
            cid_list = np.unique(np.hstack((test_data['cid_row'].values,
                                            test_data['cid_col'].values)))
            missing_images = [cid for cid in cid_list if cid not in dic_img_dic]
            dic_img_dic.update(load_imgdataset(str(dataset_path), missing_images))
            drug_features = None
            model.eval()
            vq_vae.eval()
            if not args.direct_images:
                drug_features = {}
                with torch.no_grad():
                    for start in range(0, len(cid_list), 32):
                        ids = cid_list[start:start + 32]
                        images = torch.stack([dic_img_dic[cid] for cid in ids]).to(device)
                        img_z = vq_vae._pre_vq_conv(vq_vae._encoder(images))
                        features = model.img2drugGeneRelation(img_z).cpu()
                        drug_features.update(zip(ids, features))
            print('testing:', metric, scenario, split, 'samples:', len(test_dataset),
                  'device:', device, flush=True)
            criterion = nn.MSELoss()
            mse_synergy, spearman_synergy, pearson_synergy = utils.test_new(
                testLoader, device, model, criterion, criterion, vq_vae, dic_img_dic, drug_features)
            rows.append({'metric': metric, 'scenario': scenario, 'fold': split,
                         'samples': len(test_dataset), 'mse': float(mse_synergy),
                         'spearman': float(spearman_synergy[0]),
                         'pearson': float(pearson_synergy[0])})
            print(rows[-1], flush=True)
            if args.output:
                output = Path(args.output).resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(output, index=False)
            del model, vq_vae

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print('mean over evaluated folds:')
print(summary.groupby(['metric', 'scenario'])[['mse', 'spearman', 'pearson']].mean())
