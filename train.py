import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn

from dataset import load_imgdataset, var_mean
from model_torch import DrugCombinationModel
from vq_vae import Model
import utils
import utils_step3


parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default=None)
parser.add_argument('--output-dir', default=None)
parser.add_argument('--metric', choices=['bliss', 'loewe'], default='bliss')
parser.add_argument('--fold', choices=['0', '1', '2', '3', '4', 'all'], default='0')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--stage2-epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--toy', action='store_true')
parser.add_argument('--device', default='auto')
parser.add_argument('--threads', type=int, default=4)
args = parser.parse_args()
if args.epochs < 1 or args.stage2_epochs < 1 or args.batch_size < 2:
    parser.error('Epoch counts must be positive and batch size must be at least 2.')

code_path = Path(__file__).resolve().parent
data_root = Path(args.data_root).resolve() if args.data_root else code_path.parent / 'Mol2PathVC-dataset'
dataset_path = data_root / 'DrugComb'
result_path = data_root / 'results' / args.metric / 'S1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
torch.set_num_threads(args.threads)
synergy_metric = 'synergy_' + args.metric
random_state = 20241014
batchsize = 8 if args.toy else args.batch_size
step2_epochs = 1 if args.toy else args.stage2_epochs
n_epochs = 1 if args.toy else args.epochs

haveGene_drugCom2 = pd.read_csv(data_root / 'all_dataset' / 'DrugComb.csv')
matrix_cellExp = pd.read_csv(dataset_path / '0matrix_all_forCell_cellExp_landmark_ENSG.csv', index_col=0)
matrix_drugExp = pd.read_csv(dataset_path / '0matrix_all_drugExp_landmark_ENSG.csv', index_col=0)
matrix_cellExp.iloc[:, :] = utils.normalize(matrix_cellExp.values)
matrix_all_pathway_ENSG = pd.read_csv(dataset_path / '0matrix_all_pathway_ENSG.csv', index_col=0)
matrix_all_pathway_ENSG[matrix_all_pathway_ENSG > 0] = 1
Gene_pathway = matrix_all_pathway_ENSG.T
pathway_list = list(matrix_all_pathway_ENSG.index)
matrix_cellExp.index = matrix_cellExp.index.str[5:].astype(int)

kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
for split, (train_index, test_index) in enumerate(kf.split(haveGene_drugCom2)):
    if args.fold != 'all' and split != int(args.fold):
        continue
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    stored_test = np.load(result_path / str(split) / 'test_index.npy')
    if not np.array_equal(test_index, stored_test):
        raise ValueError('Dataset row order does not match the saved split.')
    if args.toy:
        rng = np.random.default_rng(random_state)
        train_index = rng.choice(train_index, size=64, replace=False)
        test_index = rng.choice(test_index, size=32, replace=False)

    run_name = 'toy' if args.toy else 'trained'
    output_path = Path(args.output_dir).resolve() if args.output_dir else data_root / 'results' / run_name / args.metric / str(split)
    if args.output_dir and args.fold == 'all':
        output_path = output_path / str(split)
    output_path.mkdir(parents=True, exist_ok=True)
    if output_path == result_path / str(split):
        raise ValueError('Choose an output directory separate from the supplied weights.')

    trainLoader, validLoader, testLoader, trainLoader_image, validLoader_image, testLoader_image = utils.define_dataloader(
        synergy_metric, train_index, test_index, matrix_drugExp, matrix_cellExp,
        haveGene_drugCom2, batchsize, str(dataset_path), device)
    print('fold:', split, 'train:', len(trainLoader.dataset), 'valid:', len(validLoader.dataset),
          'test:', len(testLoader.dataset), 'device:', device, flush=True)

    layers = {'encoder_layers': [512, 256, 128, 64],
              'ic50_layers': [256, 128, 64], 'synergy_layers': [256, 128, 64]}
    model = DrugCombinationModel(len(pathway_list), 64, 32, 2, layers, 2, 0,
                                 Gene_pathway.values, 0.3, 512, fine_tune=False).to(device)
    vq_vae = Model(128, 2, 32, 512, 64, 0.25, 0.99).to(device)
    vq_vae.load_state_dict(torch.load(result_path / 'all_vae_weights.pth', map_location=device, weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer_v = torch.optim.Adam(vq_vae.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = nn.MSELoss()
    criterion2 = nn.MSELoss()
    data_variance = var_mean(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)
    all_drug = haveGene_drugCom2.iloc[np.hstack((train_index, test_index))]
    cid_list = np.unique(np.hstack((all_drug['cid_row'].values, all_drug['cid_col'].values)))
    dic_img_dic = load_imgdataset(str(dataset_path), cid_list)

    print('**********-step 2-*************', flush=True)
    min_loss = float('inf')
    num_epoch_no_gain = 0
    best_model = copy.deepcopy(model.state_dict())
    best_model_vae = copy.deepcopy(vq_vae.state_dict())
    for epochind in range(step2_epochs):
        epoch_loss, model = utils_step3.train(
            trainLoader, model, epochind, optimizer, device, scheduler, criterion,
            criterion2, vq_vae, optimizer_v, data_variance, dic_img_dic, 0.99)
        best_model, min_loss, num_epoch_no_gain, metric, best_model_vae = utils_step3.validate_new(
            validLoader, device, model, scheduler, criterion, criterion2, epoch_loss,
            min_loss, num_epoch_no_gain, best_model, vq_vae, data_variance,
            best_model_vae, dic_img_dic)
        model.load_state_dict(best_model)
        vq_vae.load_state_dict(best_model_vae)
        if num_epoch_no_gain >= 15:
            break

    print('**********-step 3-*************', flush=True)
    min_loss = float('inf')
    num_epoch_no_gain = 0
    best_model = copy.deepcopy(model.state_dict())
    vq_vae.eval()
    for parameter in vq_vae.parameters():
        parameter.requires_grad = False
    for epochind in range(n_epochs):
        epoch_loss, model = utils.train(trainLoader, model, epochind, optimizer, device,
                                        scheduler, criterion, criterion2, vq_vae, dic_img_dic)
        best_model, min_loss, num_epoch_no_gain, metric = utils.validate_new(
            validLoader, device, model, scheduler, criterion, criterion2, epoch_loss,
            min_loss, num_epoch_no_gain, best_model, vq_vae, dic_img_dic)
        model.load_state_dict(best_model)
        if num_epoch_no_gain >= 15:
            break

    torch.save(model.state_dict(), output_path / 'model_weights_final.pth')
    torch.save(vq_vae.state_dict(), output_path / 'vae_weights.pth')
    np.save(output_path / 'test_index.npy', test_index)
    mse_synergy, spearman_synergy, pearson_synergy = utils.test_new(
        testLoader, device, model, criterion, criterion2, vq_vae, dic_img_dic)
    print('mse_synergy:', mse_synergy, 'spearman_synergy:', spearman_synergy,
          'pearson_synergy:', pearson_synergy, 'saved:', output_path, flush=True)
