import copy
from operator import itemgetter
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SynergyEncoderDataset, ImageDataset

TINY = 1e-15

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1!=0
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    # X = (X-means1)/std1[feat_filt]
    X[:,feat_filt] = (X[:,feat_filt]-means1[feat_filt])/std1[feat_filt]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X

def define_dataloader(synergy_metric,train_index, test_index, matrix_drugExp, matrix_cellExp, haveGene_drugCom2, batch_size, dataset_path, device):
    train_index, valid_index = train_test_split(train_index, test_size=1/8, random_state=42)
    haveGene_drugCom_train = haveGene_drugCom2.loc[train_index]
    haveGene_drugCom_val = haveGene_drugCom2.loc[valid_index]

    train_drug_row = haveGene_drugCom_train['cid_row'].values
    train_drug_col = haveGene_drugCom_train['cid_col'].values
    train_cell = haveGene_drugCom_train['COSMIC identifier'].values
    train_Y_row = haveGene_drugCom_train['ic50_row'].values
    train_Y_col = haveGene_drugCom_train['ic50_col'].values
    train_Y = haveGene_drugCom_train[synergy_metric].values


    test_drug_row = haveGene_drugCom2['cid_row'][test_index].values
    test_drug_col = haveGene_drugCom2['cid_col'][test_index].values
    test_cell = haveGene_drugCom2['COSMIC identifier'][test_index].values
    test_Y_row = haveGene_drugCom2['ic50_row'][test_index].values
    test_Y_col = haveGene_drugCom2['ic50_col'][test_index].values
    test_Y = haveGene_drugCom2[synergy_metric][test_index].values


    valid_drug_row = haveGene_drugCom_val['cid_row'].values
    valid_drug_col = haveGene_drugCom_val['cid_col'].values
    valid_cell = haveGene_drugCom_val['COSMIC identifier'].values
    valid_Y_row = haveGene_drugCom_val['ic50_row'].values
    valid_Y_col = haveGene_drugCom_val['ic50_col'].values
    valid_Y = haveGene_drugCom_val[synergy_metric].values


    train_dataset = SynergyEncoderDataset(train_drug_row,
                                          train_drug_col,
                                          train_cell,
                                          train_Y_row,
                                          train_Y_col
                                          , train_Y, matrix_drugExp, matrix_cellExp,device=device, embedding_file=Path(dataset_path).parent/'cell_LLM'/'cell_embeddings.npz')

    valid_dataset = SynergyEncoderDataset(valid_drug_row,
                                          valid_drug_col,
                                          valid_cell,
                                          valid_Y_row,
                                          valid_Y_col
                                          , valid_Y, matrix_drugExp, matrix_cellExp,device=device, embedding_file=Path(dataset_path).parent/'cell_LLM'/'cell_embeddings.npz')

    test_dataset = SynergyEncoderDataset(test_drug_row,
                                         test_drug_col,
                                         test_cell,
                                         test_Y_row,
                                         test_Y_col
                                         , test_Y, matrix_drugExp, matrix_cellExp,device=device, embedding_file=Path(dataset_path).parent/'cell_LLM'/'cell_embeddings.npz')

    trainLoader = DataLoader(train_dataset,
                             batch_size=batch_size, shuffle=True)

    validLoader = DataLoader(valid_dataset,
                             batch_size=batch_size, shuffle=True)

    testLoader = DataLoader(test_dataset,
                            batch_size=batch_size, shuffle=False)

    train_dataset = ImageDataset(np.unique(np.hstack((train_drug_row, train_drug_col))), dataset_path,device=device)

    valid_dataset = ImageDataset(np.unique(np.hstack((valid_drug_row, valid_drug_col))), dataset_path,device=device)

    test_dataset = ImageDataset(np.unique(np.hstack((test_drug_row, test_drug_col))), dataset_path,device=device)

    trainLoader_image = DataLoader(train_dataset,
                             batch_size=batch_size//4, shuffle=True)

    validLoader_image = DataLoader(valid_dataset,
                             batch_size=batch_size//4, shuffle=True)

    testLoader_image = DataLoader(test_dataset,
                            batch_size=batch_size//4, shuffle=False)


    return trainLoader, validLoader, testLoader, trainLoader_image, validLoader_image, testLoader_image

def validate_new(valid_loader, device,model,scheduler,criterion,criterion2,epoch_loss,min_loss,num_epoch_no_gain,best_model,vq_vae,dic_img_dic):  # valid_loader
    model.eval()
    vq_vae.eval()
    test_epoch_loss = 0.0
    num_batches = 0
    preds = torch.Tensor()
    trues = torch.Tensor()

    for i, batch in enumerate(valid_loader):
        drugR_exp, drugC_exp, cell_exp, cell_emb,synergyScore, Y_r, Y_c, drug_index = batch
        drugR_exp, drugC_exp, cell_exp, cell_emb,synergyScore, Y_r, Y_c = drugR_exp.to(device), drugC_exp.to(device), cell_exp.to(device),cell_emb.to(device), synergyScore.to(device), Y_r.to(device), Y_c.to(device)
        with torch.no_grad():
            img_zR = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[0].tolist()]).to(device)))
            img_zC = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[1].tolist()]).to(device)))
            pre_synergy, gerGene_loss, kl_loss = model(cell_emb,drugR_exp, drugC_exp, cell_exp, vq_vae._vq_vae._embedding.weight, img_zR, img_zC)
            loss1 = 10*kl_loss+10*gerGene_loss+criterion(pre_synergy[:, 0] + TINY, synergyScore[:, 0])

            loss=loss1
            test_epoch_loss += loss.item()
        preds = torch.cat((preds, pre_synergy.cpu().detach()), 0)
        trues = torch.cat((trues, synergyScore.cpu()), 0)
        num_batches += 1
    test_epoch_loss /= num_batches
    print("val_epoch_loss", test_epoch_loss)

    if test_epoch_loss < min_loss:
        best_model = copy.deepcopy(model.state_dict())
        min_loss = test_epoch_loss
        num_epoch_no_gain = 0
    else:
        num_epoch_no_gain += 1
    scheduler.step(epoch_loss)

    # roc_auc = roc_auc_score(trues, preds)

    mse_synergy = mse(preds,trues)
    spearman_synergy = spearman(preds, trues)
    pearson_synergy = pearson(preds[:,0], trues[:,0])

    return best_model,min_loss,num_epoch_no_gain,mse_synergy

def test_new(test_loader, device,model,criterion,criterion2,vq_vae,dic_img_dic,drug_features=None):  # valid_loader
    model.eval()
    vq_vae.eval()
    test_epoch_loss = 0.0
    num_batches = 0
    preds = torch.Tensor()
    trues = torch.Tensor()

    for i, batch in enumerate(test_loader):
        drugR_exp, drugC_exp, cell_exp, cell_emb,synergyScore, Y_r, Y_c, drug_index = batch
        drugR_exp, drugC_exp, cell_exp, cell_emb,synergyScore, Y_r, Y_c = drugR_exp.to(device), drugC_exp.to(device), cell_exp.to(device),cell_emb.to(device), synergyScore.to(device), Y_r.to(device), Y_c.to(device)
        with torch.no_grad():
            features = None
            if drug_features is not None:
                features = (
                    torch.stack([drug_features[k] for k in drug_index[0].tolist()]).to(device),
                    torch.stack([drug_features[k] for k in drug_index[1].tolist()]).to(device))
                img_zR, img_zC = None, None
            elif len(drug_index[0])==1:
                img_zR = vq_vae._pre_vq_conv(vq_vae._encoder(dic_img_dic[drug_index[0].tolist()[0]].unsqueeze(0).to(device)))
                img_zC = vq_vae._pre_vq_conv(vq_vae._encoder(dic_img_dic[drug_index[1].tolist()[0]].unsqueeze(0).to(device)))
            else:
                img_zR = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[0].tolist()]).to(device)))
                img_zC = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[1].tolist()]).to(device)))
            pre_synergy, gerGene_loss, kl_loss = model(cell_emb,drugR_exp, drugC_exp, cell_exp, vq_vae._vq_vae._embedding.weight, img_zR, img_zC, drug_features=features)
            loss1 = 10*kl_loss+10*gerGene_loss+criterion(pre_synergy[:, 0] + TINY, synergyScore[:, 0])
            test_epoch_loss += loss1.item()
        preds = torch.cat((preds, pre_synergy.cpu().detach()), 0)
        trues = torch.cat((trues, synergyScore.cpu()), 0)
        num_batches += 1
    test_epoch_loss /= num_batches
    print("test_epoch_loss", test_epoch_loss)

    mse_synergy = mse(preds,trues)
    spearman_synergy = spearman(preds, trues)
    pearson_synergy = pearson(preds[:,0], trues[:,0])

    return mse_synergy,spearman_synergy,pearson_synergy

def train(train_loader, model, epoch, optimizer, device, scheduler,criterion,criterion2, vq_vae,dic_img_dic,print_freq=200):
    model.train()
    vq_vae.eval()
    losses = AverageMeter()
    epoch_loss = 0.0
    num_batches = 0

    preds = torch.Tensor()
    trues = torch.Tensor()

    for i, batch in tqdm(enumerate(train_loader), disable=False, desc="Traininng Epoch"):

        drugR_exp, drugC_exp, cell_exp,cell_emb, synergyScore, Y_r, Y_c, drug_index = batch
        drugR_exp, drugC_exp, cell_exp,cell_emb, synergyScore, Y_r, Y_c = drugR_exp.to(device), drugC_exp.to(device), cell_exp.to(device),cell_emb.to(device), synergyScore.to(device), Y_r.to(device), Y_c.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            img_zR = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[0].tolist()]).to(device)))
            img_zC = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[1].tolist()]).to(device)))
        pre_synergy, gerGene_loss, kl_loss = model(cell_emb,drugR_exp, drugC_exp, cell_exp, vq_vae._vq_vae._embedding.weight, img_zR, img_zC)

        lossa = criterion(pre_synergy[:, 0] + TINY, synergyScore[:, 0])
        loss1 = lossa+10*gerGene_loss+10*kl_loss

        loss=loss1
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        losses.update(loss.item(), len(drugR_exp))
        if np.isnan(loss.item()):
            raise Exception("Training model diverges.")
        preds = torch.cat((preds, pre_synergy.cpu().detach()), 0)
        trues = torch.cat((trues, synergyScore.cpu()), 0)
        if i % print_freq == 0:
            log_str = 'TRAIN -> Epoch{epoch}: \tIter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f})'.format(epoch=epoch, iter=i, loss=losses)
            print(f'{log_str}, pre_synergy:{lossa.data}, gerGene_loss:{gerGene_loss.data},kl_loss:{kl_loss.data}')

    epoch_loss /= num_batches
    print("epoch_loss", epoch_loss)
    return epoch_loss,model

def pearson(y, pred):
    pear = stats.pearsonr(y, pred)
    pear_value = pear[0]
    pear_p_val = pear[1]
    print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))
    return pear_value, pear_p_val

def spearman(y, pred):
    spear = stats.spearmanr(y, pred)
    spear_value = spear[0]
    spear_p_val = spear[1]
    print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
    return spear_value, spear_p_val

def mse(y, pred):
    err = mean_squared_error(y, pred)
    print("Mean squared error is {}".format(err))
    return err
