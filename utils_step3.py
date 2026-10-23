import copy
import random
from operator import itemgetter
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

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

def validate_new(valid_loader, device,model,scheduler,criterion,criterion2,epoch_loss,min_loss,num_epoch_no_gain,best_model,vq_vae,data_variance,best_model_vae,dic_img_dic):  # valid_loader
    model.eval()  # 设为评估模式
    vq_vae.eval()
    test_epoch_loss = 0.0
    num_batches = 0
    preds = torch.Tensor()
    trues = torch.Tensor()


    for i, batch in enumerate(valid_loader):
        drugR_exp, drugC_exp, cell_exp, cell_emb,synergyScore, Y_r, Y_c, drug_index = batch
        drugR_exp, drugC_exp, cell_exp, cell_emb,synergyScore, Y_r, Y_c = drugR_exp.to(device), drugC_exp.to(device), cell_exp.to(device),cell_emb.to(device), synergyScore.to(device), Y_r.to(device), Y_c.to(device)
        drug_index_list = torch.unique(torch.vstack(drug_index)).tolist()

        with torch.no_grad():
            loss_v = 0.0
            for batch in batch_sampler(drug_index_list, batch_size=max(1, len(drugR_exp) // 4)):
                if len(batch) == 1:
                    img = dic_img_dic[batch[0]].to(device)
                    img = img.unsqueeze(0)
                else:
                    img = torch.stack(itemgetter(*(batch))(dic_img_dic)).to(device)
                vq_loss, data_recon, perplexity, weight = vq_vae(img)
                recon_error = F.mse_loss(data_recon, img) / data_variance
                loss_v = vq_loss + recon_error + loss_v

            with torch.no_grad():
                img_zR = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[0].tolist()]).to(device)))
                img_zC = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[1].tolist()]).to(device)))
            pre_synergy, gerGene_loss, kl_loss = model(cell_emb,drugR_exp, drugC_exp, cell_exp, vq_vae._vq_vae._embedding.weight, img_zR, img_zC, False)
            loss1 = 10*kl_loss+10*gerGene_loss+10*loss_v+criterion(pre_synergy[:, 0] + TINY, synergyScore[:, 0])

            loss=loss1
            test_epoch_loss += loss.item()
        preds = torch.cat((preds, pre_synergy.cpu().detach()), 0)
        trues = torch.cat((trues, synergyScore.cpu()), 0)
        num_batches += 1
    test_epoch_loss /= num_batches
    print("val_epoch_loss", test_epoch_loss)

    if test_epoch_loss < min_loss:
        best_model = copy.deepcopy(model.state_dict())
        best_model_vae = copy.deepcopy(vq_vae.state_dict())
        min_loss = test_epoch_loss
        num_epoch_no_gain = 0
    else:
        num_epoch_no_gain += 1
    scheduler.step(epoch_loss)

    # roc_auc = roc_auc_score(trues, preds)

    mse_synergy = mse(preds,trues)
    spearman_synergy = spearman(preds, trues)
    pearson_synergy = pearson(preds[:,0], trues[:,0])

    return best_model,min_loss,num_epoch_no_gain,mse_synergy,best_model_vae

def test_new(test_loader, device,model,criterion,criterion2,vq_vae,data_variance,dic_img_dic):  # valid_loader
    model.eval()  # 设为评估模式
    vq_vae.eval()
    test_epoch_loss = 0.0
    num_batches = 0
    preds = torch.Tensor()
    trues = torch.Tensor()

    for i, batch in enumerate(test_loader):
        drugR_exp, drugC_exp, cell_exp,cell_emb, synergyScore, Y_r, Y_c, drug_index = batch
        drugR_exp, drugC_exp, cell_exp,cell_emb, synergyScore, Y_r, Y_c = drugR_exp.to(device), drugC_exp.to(device), cell_exp.to(device),cell_emb.to(device), synergyScore.to(device), Y_r.to(device), Y_c.to(device)
        drug_index_list = torch.unique(torch.vstack(drug_index)).tolist()

        with torch.no_grad():
            loss_v = 0.0
            if len(drugR_exp) // 4==0:
                batch_size_ = len(drugR_exp)
            else: batch_size_ = len(drugR_exp) // 4

            for batch in batch_sampler(drug_index_list, batch_size=batch_size_):
                if len(batch) == 1:
                    img = dic_img_dic[batch[0]].to(device)
                    img = img.unsqueeze(0)
                else:
                    img = torch.stack(itemgetter(*(batch))(dic_img_dic)).to(device)
                vq_loss, data_recon, perplexity, weight = vq_vae(img)
                recon_error = F.mse_loss(data_recon, img) / data_variance
                loss_v = vq_loss + recon_error + loss_v

            with torch.no_grad():
                #img_zR = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[0].tolist()]).to(device)))
                #img_zC = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[1].tolist()]).to(device)))
                if len(drug_index[0]) == 1:
                    img_zR = vq_vae._pre_vq_conv(
                        vq_vae._encoder(dic_img_dic[drug_index[0].tolist()[0]].unsqueeze(0).to(device)))
                else:
                    img_zR = vq_vae._pre_vq_conv(
                        vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[0].tolist()]).to(device)))
                if len(drug_index[0]) == 1:
                    img_zC = vq_vae._pre_vq_conv(
                        vq_vae._encoder(dic_img_dic[drug_index[1].tolist()[0]].unsqueeze(0).to(device)))
                else:
                    img_zC = vq_vae._pre_vq_conv(
                        vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[1].tolist()]).to(device)))
            pre_synergy, gerGene_loss, kl_loss = model(cell_emb,drugR_exp, drugC_exp, cell_exp,vq_vae._vq_vae._embedding.weight, img_zR, img_zC, False)

            loss1 = 10*kl_loss+10*gerGene_loss+10*loss_v+criterion(pre_synergy[:, 0] + TINY, synergyScore[:, 0])

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

def train(train_loader, model, epoch, optimizer, device, scheduler,criterion,criterion2, vq_vae,optimizer_v,data_variance,dic_img_dic,decay,print_freq=200):
    model.train()
    vq_vae.train()
    losses = AverageMeter()
    epoch_loss = 0.0
    num_batches = 0

    preds = torch.Tensor()
    trues = torch.Tensor()

    for i, batch in tqdm(enumerate(train_loader), disable=False, desc="Traininng Epoch"):

        drugR_exp, drugC_exp, cell_exp, cell_emb,synergyScore, Y_r, Y_c, drug_index = batch
        drugR_exp, drugC_exp, cell_exp, cell_emb,synergyScore, Y_r, Y_c = drugR_exp.to(device), drugC_exp.to(device), cell_exp.to(device),cell_emb.to(device), synergyScore.to(device), Y_r.to(device), Y_c.to(device)
        drug_index_list = torch.unique(torch.vstack(drug_index)).tolist()
        random.shuffle(drug_index_list)

        optimizer.zero_grad()
        optimizer_v.zero_grad()

        loss_v = 0.0

        for batch in batch_sampler(drug_index_list, batch_size=max(1, len(drugR_exp)//4)):
            if len(batch) == 1:
                img = dic_img_dic[batch[0]].to(device)
                img = img.unsqueeze(0)
            else:
                img = torch.stack(itemgetter(*(batch))(dic_img_dic)).to(device)
            vq_loss, data_recon, perplexity,weight = vq_vae(img)
            recon_error = F.mse_loss(data_recon, img) / data_variance
            loss_v = vq_loss + recon_error + loss_v

        with torch.no_grad():
            img_zR = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[0].tolist()]).to(device)))
            img_zC = vq_vae._pre_vq_conv(vq_vae._encoder(torch.stack([dic_img_dic[k] for k in drug_index[1].tolist()]).to(device)))
        pre_synergy,gerGene_loss,kl_loss = model(cell_emb,drugR_exp, drugC_exp, cell_exp, vq_vae._vq_vae._embedding.weight,img_zR,img_zC,False)

        lossa = criterion(pre_synergy[:, 0] + TINY, synergyScore[:, 0])
        loss1 = lossa+10*loss_v+10*gerGene_loss+10*kl_loss

        loss=loss1
        loss.backward()

        with torch.no_grad():
            if vq_vae._vq_vae._embedding.weight.grad is not None and decay>0:
                vq_vae._vq_vae._embedding.weight.grad *= 10 # 例如增加 0.1

        if i % print_freq//2 == 0:
            vqvae_grads = []
            main_model_grads = []
            for name, param in model.named_parameters():
                try:
                    main_model_grads.append(param.grad.norm().item())
                except:
                    pass
            for name, param in vq_vae.named_parameters():
                try:
                    vqvae_grads.append(param.grad.norm().item())
                except:
                    pass
            print(f"Epoch {epoch} - VQ-VAE gradients: {np.array(vqvae_grads).sum()}")
            print(f"Epoch {epoch} - codebook gradients: {vq_vae._vq_vae._embedding.weight.grad.norm().item()}")
            print(f"Epoch {epoch} - Main model gradients: {np.array(main_model_grads).sum()}")

        optimizer_v.step()
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
            print(f'{log_str}, pre_synergy:{lossa.data}, loss_v:{loss_v.data},gerGene_loss:{gerGene_loss.data},kl_loss:{kl_loss.data}')

    epoch_loss /= num_batches
    print("epoch_loss", epoch_loss)
    return epoch_loss,model

def batch_sampler(data, batch_size=64):
    data_len = len(data)
    num_batches = data_len // batch_size

    for i in range(num_batches):
        yield data[i * batch_size: (i + 1) * batch_size]

    if data_len % batch_size != 0:
        yield data[num_batches * batch_size:]

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
