import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

class MaskedLinear(nn.Linear):
    def __init__(self, mask, bias=True):
        super(MaskedLinear, self).__init__(mask.shape[0], mask.shape[1], bias)

        mask = Variable(torch.Tensor(mask.T))
        self.register_buffer('mask', mask)

        self.iter = 0

    def forward(self, input):
        masked_weight = self.weight * self.mask

        output = torch.einsum('ij,bjk->bik', [masked_weight, input])  # torch.matmul(self.weight * self.mask, input)

        return output + self.bias.unsqueeze(0).unsqueeze(-1)

class DrugCellLineEncoder(nn.Module):
    def __init__(self, input_size, layer_sizes, inDrop=0.1, drop=0.3):
        super(DrugCellLineEncoder, self).__init__()
        layers = []
        for i in range(len(layer_sizes)):
            if i == 0:
                layers.append(nn.Sequential(
                    nn.Linear(input_size, layer_sizes[i]),
                    nn.BatchNorm1d(layer_sizes[i]),
                    nn.ReLU(),
                    nn.Dropout(inDrop)
                ))
            elif i == (len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[i - 1], layer_sizes[i]),
                    nn.BatchNorm1d(layer_sizes[i]),
                    nn.ReLU(),
                    nn.Dropout(drop)
                ))

        # Apply He normal initialization within the Encoder module
        self.encoder = nn.Sequential(*layers)
        self.apply(self.he_normal_init)

    def forward(self, x):
        return self.encoder(x)

    def he_normal_init(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)


class IC50Predictor(nn.Module):
    def __init__(self, input_size, layer_sizes, drop=0.3):
        super(IC50Predictor, self).__init__()
        layers = []
        for i in range(len(layer_sizes)):
            if len(layer_sizes) == 1:
                layers.append(nn.Sequential(
                    nn.Linear(input_size, layer_sizes[i]),
                    nn.BatchNorm1d(layer_sizes[i]),
                    nn.ReLU(),
                    nn.Linear(layer_sizes[i], 1),
                    nn.Sigmoid()
                ))
            else:
                if i == 0:
                    layers.append(nn.Sequential(
                        nn.Linear(input_size, layer_sizes[i]),
                        nn.BatchNorm1d(layer_sizes[i]),
                        nn.ReLU(),
                        nn.Dropout(drop)
                    ))
                elif i == (len(layer_sizes) - 1):
                    layers.append(nn.Sequential(
                        nn.Linear(layer_sizes[i - 1], layer_sizes[i]),
                        nn.BatchNorm1d(layer_sizes[i]),
                        nn.ReLU(),
                        nn.Linear(layer_sizes[i], 1),
                        nn.Sigmoid()
                    ))
                else:
                    layers.append(nn.Sequential(
                        nn.Linear(layer_sizes[i - 1], layer_sizes[i]),
                        nn.BatchNorm1d(layer_sizes[i]),
                        nn.ReLU(),
                        nn.Dropout(drop)
                    ))

        self.predictor = nn.Sequential(*layers)
        self.apply(self.he_normal_init)

    def forward(self, x):
        return self.predictor(x)

    def he_normal_init(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

class SynergyPredictor(nn.Module):
    def __init__(self, input_size, layer_sizes, drop=0.3):
        super(SynergyPredictor, self).__init__()
        layers = []
        for i in range(len(layer_sizes)):
            if len(layer_sizes) == 1:
                layers.append(nn.Sequential(
                    nn.Linear(input_size, layer_sizes[i]),
                    nn.BatchNorm1d(layer_sizes[i]),
                    nn.ReLU(),
                    nn.Linear(layer_sizes[i], 1)
                ))
            else:
                if i == 0:
                    layers.append(nn.Sequential(
                        nn.Linear(input_size, layer_sizes[i]),
                        nn.BatchNorm1d(layer_sizes[i]),
                        nn.ReLU(),
                        nn.Dropout(drop)
                    ))
                elif i == (len(layer_sizes) - 1):
                    layers.append(nn.Sequential(
                        nn.Linear(layer_sizes[i - 1], layer_sizes[i]),
                        nn.BatchNorm1d(layer_sizes[i]),
                        nn.ReLU(),
                        nn.Linear(layer_sizes[i], 1)
                    ))
                else:
                    layers.append(nn.Sequential(
                        nn.Linear(layer_sizes[i - 1], layer_sizes[i]),
                        nn.BatchNorm1d(layer_sizes[i]),
                        nn.ReLU(),
                        nn.Dropout(drop)
                    ))

        self.predictor = nn.Sequential(*layers)
        self.apply(self.he_normal_init)

    def forward(self, x):
        return self.predictor(x)

    def he_normal_init(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)


class GeneToFeature(nn.Module):
    def __init__(self):
        super(GeneToFeature, self).__init__()
        self.trainable_matrix = nn.Parameter(torch.randn(1, 574, 64))

    def forward(self, input_vector):

        input_vector = input_vector.unsqueeze(-1)

        output_matrix = input_vector * self.trainable_matrix

        return output_matrix

class MultiHeadAttentionInteract(nn.Module):

    def __init__(self, in_embed_size, out_embed_size, head_num, dropout, residual=True):
        """
        """
        super(MultiHeadAttentionInteract, self).__init__()
        self.in_embed_size = in_embed_size
        self.out_embed_size = out_embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = self.out_embed_size // head_num

        self.W_Q = nn.Parameter(torch.Tensor(self.in_embed_size, self.out_embed_size))
        self.W_K = nn.Parameter(torch.Tensor(self.in_embed_size, self.out_embed_size))
        self.W_V = nn.Parameter(torch.Tensor(self.in_embed_size, self.out_embed_size))

        if self.use_residual:
            self.W_R = nn.Parameter(torch.Tensor(self.in_embed_size, self.out_embed_size))

        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """

        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))

        Query = torch.stack(torch.split(Query, self.attention_head_size, dim=2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim=2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim=2))

        inner = torch.matmul(Query, Key.transpose(-2, -1))
        inner = inner / self.attention_head_size ** 0.5

        attn_w = F.softmax(inner, dim=-1)

        attn_w = F.dropout(attn_w, p=self.dropout)

        results = torch.matmul(attn_w, Value)

        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)  # (bs, fields, D)

        if self.use_residual:
            results = results + F.relu(torch.tensordot(x, self.W_R, dims=([-1], [0])))

        return results, attn_w

class LearnableGaussianNoise(nn.Module):
    def __init__(self, input_dim, std=0.1):
        super(LearnableGaussianNoise, self).__init__()
        self.mean = nn.Parameter(torch.zeros(input_dim))
        self.std = nn.Parameter(torch.ones(input_dim) * std)

    def forward(self, x):
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise*0.3

class LearnableNoise(nn.Module):
    def __init__(self, gene_num, input_dim):
        super(LearnableNoise, self).__init__()

        self.noise = nn.Parameter(torch.Tensor(1,gene_num, input_dim))

    def forward(self, x):

        return x + self.noise*0.3

class MultiHeadAttention_Codebook2Genebook(nn.Module):
    def __init__(self, in_embed_size, out_embed_size, head_num, dropout, gene_path_size, codebook_num,residual=True):
        """
        """
        super(MultiHeadAttention_Codebook2Genebook, self).__init__()
        self.in_embed_size = in_embed_size
        self.out_embed_size = out_embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = self.out_embed_size // head_num

        self.Query = nn.Parameter(torch.Tensor(gene_path_size[0], self.out_embed_size))
        self.W_K = nn.Parameter(torch.Tensor(self.in_embed_size, self.out_embed_size))
        self.W_V = nn.Parameter(torch.Tensor(self.in_embed_size, self.out_embed_size))

        self.input_transform = nn.Parameter(torch.Tensor(codebook_num, gene_path_size[0]))

        if self.use_residual:
            self.W_R = nn.Parameter(torch.Tensor(self.in_embed_size, self.out_embed_size))

        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

        self.code2gene = nn.Sequential(
            nn.LayerNorm(self.in_embed_size),
            nn.Linear(self.in_embed_size, self.in_embed_size*2),
            nn.ReLU(),
            nn.LayerNorm(self.in_embed_size*2),
            nn.Linear(self.in_embed_size*2, self.in_embed_size),
            nn.ReLU(),
            nn.LayerNorm(self.in_embed_size),
        )

        self.noise = LearnableGaussianNoise(self.in_embed_size)  # 给codebook的

        self.norm = nn.LayerNorm(self.in_embed_size)

    def forward(self, x_, isstep3):
        if isstep3:
            x = self.noise(self.code2gene(x_).unsqueeze(0))
        else:
            x = self.code2gene(x_).unsqueeze(0)
        Query = self.Query.unsqueeze(0).repeat(x.shape[0], 1, 1)
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))

        # Head (head_num, bs, fields, D / head_num)
        Query = torch.stack(torch.split(Query, self.attention_head_size, dim=2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim=2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim=2))

        inner = torch.matmul(Query, Key.transpose(-2, -1))
        inner = inner / self.attention_head_size ** 0.5

        attn_w = F.softmax(inner, dim=-1)

        attn_w = F.dropout(attn_w, p=self.dropout)

        results = torch.matmul(attn_w, Value)

        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)  # (bs, fields, D)

        if self.use_residual:
            results = results + F.relu(torch.tensordot(F.relu(torch.tensordot(x, self.W_R, dims=([-1], [0]))).permute(0, 2, 1), self.input_transform, dims=([-1], [0]))).permute(0, 2, 1)  # (3, batch_size, d_model)

        results = self.norm(results)

        return results, attn_w

class ConvFlattenModel(nn.Module):
    def __init__(self):
        super(ConvFlattenModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(4*28*28, 574)
        self.layernorm = nn.LayerNorm(574)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        x = x.view(x.size(0), -1)  # batch*16*56*56 -> batch*(16*56*56)

        x = self.fc(x)  # batch*(16*56*56) -> batch*574
        x = self.layernorm(x)
        x = self.relu3(x)
        return x

class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim):
        super(CVAE, self).__init__()
        x_dim = input_dim
        c_dim = cond_dim
        latent_dim = latent_dim
        hidden_dim = 256
        self.encoder_fc1 = nn.Linear(x_dim + c_dim, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim + c_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, x_dim)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def encode(self, x, c):
        h = torch.cat([x, c], dim=1)
        h = F.gelu(self.encoder_fc1(h))
        h = F.gelu(self.encoder_fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h = torch.cat([z, c], dim=1)
        h = F.gelu(self.decoder_fc1(h))
        h = F.gelu(self.decoder_fc2(h))
        return self.decoder_out(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, z, mu, logvar

class DrugCombinationModel(nn.Module):
    def __init__(self, pathway_size, genebook_dim, path_atten_outDim, head_num, layers, config, add_model, Gene_pathway, dropout, num_bookembeddings, fine_tune=False):
        super(DrugCombinationModel, self).__init__()

        self.genebaseFeatures = MultiHeadAttention_Codebook2Genebook(genebook_dim, genebook_dim, head_num, dropout, Gene_pathway.shape,num_bookembeddings)
        self.bn = nn.BatchNorm1d(genebook_dim)
        self.ln = nn.LayerNorm(genebook_dim)

        drugCell_mask = np.vstack((Gene_pathway, Gene_pathway))
        self.cell_drugMaskedFC = MaskedLinear(drugCell_mask)

        self.path_atten = MultiHeadAttentionInteract(genebook_dim, path_atten_outDim, head_num, dropout)

        self.encoder = DrugCellLineEncoder(pathway_size*path_atten_outDim, layers['encoder_layers'], 0.1, dropout)

        residual_layer = layers['encoder_layers'].copy()
        if add_model == 0:
            residual_layer[-1] = 2 * layers['encoder_layers'][-1]  # 拼接
        self.residual = DrugCellLineEncoder(pathway_size*path_atten_outDim, residual_layer, 0.1, dropout)

        encoder_out_size = layers['encoder_layers'][-1]

        if add_model == 0:
            synergy_in_size = 2 * encoder_out_size
        elif add_model == 1:
            synergy_in_size = encoder_out_size

        self.synergy_predictor = SynergyPredictor(synergy_in_size, layers['synergy_layers'], dropout)

        if fine_tune:
            for param in self.synergy_predictor.parameters():
                param.requires_grad = False

        self.config = config
        self.add_model = add_model

        self.img2drugGeneRelation = ConvFlattenModel()

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
        )

        self.cvae = CVAE(input_dim=128, cond_dim=256, latent_dim=synergy_in_size)


    def forward(self, cell_emb, drugR_exp, drugC_exp, cell_exp, book,img_zR,img_zC,isstep3=True,drug_features=None):

        cemb = self.mlp(cell_emb)

        G_feature,drugGeneR_attn = self.genebaseFeatures(self.ln(book),False)

        if drug_features is None:
            drugRG_de = self.img2drugGeneRelation(img_zR)
            drugCG_de = self.img2drugGeneRelation(img_zC)
        else:
            drugRG_de, drugCG_de = drug_features

        drugR_p_feature = drugRG_de.unsqueeze(-1) * G_feature
        drugC_p_feature = drugCG_de.unsqueeze(-1) * G_feature
        cell_p_feature = cell_exp.unsqueeze(-1) * G_feature

        DER0 = torch.all(drugR_exp == 0.0, dim=1)
        DEC0 = torch.all(drugC_exp == 0.0, dim=1)
        gerGene_loss = drugRG_de.new_zeros(())
        if (~DER0).any():
            gerGene_loss = gerGene_loss + F.mse_loss(drugRG_de[~DER0], drugR_exp[~DER0])
        if (~DEC0).any():
            gerGene_loss = gerGene_loss + F.mse_loss(drugCG_de[~DEC0], drugC_exp[~DEC0])

        drugR_cell_exp = torch.concat([drugR_p_feature, cell_p_feature], dim=1)
        drugC_cell_exp = torch.concat([drugC_p_feature, cell_p_feature], dim=1)
        drugR_pathways = F.tanh(self.ln(self.cell_drugMaskedFC(drugR_cell_exp)))
        drugC_pathways = F.tanh(self.ln(self.cell_drugMaskedFC(drugC_cell_exp)))
        drugR_pathways,pathwayR_attn = self.path_atten(drugR_pathways)
        drugC_pathways,pathwayC_attn = self.path_atten(drugC_pathways)
        drug_row_feats = self.encoder(drugR_pathways.reshape(drugR_pathways.size(0), -1))
        drug_col_feats = self.encoder(drugC_pathways.reshape(drugC_pathways.size(0), -1))

        residual = self.residual(drugR_pathways.reshape(drugR_pathways.size(0), -1)+drugC_pathways.reshape(drugC_pathways.size(0), -1))
        
        if self.add_model == 0:
            concat_feats = torch.cat([drug_row_feats, drug_col_feats], dim=1)
        elif self.add_model == 1:
            concat_feats = (drug_row_feats + drug_col_feats)/2

        de_concat_feats, z, mu, logvar = self.cvae(concat_feats+residual, cemb)

        recon_loss = F.mse_loss(de_concat_feats, concat_feats)  # 可选
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        synergy_loewe = self.synergy_predictor(z)
        return synergy_loewe, gerGene_loss, recon_loss + kl_loss
