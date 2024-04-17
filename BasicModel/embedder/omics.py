import torch
from torch import nn
import os
import torch.nn.functional as F
from ..utils.pe import select_pe_encoder
from ..utils import create_norm, create_activation
import numpy as np
from ..utils.sparse import sparse_normalize, sparse_tpm

class Embeds(nn.Module):
    def __init__(self,gene_list,gene_emb,num_hid,fix_embedding):
        super().__init__()
        if gene_emb is not None:
            self.emb = nn.Parameter(gene_emb, requires_grad=not fix_embedding)
        else:
            self.emb = nn.Parameter(torch.randn([len(gene_list), num_hid], dtype=torch.float32)*0.005)
            if fix_embedding:
                self.emb.requires_grad = False
    def forward(self,x):
        return self.emb

class OmicsEmbedder(nn.Module):
    def __init__(self, pretrained_gene_list, num_hid, gene_emb=None, fix_embedding=False,modality_list=['RNA']):
        super().__init__()
        self.pretrained_gene_list = pretrained_gene_list
        all_gene=[]
        self.gene_index_dicts={}
        for m in pretrained_gene_list.keys():
            all_gene=pretrained_gene_list[m]
            # print(m,len(all_gene))
            gene_index = dict(zip(all_gene, list(range(len(all_gene)))))
            self.gene_index_dicts[m]=gene_index
        self.num_hid = num_hid
        self.emb_dicts={}
        for m in modality_list:
            if gene_emb is not None:
                emb = nn.Parameter(gene_emb, requires_grad=not fix_embedding)
            else:
                emb = nn.Parameter(torch.randn([len(pretrained_gene_list[m]), num_hid], dtype=torch.float32)*0.005)
                if fix_embedding:
                    emb.requires_grad = False
            self.emb_dicts[m]=emb
        self.emb_dicts=nn.ParameterDict(self.emb_dicts)

    def forward(self, x_dict, input_gene_list=None,modality=None,preprocessing=False,saved_embs=''):
        if 'masked_x_seq' in x_dict:
            x = x_dict['masked_x_seq']
        else:
            x = x_dict['x_seq']

            
        if input_gene_list is not None:
            gene_idx = torch.tensor([self.gene_index_dicts[modality][o] for o in input_gene_list]).long()
        else:
            if x.shape[1] != len(self.pretrained_gene_list):
                raise ValueError('The input gene size is not the same as the pretrained gene list. Please provide the input gene list.')
            gene_idx = torch.arange(x.shape[1]).long()
        gene_idx = gene_idx.to(x.device)
        embs= self.emb_dicts[modality].to(gene_idx.device)
        feat = F.embedding(gene_idx, embs)
        feat = torch.sparse.mm(x, feat)
        return feat

class OmicsEmbeddingLayer(nn.Module):
    def __init__(self, gene_list, num_hidden, norm, activation='gelu', dropout=0.3, pe_type=None, cat_pe=True, gene_emb=None,
                 inject_covariate=False, batch_num=None,modality_list=['RNA']):
        super().__init__()

        self.pe_type = pe_type
        self.cat_pe = cat_pe
        self.act = nn.ReLU()#create_activation(activation)
        self.norm0 = create_norm(norm, num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.extra_linear = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            create_norm(norm, num_hidden),
        )
        if pe_type is not None:
            if cat_pe:
                num_emb = num_hidden // 2
            else:
                num_emb = num_hidden
            self.pe_enc = select_pe_encoder(pe_type)(num_emb)
        else:
            self.pe_enc = None
            num_emb = num_hidden

        if gene_emb is None:
            self.feat_enc = OmicsEmbedder(gene_list, num_emb,modality_list=modality_list)
        else:
            self.feat_enc = OmicsEmbedder(gene_list, num_emb, gene_emb,modality_list=modality_list)

        if inject_covariate:
            self.cov_enc = nn.Embedding(batch_num, num_emb)
            self.inject_covariate = True
        else:
            self.inject_covariate = False

    def forward(self, x_dict, input_gene_list=None,modality=None,saved_embs=''):
        
        x = self.feat_enc(x_dict, input_gene_list,modality=modality,saved_embs=saved_embs)#self.act(self.feat_enc(x_dict, input_gene_list))
        if self.pe_enc is not None:
            pe_input = x_dict[self.pe_enc.pe_key]
            pe = 0.#self.pe_enc(pe_input)
            if self.inject_covariate:
                pe = pe + self.cov_enc(x_dict['batch'])
            if self.cat_pe:
                x = torch.cat([x, pe], 1)
            else:
                x = x + pe
        x = self.extra_linear(x)
        return x
