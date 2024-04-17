import anndata as ad
import hdf5plugin
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
import math
import logging
import scanpy as sc
import wandb
import pickle
import argparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm import tqdm
from copy import deepcopy
from collections import Counter
# from dance.utils import set_seed
from scipy.sparse import csr_matrix
import pickle
import json
# import ipdb
import yaml
from sklearn import preprocessing
import os
import random

import torch.nn as nn
import torch.nn.functional as F

from BasicModel.utils.utils import get_idxs
from BasicModel.utils.eval import downstream_eval
from BasicModel.utils.data import XDict
from BasicModel.model import OmicsFormer
from BasicModel.utils.preprocessing import gene_preprocessing,protein_preprocessing

import time

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def set_seed(rndseed, cuda: bool = True, extreme_mode: bool = False):
    os.environ["PYTHONHASHSEED"] = str(rndseed)
    random.seed(rndseed)
    np.random.seed(rndseed)
    torch.manual_seed(rndseed)
    if cuda:
        torch.cuda.manual_seed(rndseed)
        torch.cuda.manual_seed_all(rndseed)
    if extreme_mode:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def buildNetwork(layers, dropouts, activation=nn.ReLU()):
    net = []
    for i in range(1, len(layers)):
        if dropouts[i-1] > 0:
            net.append(nn.Dropout(dropouts[i-1]))
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if i < len(layers) - 1:
            net.append(activation)
    net = nn.Sequential(*net)
    return net

class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,num_layers,hid_dim,dropout=0.):
        super().__init__()
        layers = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim] 
        dropouts = [dropout] * len(layers)
        self.model=buildNetwork(layers,dropouts)
        self.loss_func=nn.MSELoss()
    def forward(self,x,a=None):
        h = x['RNA']['x_seq'].to_dense()
        h=self.model(h)
        out_dict={}
        out_dict['pred']=h[x['RNA']['loss_mask'],:]
        out_dict['label']=x['RNA']['label'][x['RNA']['loss_mask'],:]
        # print(out_dict['label'])

        loss=self.loss_func(out_dict['pred'],out_dict['label'])
        return out_dict,loss

# def get_xdict()
def get_sampling(data_num,max_batch_size=2000,drop_ratio=0.7):
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(range(data_num))
    n_cells=min(max_batch_size,data_num*drop_ratio)
    return idx[:n_cells]

def get_gene_mask(pretrained_list,target_list):
        pretrained_list= pretrained_list #  gene_dicts[m]
        batch_name_list= target_list # modality_dict[m]['dicts'][batch_i][m]
        label_mask=[]
        label_name=[]
        for idx,i in enumerate(pretrained_list):
            if i in batch_name_list:
                label_mask.append(pretrained_list.index(i))
                label_name.append(i)
        return label_mask,label_name # gene_mask.tolist()

def main(seed,task=None):
    global config,input_modality,output_modality #  gene_list, batch_labels, seq_list, order_list, coord_list, label_list, train_idx, val_idx, test_idx,protein_list,time_string
    
    paired_data={}
    paired_data['RNA']=['protein']
    paired_data['protein']=['RNA']
    # print(gene_dicts['protein'])
    e_data,e_labels,e_gene_dicts,e_summary_list,e_test_idx,e_config=evaluate_init(config['name'],protein_num=134)
    b_data,b_labels,b_gene_dicts,b_summary_list,b_test_idx,b_config=evaluate_init(config['name'],protein_num=36)

    data_path=config['pretrain_data']
    with open(os.path.join(data_path,'data_infor.json'),'r') as f:
        data_infor=json.load(f)
    modality_dict={}
    for m in data_infor['modality']:
        modality_dict[m]={}
        modality_dict[m]['seq']=[[] for _ in range(data_infor['max_batch'])]
        modality_dict[m]['batch']=[[] for _ in range(data_infor['max_batch'])]
        modality_dict[m]['train_idx']=[[] for _ in range(data_infor['max_batch'])]
        modality_dict[m]['val_idx']=[[] for _ in range(data_infor['max_batch'])]
        modality_dict[m]['dicts']=[[] for _ in range(data_infor['max_batch'])]
        
    for files in os.listdir(data_path):
        if 'json' in files:continue
        for modality_file in os.listdir(os.path.join(data_path,files)):
            file_id=int(modality_file)
            loading_file=os.path.join(data_path,files,modality_file)
            modality_dict[files]['seq'][file_id]=torch.load(os.path.join(loading_file,'seq.pt'))
            modality_dict[files]['batch'][file_id]=torch.load(os.path.join(loading_file,'batch.pt'))
            modality_dict[files]['val_idx'][file_id]=torch.load(os.path.join(loading_file,'val_idx.pt'))
            modality_dict[files]['train_idx'][file_id]=torch.load(os.path.join(loading_file,'train_idx.pt'))
            with open(os.path.join(loading_file,'dicts.json'),'r') as f:
                modality_dict[files]['dicts'][file_id]=json.load(f)

        
    
    if task is None:
        task = config['head_type']
    device = torch.device('cuda')

    
    if args.small:
        proj_name='Mcellbert_small'
    else:
        proj_name='MCellbert3'
        if args.pretrain:
            proj_name+='_pretrained'
    
    
    group = str(args.latent_mod)+'_'+str(args.enc_mod)+'_'+str(args.enc_layers)+'_'+args.dataset+'_'+str(args.model_dropout)+'_'+str(args.head_num)+'_'+str(args.enc_hid)
    if args.alpha!=0.3:
        group+='_'+str(args.alpha)
    if args.cell_num!=30:
        group+='_'+str(args.cell_num)
    if args.wandb:
        wandb.login()
        wandb.init(group=group, config=config, project=proj_name)
        print('wandb login')
        
    model = OmicsFormer(**config,gene_list=gene_dicts)
    ### pretrain_modes
    optim = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['wd'])
    model.to(device)
    print(model)
    
    if config['scheduler'] == 'plat':
        scheduler = ReduceLROnPlateau(optim, 'min', patience=config['patience'], factor=0.95)

    eval_dict = {}
    train_loss = []
    valid_loss = []
    valid_pcc=[]
    valid_metric = []

    

    
    def get_xdict(m,batch_label_i,gene_dicts,selected_idx=None):
        x_dict_list=[]
        label=modality_dict[m]['seq'][batch_label_i].float()

        new_labels=torch.zeros((label.shape[0],len(gene_dicts[m])))
        label_mask=torch.zeros((label.shape[0],len(gene_dicts[m])))
        known_list=modality_dict[m]['dicts'][batch_label_i][m]
        label_mask_name=np.zeros(len(gene_dicts[m])).tolist()

        for idx,n in enumerate(gene_dicts[m]):
            if n in known_list:
                tmp=label[:,known_list.index(n)]
                new_labels[:,idx]=tmp
                label_mask[:,idx]=1
                label_mask_name[idx]=n
        label=new_labels
        

        if selected_idx is not None:
            in_label=label[selected_idx,:].float()
            in_loss_mark=torch.ones(label[selected_idx,:].shape[0]).bool()
            in_batch=modality_dict[m]['batch'][batch_label_i][selected_idx].long()
            seq_tmp=modality_dict[m]['seq'][batch_label_i][selected_idx,:].float().to(device)
        else:
            in_label=label.float()
            in_loss_mark=torch.ones(label.shape[0]).bool()
            in_batch=modality_dict[m]['batch'][batch_label_i].long()
            seq_tmp=modality_dict[m]['seq'][batch_label_i].float()
            # print(seq_tmp.shape)

        # label_idx,label_name=get_gene_mask(labels_name,batch_label_i)
        input_idx,input_name=get_gene_mask(gene_dicts[m],modality_dict[m]['dicts'][batch_label_i][m])
        input_dict = {
            'label': in_label.to(device),  # [cur].to(device),
            'loss_mask': in_loss_mark.to(device),  # [cur].to(device).bool(),
            'batch':in_batch.long().to(device),
            'label_mask':label_mask,
            'input_name_mask':input_idx,
            'input_name':input_name
        }
                        
        input_dict['x_seq'] = seq_tmp.to(device) # seq_list[i].to(device)
        x_dict = XDict(input_dict)
        x_dict_list.append((x_dict,input_name))
        return x_dict_list
    
    best_dicts1={}
    best_dicts2={}
    for m1 in input_modality:
        for m2 in output_modality:
            m=m1+'_'+m2
            best_dicts1[m+' PEARS']=0
            best_dicts2[m+' PEARS']=0

    for epoch in tqdm(range(config['epochs'])):
        epoch_loss = []
        train_score_dicts={}
        for m1 in input_modality:
            for m2 in output_modality:
                m=m1+'_'+m2
                train_score_dicts[m]=[]
        model.train()
        train_start_time=time.time()
        if epoch < 30 and config['scheduler'] != 'cos':
            for param_group in optim.param_groups[1:]:
                param_group['lr'] = config['lr'] * (epoch + 1) / 30

        for i in range(data_infor['max_batch']):
            selected_idx=get_sampling(modality_dict['RNA']['batch'][i].shape[0])
            x_dict,summary_list_x=get_xdict('RNA',i,gene_dicts,selected_idx=selected_idx)[0]
            y_dict,summary_list_y=get_xdict('protein',i,gene_dicts,selected_idx=selected_idx)[0]


            x_dicts={}
            x_dicts['RNA']=x_dict
            x_dicts['protein']=y_dict
            summary_list={}
            summary_list['RNA']=summary_list_x# modality_dict['RNA']['dicts'][i]['RNA']
            summary_list['protein']=summary_list_y# modality_dict['protein']['dicts'][i]['protein']
            tmp_out_dict, loss,lost_list = model(x_dicts, summary_list,alpha=args.alpha,input_m=input_modality,output_m=output_modality,beta=args.lbeta)
            
            all_loss=0
            for m1 in input_modality:
                for m2 in output_modality:
                    m=m1+'_'+m2
                    out_dict=tmp_out_dict[m]
                    # out_dict['label']=x_dicts[m2]['label']
                    with torch.no_grad():
                        # print(out_dict['pred'].shape, out_dict['label'].shape)
                        train_score_dicts[m].append(downstream_eval(task, out_dict['pred'], out_dict['label'], **eval_dict))

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()
            epoch_loss.append(loss.item())

            if config['scheduler'] == 'cos':
                scheduler.step()

            elif config['scheduler'] == 'plat':
                scheduler.step(loss.item())
        
        
        for m1 in input_modality:
            for m2 in output_modality:
                m=m1+'_'+m2
                train_scores_new = {}
                for k in train_score_dicts[m][0].keys():
                    train_scores_new[k] = []
                    for t in train_score_dicts[m]:
                        if np.isnan(t[k])==False: # for the test batches the metrics are nan
                            train_scores_new[k].append(t[k])
                    try:
                        train_scores_new[k] = sum(train_scores_new[k]) / len(train_scores_new[k])
                    except:
                        print('wrong',train_scores_new)
                        train_scores_new[k]=0
                train_score_dicts[m] = train_scores_new
        
        train_loss.append(sum(epoch_loss) / len(epoch_loss))
        del x_dicts
        train_end_time=time.time()

        with torch.no_grad():
            model.eval()
            epoch_loss = []
            valid_epoch = []
            valid_pcc_epoch=[]
            valid_pred_dict = {}
            valid_lb_dict = {}
            valid_score_dicts={}

            for m1 in input_modality:
                for m2 in output_modality:
                    m=m1+'_'+m2
                    valid_pred_dict[m]=[]
                    valid_lb_dict[m]=[] 
                    valid_score_dicts[m]={}       
            
            for i in range(data_infor['max_batch']):
                
                x_dict,summary_list_x=get_xdict('RNA',i,gene_dicts,selected_idx=None)[0]
                y_dict,summary_list_y=get_xdict('protein',i,gene_dicts,selected_idx=None)[0]

                x_dicts={}
                x_dicts['RNA']=x_dict
                x_dicts['protein']=y_dict
                summary_list={}
                summary_list['RNA']=summary_list_x# modality_dict['RNA']['dicts'][i]['RNA']
                summary_list['protein']=summary_list_y# modality_dict['protein']['dicts'][i]['protein']
                tmp_out_dict, loss,_ = model(x_dicts, summary_list,alpha=args.alpha,input_m=input_modality,output_m=output_modality,beta=args.lbeta)
                
                # print(memorized_cells)
                tmp_out_dict['label']=x_dict['label']
                
                val_idx=modality_dict['RNA']['val_idx']
                del x_dicts
                for m1 in input_modality:
                    for m2 in output_modality:
                        m=m1+'_'+m2

                        out_dict={'pred':[],'label':[]}
                        out_dict['pred'].append(tmp_out_dict[m]['pred'])
                        out_dict['label'].append(tmp_out_dict[m]['label'])

                        out_dict['label']=torch.cat(out_dict['label'],dim=0)
                        out_dict['pred']=torch.cat(out_dict['pred'],dim=0)

                        valid_scores = downstream_eval(task, out_dict['label'], out_dict['pred'], **eval_dict)
                        for key in valid_scores.keys():
                            if key not in valid_score_dicts[m]:
                                valid_score_dicts[m][key]=[]
                            valid_score_dicts[m][key].append(valid_scores[key])
                        valid_pred_dict[m].append(out_dict['pred'][val_idx[i],:])
                        valid_lb_dict[m].append(out_dict['label'][val_idx[i],:])

                
                epoch_loss.append(loss.item())
            tmp_score=[]
            tmp_pcc=[]

            
            for m1 in input_modality:
                for m2 in output_modality:
                    m=m1+'_'+m2
                    for key in valid_score_dicts[m].keys():
                        valid_score_dicts[m][key]=sum(valid_score_dicts[m][key])/len(valid_score_dicts[m][key])        
                    valid_scores=valid_score_dicts[m]
                    tmp_score.append(valid_scores['mse'])
                    tmp_pcc.append(valid_scores['pears'])
            valid_epoch.append(sum(tmp_score)/len(tmp_score))
            valid_pcc_epoch.append(sum(tmp_pcc)/len(tmp_pcc))
        valid_loss.append(sum(epoch_loss) / len(epoch_loss))
        valid_metric.append(sum(valid_pcc_epoch) / len(valid_epoch))
        

        print(f'Epoch {epoch} | Train loss: {train_loss[-1]:.4f} | Valid loss: {valid_loss[-1]:.4f}| Valid PCC: {valid_metric[-1]:.4f}')
        
        if max(valid_metric) == valid_metric[-1]:
        
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'valid_loss': valid_loss,
            }, f'../ckpt3/{config["name"]}/{config["name"]}.best.{epoch}.ckpt')
            final_epoch = epoch

            name1='all'
            name2='36 marker'
            eval_start_time=time.time()
            dicts1=evaluate(e_data,e_labels,e_gene_dicts,e_summary_list,device,model,e_test_idx,mark_name=name1)
            eval_end_time=time.time()
            dicts2=evaluate(b_data,b_labels,b_gene_dicts,b_summary_list,device,model,b_test_idx,mark_name=name2)
            for m1 in input_modality:
                for m2 in output_modality:
                    m=m1+'_'+m2
                    if dicts1[m+'_'+name1+'_test PEARS']>best_dicts1[m+' PEARS']:
                        best_dicts1[m+'_'+name1+' PEARS']=dicts1[m+'_'+name1+'_test PEARS']
                        # best_dicts1[m+'_'+name1+' MSE']=dicts1[m+'_'+name1+'_test MSE']
                        # best_dicts1[m+'_'+name1+' RMSE']=dicts1[m+'_'+name1+'_test RMSE']
                        best_dicts1[m+'_'+name1+' PEARS C']=dicts1[m+'_'+name1+'_test PEARS C']
                        best_dicts1[m+' PEARS']=dicts1[m+'_'+name1+'_test PEARS']
                    
                    if dicts2[m+'_'+name2+'_test PEARS']>best_dicts2[m+' PEARS']:
                        best_dicts2[m+'_'+name2+' PEARS']=dicts2[m+'_'+name2+'_test PEARS']
                        # best_dicts2[m+'_'+name2+' MSE']=dicts2[m+'_'+name2+'_test MSE']
                        # best_dicts2[m+'_'+name2+' RMSE']=dicts2[m+'_'+name2+'_test RMSE']
                        best_dicts2[m+'_'+name2+' PEARS C']=dicts2[m+'_'+name2+'_test PEARS C']
                        best_dicts2[m+' PEARS']=dicts2[m+'_'+name2+'_test PEARS']

        with open(args.enc_mod+'time.txt','a') as f:
            f.write(str(train_end_time-train_start_time)+'\t'+str(eval_end_time-eval_start_time)+'\n')

        if max(valid_metric) != max(valid_metric[-config['es']:]):
            print('Early stopped.')
            break
        dicts={}
        
        for m1 in input_modality:
            for m2 in output_modality:
                m=m1+'_'+m2
                dicts['Train MSE'+m]=train_score_dicts[m]['mse']
                dicts['Valid MSE'+m]=valid_score_dicts[m]['mse']
                dicts['Train PEARS'+m]=train_score_dicts[m]['pears']
                dicts['Valid PEARS'+m]=valid_score_dicts[m]['pears'] 
        # print(dicts)
        if args.wandb:
            wandb.log(best_dicts1)
            wandb.log(best_dicts2)
            wandb.log(dicts)
    

    wandb.finish()

def preprocess_op_2022(file_path,save_path,write_data=True):
    name='op2022'
    gene_adata=sc.read_h5ad(os.path.join(file_path,'train_cite_inputs_raw.h5ad'))
    protein_adata=sc.read_h5ad(os.path.join(file_path,'train_cite_targets_raw.h5ad'))
    protein_adata.obs['batch_labels']= protein_adata.obs['day']+protein_adata.obs['donor']
    gene_adata.obs['batch_labels']= gene_adata.obs['day']+gene_adata.obs['donor']
    gene_adata.var_names_make_unique()
    protein_adata.var_names_make_unique()
    g_s=[]
    for g in gene_adata.var.index:
        g_s.append(g.split('_')[1])
    gene_adata.var.index=g_s
    gene_adata=gene_preprocessing(gene_adata)
    protein_adata=protein_preprocessing(protein_adata)
    new_list=[]

    if write_data:
        protein_adata.write_h5ad(os.path.join(save_path,name,name+'protein.h5ad'))
        gene_adata.write_h5ad(os.path.join(save_path,name,name+'RNA.h5ad'))
    else:
        return gene_adata,protein_adata

def evaluate_init(name,protein_num):
    path= ''
    x,y=preprocess_op_2022(os.path.join(path,'data/cite/cite-seq/'),'../embs_2022/ori_data',write_data=False)
    print(x.obs['batch_labels'])
    print(x.X.shape)
    print(y.X.shape)
    x.var_names_make_unique()
    y.var_names_make_unique()


    sc.pp.highly_variable_genes(x, n_top_genes=5000, subset=True, flavor='seurat_v3')

    x.X=x.X.todense()
    # x.X=csr_matrix(x.X)

    y.X=y.X.toarray()

    x.obs['split']='test'
    y.obs['split']='test'


    gene_dicts={}
    with open(f'../ckpt3/{name}/dicts.json','r') as f:
        gene_dicts=json.load(f)
    
    pretrained_gene_list=gene_dicts['RNA']
    pretrained_protein_list=gene_dicts['protein']

    x_list=x.var.index.tolist()
    y_list=y.var.index.tolist()

    common_genes=list(set(x_list).intersection(set(pretrained_gene_list)))
    common_proteins=list(set(y_list).intersection(set(pretrained_protein_list)))
    if protein_num==36:
        common_proteins=['TIGIT', 'CD24', 'CD58', 'KLRG1', 'CD28', 'CD82', 'CD86', 'HLA-E', 'CD2', 'CD40', 'CX3CR1', 'CD244', 'CD33', 'CD163', 'CD14', 'CD47', 'CD52', 'CD274', 'CD45RA', 'CD19', 'CD38', 'CD226', 'CD27', 'CD81', 'CD22', 'CD5', 'CD69', 'CD83', 'CD48', 'CD101', 'CD7', 'CD36', 'CD44', 'CD45RO', 'CD4', 'HLA-DR']
    if protein_num==117:
        common_proteins=['CD335', 'CD224', 'CD112', 'CD8', 'CD40', 'CD185', 'CD163', 'CD169', 'CD27', 'CD107a', 'CD49f', 'CD54', 'CD16', 'CD314', 'CD31', 'CD56', 'CD28', 'CD11a', 'CD44', 'CD119', 'CD146', 'CD279', 'CD79b', 'CD11b', 'CD45', 'CD1c', 'CD152', 'LOX-1', 'CD24', 'CD244', 'CD2', 'CD21', 'CD161', 'CD49b', 'CD33', 'CD62P', 'CD85j', 'CD83', 'CD81', 'CD73', 'CD36', 'CD42b', 'CD71', 'CD274', 'CD158b', 'CD328', 'CD88', 'CD95', 'HLA-DR', 'CD154', 'CD223', 'CD64', 'CD134', 'CD1d', 'CD35', 'CD13', 'CD62L', 'CD86', 'CD352', 'CD158', 'CD122', 'FceRIa', 'CD18', 'CD196', 'CD11c', 'CD3', 'CD29', 'CD25', 'CD41', 'CD47', 'KLRG1', 'CD272', 'CD45RA', 'CD69', 'CD49d', 'CD4', 'IgM', 'TIGIT', 'CD155', 'CD49a', 'CD82', 'CD195', 'CD38', 'CD52', 'CD226', 'CD268', 'CD319', 'CD48', 'CD158e1', 'CD45RO', 'CD14', 'CD26', 'CX3CR1', 'CD32', 'CD137', 'CD124', 'CD20', 'CD101', 'CD19', 'CD103', 'IgD', 'CD22', 'CD194', 'CD5', 'CD7', 'CD123', 'CD57', 'CD23', 'CD270', 'CD127', 'CD141', 'HLA-E', 'CD105', 'CD39', 'CD94', 'CD303', 'CD58']

    x=x[:,common_genes]
    y=y[:,common_proteins]

    labels=y.X
    filled_cells=[]
    selected_cells=[]
    for i in range(labels.shape[0]):
        if np.sum(labels[i,:])==0:
            filled_cells.append(i)
        else:
            selected_cells.append(i)
    print(filled_cells)
    y=y[selected_cells,:]
    x=x[selected_cells,:]

    gene_list=common_genes
    protein_list=common_proteins

    summary_list={}
    summary_list['RNA']=gene_list
    summary_list['protein']=protein_list
    print(f'../ckpt3/{name}/{name}.json')
    with open(f'../ckpt3/{name}/{name}.json','r') as f:
        config=json.load(f)


    batch_labels = LabelEncoder().fit_transform(x.obs['batch_labels'])
    data=x
    labels=y.X # .todense()
    # print(labels)

    train_idx,val_idx,test_idx=get_idxs(x_data=x,batch_labels=batch_labels,seed=args.seed)

    

    return data,labels,gene_dicts,summary_list,test_idx,config



def evaluate(data,labels,gene_dicts,summary_list,device,model,test_idx,mark_name='none'):
    batch_labels = LabelEncoder().fit_transform(data.obs['batch_labels'])
    

    seq_list = []
    batch_list = []
    order_list = []
    coord_list = []
    label_list = []
    batch_id_list=[]
    gene_mask_list=[]
    protein_mask_list=[]
    label_masks_list=[]

    pretrained_proteins=gene_dicts['protein']
    protein_list=summary_list['protein']

    pretrained_gene_list=gene_dicts['RNA']
    gene_list=summary_list['RNA']


    data_hvg = data.copy()
    ori_label=[]
    target_seq_list=[]
    for batch in tqdm(range(batch_labels.max() + 1)):
        x = data_hvg[batch_labels == batch].X.astype(float)
        # seq_list.append(create_sparse_tensor(x))
        seq_list.append(torch.from_numpy(x).float())
        batch_list.append(torch.from_numpy(batch_labels[batch_labels == batch]))
        coord_list.append(torch.zeros(x.shape[0], 2) - 1)
        get_label=labels[batch_labels == batch].astype(float)# # .todense()
        
        new_labels=np.zeros((get_label.shape[0],len(pretrained_proteins)))
        label_mask=[]
        for i,n in enumerate(pretrained_proteins):
            if n in protein_list:
                tmp=get_label[:,protein_list.index(n)]
                tmp=np.squeeze(tmp)
                new_labels[:,i]=tmp # np.squeeze(tmp)
                label_mask.append(i)
        label_list.append(torch.from_numpy(new_labels).float())
        ori_label.append(torch.from_numpy(get_label).float())

        new_x=np.zeros((x.shape[0],len(pretrained_gene_list)))
        label_mask=[]
        for i,n in enumerate(pretrained_gene_list):
            if n in gene_list:
                tmp=x[:,gene_list.index(n)]
                tmp=np.squeeze(tmp)
                new_x[:,i]=tmp # np.squeeze(tmp)
                label_mask.append(i)
        target_seq_list.append(torch.from_numpy(new_x).float())


        batch_id_list.append(torch.zeros([x.shape[0]]).long() + int(batch))
        gene_mask_list.append(torch.ones([x.shape[1]]).bool())
        protein_mask_list.append(torch.ones([new_labels.shape[1]]).bool())

        
        
        label_masks_list.append(label_mask)
    eval_dict = {}
    test_metric = {}
    test_pcc={}
    test_pcc_c={}
    test_metric_rmse={}
    best_dicts={}
    final_test = -1
    final_epoch = -1

    epoch_loss = []
    with torch.no_grad():
        model.eval()
        epoch_loss = []
        test_pred={}
        test_lb={}
        for m1 in input_modality:
            for m2 in output_modality:
                m=m1+'_'+m2
                test_pred[m]=[]
                test_lb[m]=[]
                test_metric[m]=[]
                test_metric_rmse[m]=[]
                test_pcc[m]=[]
                test_pcc_c[m]=[]

                best_dicts[m]={}
                best_dicts[m]['test PEARS']=0
        for i in range(len(seq_list)):
            seq_flag=0
            out_dict={'pred':[],'label':[]}

            input_idx,input_name=get_gene_mask(gene_dicts['RNA'],summary_list['RNA'])
            input_dict = {
                'coord': coord_list[i].to(device),  # [cur].to(device),
                'label': target_seq_list[i].to(device),  # [cur].to(device),
                'loss_mask': torch.ones([label_list[i].shape[0]]).to(device).bool(),
                'batch':batch_id_list[i].to(device),
                'gene_mask':gene_mask_list[i].to(device),
                'input_name_mask':input_idx,
                'input_name':input_name
            }
            input_dict['x_seq'] = seq_list[i].to(device)  # .index_select(0, cur.to(device))
            x_dict = XDict(input_dict)

            input_idx,input_name=get_gene_mask(gene_dicts['protein'],summary_list['protein'])

            input_dict = {
                'coord': coord_list[i].to(device),  # [cur].to(device),
                'label': label_list[i].to(device),  # [cur].to(device),
                'loss_mask': torch.ones([label_list[i].shape[0]]).to(device).bool(),
                'batch':batch_id_list[i].to(device),
                'gene_mask':protein_mask_list[i].to(device),
                'input_name_mask':input_idx,
                'input_name':input_name
            }
            input_dict['x_seq'] = ori_label[i].to(device)  # .index_select(0, cur.to(device))
            
            y_dict = XDict(input_dict)
            x_dicts={}
            x_dicts['RNA']=x_dict
            x_dicts['protein']=y_dict
            tmp_out_dict, loss,_ = model(x_dicts, summary_list,input_m=input_modality,output_m=output_modality,alpha=args.alpha,infer=True,beta=args.lbeta)
            del x_dicts
            for m1 in input_modality:
                for m2 in output_modality:
                    m=m1+'_'+m2
                    tmp_dict=tmp_out_dict[m]
                    preds=tmp_dict['pred'][test_idx[i],:]
                    label=tmp_dict['label'][test_idx[i],:]
                    test_pred[m].append(preds)
                    test_lb[m].append(label)

        for m1 in input_modality:
            for m2 in output_modality:
                m=m1+'_'+m2
                if m2=='RNA':
                    mask=x_dict['input_name_mask']
                elif m2=='protein':
                    mask=y_dict['input_name_mask']
                # print(m2,mask)
                test_scores = downstream_eval('predmodality', torch.cat(test_pred[m])[:,mask], torch.cat(test_lb[m])[:,mask], **eval_dict)
                # print(torch.cat(test_pred[m]).shape)
                # print(test_scores)
                test_metric[m].append(test_scores['mse'])
                test_metric_rmse[m].append(test_scores['rmse'])
                test_pcc[m].append(test_scores['pears'])
                test_pcc_c[m].append(test_scores['pears_c'])
        dicts={}
        for m1 in input_modality:
            for m2 in output_modality:
                m=m1+'_'+m2
                dicts[m+'_'+mark_name+'_test MSE']=sum(test_metric[m])/len(test_metric[m])
                dicts[m+'_'+mark_name+'_test RMSE']=sum(test_metric_rmse[m])/len(test_metric_rmse[m])

                dicts[m+'_'+mark_name+'_test PEARS']=sum(test_pcc[m])/len(test_pcc[m])
                dicts[m+'_'+mark_name+'_test PEARS C']=sum(test_pcc_c[m])/len(test_pcc_c[m])

        return dicts

def create_sparse_tensor(x):
    return torch.sparse_csr_tensor(x.indptr, x.indices, x.data, (x.shape[0], x.shape[1])).to_sparse().float().coalesce()


if __name__ == '__main__':
    # print('hello')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2021')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--lbeta", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--pre_model", type=str, default='20231211_75M_50M_OPL3')
    parser.add_argument("--config_file", type=str, default='t_test')
    parser.add_argument("-m", "--model_folder", default="./models")
    parser.add_argument("--outdir", "-o", default="./logs", help="Directory to output to")
    parser.add_argument("-seed", "--rnd_seed", default=10, type=int)
    parser.add_argument('--subtask',default='openproblems_2022_cite_gex2adt',type=str)
    parser.add_argument('--device',default='cuda:0',type=str)
    parser.add_argument('--latent_mod',default='memory',type=str)
    parser.add_argument('--enc_mod',default='flowformer',type=str)# transformer
    parser.add_argument('--pretrain',action="store_true")
    parser.add_argument('--pretrained_model',action="store_true")
    parser.add_argument('--span',default=1.0,type=float)
    parser.add_argument('--model_dropout',default=0.3,type=float)
    parser.add_argument('--wd',default=1e-7,type=float)
    parser.add_argument('--lr',default=5e-4,type=float)
    parser.add_argument('--alpha',default=0.3,type=float)
    parser.add_argument('--cell_num',default=30,type=int)
    parser.add_argument('--topk',default=10,type=int)
    parser.add_argument('--head_num',default=4,type=int)
    parser.add_argument('--enc_layers',default=2,type=int)
    parser.add_argument('--enc_hid',default=1024,type=int)
    parser.add_argument('--add_cell',default=0,type=int)
    parser.add_argument('--wandb',default=True,type=bool)
    parser.add_argument('--small',action="store_true")
    parser.add_argument('--ckpt',default=1,type=int)
    parser.add_argument("-cpu", "--cpus", default=1, type=int)
    args = parser.parse_args()
    

    config={}
    config['pretrain_data']=args.dataset
    config['es'] = 2000
    config['lr'] = args.lr# 5e-4# 5e-4
    config['wd'] = args.wd
    config['scheduler'] = 'plat'
    config['drop_node_rate'] =0.0 # 0.3
    config['dec_layers'] = 3
    config['model_dropout'] = args.model_dropout # 0.5
    config['mask_node_rate'] = 0.0 # 0.75
    config['mask_feature_rate'] = 0.0 # 0.25

    config['dec_mod'] = 'nbmlp'
    config['dec_layers'] = 2
    config['dec_hid'] = 1024
    
    config['epochs'] = args.epochs
    # config['head_type'] = 'predmodality'
    config['head_type'] = None
    
    config['max_batch_size'] = 70000
    config['hvg'] = 5000
    config['patience'] = 10
    config['topk'] = args.topk
    config['modality_list']=['RNA','protein']
    config['dependence']=False
    config['contrastive_method']=False
    config['add_cell']=args.add_cell
    config['cluster']=False
    config['split']=True
    config['downstream']=True
    config['split']=False
    config['head_num']=args.head_num
    config['post_latent_dim']=1024

    config['cell_num']=args.cell_num
    config['enc_layers']=args.enc_layers
    config['enc_hid']=args.enc_hid
    config['latent_mod'] = args.latent_mod
    config['enc_mod']=args.enc_mod
    config['ckpt_list']=[0]

    config['lamda']=5
    config['norm']='groupnorm'
    config['architecture']='OmicsFormer'
    config['w_li']=5.
    config['w_en']=5.
    config['cat_pe']=False
    config['dar']=False
    config['input_covariate']=False
    config['beta']=[2.,1.5]
    config['batch_mixing']=False
    config['ecs']=False
    config['dsbn']=False
    config['dae']=True
    config['batch_flip']=True
    config['mask_beta']=True
    config['lbeta']=args.lbeta
    
    set_seed(args.seed)

    args.resume = True

    torch.set_num_threads(args.cpus)
    rndseed = args.rnd_seed
    with open(os.path.join(args.dataset,'dicts.json'),'r') as f:
        gene_dicts=json.load(f)

    out_dim={'RNA':len(gene_dicts['RNA']),'protein':len(gene_dicts['protein'])}
    # print(out_dim.shape)
    config['out_dim'] = out_dim

    

    input_modality=['RNA','protein']
    output_modality=['RNA','protein']

    time_string=time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))+'_'+str(args.alpha)+'_'+str(args.cell_num)+'_'+str(args.latent_mod)+'_'+str(args.enc_mod)+'_'+str(args.enc_layers)+'_'+str(args.seed)+'.'+args.dataset
    config['name']=time_string
    if os.path.exists(f'../ckpt3/{time_string}')==False:
        os.makedirs(f'../ckpt3/{time_string}')
    with open(f'../ckpt3/{time_string}/{time_string}.json','w') as f:
        json.dump(config,f)
    with open(f'../ckpt3/{time_string}/dicts.json','w') as f:
        json.dump(gene_dicts,f)
    
    main(args.seed,'predmodality')

