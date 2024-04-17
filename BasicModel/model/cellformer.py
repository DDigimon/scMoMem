import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from ..embedder import OmicsEmbeddingLayer
from ..utils.mask import MaskBuilder, NullMaskBuilder, HiddenMaskBuilder
from ..encoder import setup_encoder
from ..decoder import setup_decoder
from ..latent import LatentModel, PreLatentNorm
from ..latent.adversarial import AdversarialLatentLayer
from ..objective import Objectives
from ..head import setup_head
from copy import deepcopy


def get_logits( image_features, text_features, logit_scale):
    
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logit_scale * text_features @ image_features.T
    return logits_per_image, logits_per_text

def cal_clip_loss(image_features, text_features):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    device = image_features.device
    logits_per_image, logits_per_text = get_logits(image_features, text_features, logit_scale)
    labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
    total_loss = (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2
    return total_loss

class OmicsFormer(nn.Module):
    def __init__(self, gene_list, enc_mod, enc_hid, enc_layers, post_latent_dim, dec_mod, dec_hid, dec_layers,
                 out_dim, batch_num=0, dataset_num=0, platform_num=0, mask_type='input', model_dropout=0.1,
                 activation='gelu', norm='layernorm', enc_head=8, mask_node_rate=0.5,
                 mask_feature_rate=0.8, drop_node_rate=0., max_batch_size=2000, cat_dim=None, conti_dim=None,
                 pe_type=None, cat_pe=True,
                 gene_emb=None, latent_mod='vae', w_li=1., w_en=1., w_ce=1.,
                 head_type=None, dsbn=False, ecs=False, dar=False, input_covariate=False,
                 num_clusters=16, dae=True,cell_num=10,head_num=4, lamda=0.5, mask_beta=False,modality_list=['RNA'],split=False,dependence=False,topk=None,contrastive_method='clip',cluster=False,downstream=False,add_cell=0,pretrain_mode=False,**kwargs):
        super(OmicsFormer, self).__init__()
        '''
        pe_type: sin -> None
        '''

        self.embedder = OmicsEmbeddingLayer(gene_list, enc_hid, norm, activation, model_dropout,
                                            pe_type, cat_pe, gene_emb, inject_covariate=input_covariate, batch_num=batch_num,modality_list=modality_list)
        self.contrastive_method=contrastive_method
        
        self.downstream=downstream
        self.mask_type = mask_type
        self.cluster=cluster
        if topk==None:topk=cell_num
        if topk>cell_num:topk=cell_num
        self.top_k=topk
        if mask_node_rate > 0 and mask_feature_rate > 0:
            if mask_type == 'input':
                self.mask_model = MaskBuilder(mask_node_rate, mask_feature_rate, drop_node_rate, max_batch_size, mask_beta)
            elif mask_type == 'hidden':
                self.mask_model = HiddenMaskBuilder(mask_node_rate, mask_feature_rate, drop_node_rate, max_batch_size)
            else:
                raise NotImplementedError(f"Only support mask_type in ['input', 'hidden'], but got {mask_type}")
        else:
            self.mask_model = NullMaskBuilder(drop_node_rate, max_batch_size)
        
        self.modality_dict_enc={}
        for m in modality_list:
            self.modality_dict_enc[m]=setup_encoder(enc_mod, enc_hid, enc_layers, model_dropout, activation, norm, enc_head)
        self.modality_dict_enc=nn.ModuleDict(self.modality_dict_enc)

        self.latent = LatentModel()
        self.latent_mod = latent_mod
        if latent_mod=='vae':
            self.latent.add_layer(type='vae', enc_hid=enc_hid, latent_dim=post_latent_dim)
        elif latent_mod=='memory':
            post_latent_dim=enc_hid
            self.latent.add_layer(type='memory',cell_num=cell_num,head_num=head_num,hid_dim=enc_hid,out_dim=post_latent_dim,modality_dicts=modality_list,add_cell=add_cell,dependence=dependence,topk=topk,pretrain_mode=pretrain_mode)
        
        else:
            raise NotImplementedError(f'Latent mod "{latent_mod}" is not implemented.')
        
        
        if latent_mod is not None:
            if dar:
                self.latent.add_layer(type='adversarial', input_dims=np.arange(post_latent_dim), label_key='batch',
                                      discriminator_hidden=64, disc_lr=1e-3,
                                      target_classes=batch_num)
            if ecs:
                self.latent.add_layer(type='ecs')

        self.head_type = head_type

        dec_in_dim=post_latent_dim
        self.scgpt_transfer=nn.Linear(512,256)
        if head_type is not None:
            self.head = setup_head(head_type, dec_in_dim, dec_hid, out_dim, dec_layers,
                                   model_dropout, norm, batch_num=batch_num)
        else:
            self.modality_dict_dec={}
            for idx,m1 in enumerate(modality_list):
                for m2 in modality_list:
                    self.modality_dict_dec[m1+'_'+m2]=setup_decoder(dec_mod, dec_in_dim, dec_hid, out_dim[m2], dec_layers,
                                                model_dropout, norm, batch_num=batch_num, dataset_num=dataset_num, platform_num=platform_num)
            self.modality_dict_dec=nn.ModuleDict(self.modality_dict_dec)

            if 'nb' in dec_mod:
                self.objective = Objectives([{'type': 'nb', 'dae': dae}])
            else:
                self.objective = Objectives([{'type': 'recon'}])

        if dsbn:
            self.pre_latent_norm = PreLatentNorm('dsbn', enc_hid, dataset_num)
        else:
            self.pre_latent_norm = PreLatentNorm('ln', enc_hid)
        

    def forward(self, x_dict_list, input_gene_list=None, d_iter=False,infer=False,alpha=1,input_m=['RNA'],output_m=['protein'],saved_embs='',beta=30):
        types=self.latent_mod
        x_dict_enc={}
        value_dict={}
        cell_embs={}
        for m in x_dict_list.keys():
            x_dict=x_dict_list[m]
            if x_dict['scgpt_embs'] is None:
                # print('no scgpt')
                if self.mask_type == 'input':
                    x_dict = self.mask_model.apply_mask(x_dict)
                x_dict['h'] = self.embedder(x_dict, input_gene_list[m],modality=m,saved_embs=saved_embs)
                if self.mask_type == 'hidden':
                    x_dict = self.mask_model.apply_mask(x_dict)
                x_dict['h'] = self.modality_dict_enc[m](x_dict)['hidden']
                x_dict['h'] = self.pre_latent_norm(x_dict)
            else:
                x_dict['h']=x_dict['scgpt_embs']
                parameters = list(self.parameters())
                device = parameters[0].device
                x_dict['h']=x_dict['h'].to(device)
                x_dict['h']=F.gelu(self.scgpt_transfer(x_dict['h']))
                x_dict['h'] = self.pre_latent_norm(x_dict)
            if len(saved_embs)!=0:
                torch.save(x_dict,saved_embs+m+'cell_embs.pt')
            embs={}
            for key in x_dict.keys():
                if torch.is_tensor(x_dict[key]): 
                    embs[key]=x_dict[key].clone().detach()
            if types=='memory':
                output, latent_loss,c_loss = self.latent(x_dict,m,types=types)
                cell_embs[m]=output
            else:
                x_dict['h'], latent_loss= self.latent(x_dict,m,types=types)

            value_dict[m]=x_dict['h']
            x_dict_enc[m]=x_dict
        
        if types=='memory':
            aggregation_cell_embs=[]
            agg_cells=[]
            for m in input_m:
                aggregation_cell_embs.append(cell_embs[m].unsqueeze(1))

            aggregation_cell_embs=torch.cat(aggregation_cell_embs,dim=1)

            latent_cell_embs=torch.mean(aggregation_cell_embs,dim=1)
            for m in input_m:
                v1=x_dict_enc[m]['h']
                v2=self.cell_norm(latent_cell_embs)

                x_dict_enc[m]['h']=(1-alpha)*v1+alpha*v2
                
            if self.contrastive_method=='clip':
                latent_loss=cal_clip_loss(value_dict['RNA'],value_dict['protein'])*1e-2
            elif self.contrastive_method=='no':
                latent_loss=0
            if self.cluster:
                latent_loss+=c_loss
        if self.downstream:
            latent_loss=0

        if d_iter:
            return self.latent.d_train(x_dict)
        else:
            if self.head_type is not None:
                out_dict_modality={}
                
                if self.head_type=='modalitymatch':
                    x_dicts={}
                    for m in x_dict_list.keys():
                        x_dicts[m]=x_dict_enc[m]
                    out_dict, loss = self.head(x_dicts)
                else:
                    for m in x_dict_list.keys():
                        x_dict=x_dict_enc[m]
                    out_dict, loss = self.head(x_dict)
                out_dict['latent_loss'] = latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss
                out_dict['target_loss'] = loss.item()
                out_dict_modality=out_dict

            else:
                
                loss=0
                loss_list={}
                out_dict_modality={}
                for m1 in input_m:
                    for m2 in output_m:
                        m=m1+'_'+m2
                        x_dict_m1=x_dict_enc[m1]
                        x_dict_m2=x_dict_enc[m2]

                        out_dict = self.modality_dict_dec[m1+'_'+m2](x_dict_m1)
                        # print(out_dict['recon'].shape)
                        out_dict['pred']=out_dict['recon']
                        out_dict['label']=x_dict_m2['label']
                        if infer==False:
                            if m2=='protein':
                                loss +=beta*( latent_loss + self.objective(out_dict, x_dict_m2)) # / 1e2
                            else:
                                loss += latent_loss + self.objective(out_dict, x_dict_m2) # / 1e2
                            # loss_list[m]=latent_loss + self.objective(out_dict, x_dict_m2)
                            out_dict['latent_loss'] = latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss
                            out_dict['target_loss'] = loss.item() - out_dict['latent_loss']
                        out_dict_modality[m]=out_dict
                    
                out_dict_modality['latent_loss']=0
                out_dict_modality['target_loss']=0
                if infer==False:
                    for m1 in input_m:
                        for m2 in output_m:
                            m=m1+'_'+m2
                            out_dict_modality['latent_loss']+=out_dict_modality[m]['latent_loss']
                            out_dict_modality['target_loss']+=out_dict_modality[m]['target_loss']
                return out_dict_modality, loss,loss_list
                
            return out_dict_modality, loss

    def nondisc_parameters(self):
        other_params = []
        for pname, p in self.named_parameters():
            if 'discriminator' not in pname:
                other_params += [p]
            else:
                print(pname)
        return other_params
    def add_cells(self,num):
        self.latent.add_cells(num)
