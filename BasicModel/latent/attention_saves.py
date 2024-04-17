import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, scale,head_num):
        super().__init__()

        self.scale = scale
        self.head_num=head_num
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v,topk=10, mask=None):
        # print(q.shape,k.shape)
        u = torch.matmul(q, k.transpose(0, 1)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -torch.inf) # 3.Mask
        attn = self.softmax(u) # 4.Softmax
        output = torch.matmul(attn, v) # 5.Output
        top_k=torch.topk(u,k=topk*self.head_num,dim=1)
        selected_value=v[top_k[1],:]
        selected_value=selected_value.mean(1)

        return attn, output,selected_value
# attention = ScaledDotProductAttention(scale=np.power(128, 0.5))

# Q=torch.rand(32,10,128)
# K=torch.rand(64,10,128)
# V=torch.rand(64,10,128)
# # K=K.transpose(-1,0)
# print(K.shape)
# K=K.view(-1,128)
# Q=Q.view(-1,128)
# V=V.view(-1,128)
# attn, output = attention(Q,K,V )
# print(attn.shape,output.shape)

# class ClusterAssignment(nn.Module):
#     def __init__(
#         self,
#         cluster_number: int,
#         embedding_dimension: int,
#         alpha: float = 1.0,
#         cluster_centers: Optional[torch.Tensor] = None,
#     ) -> None:
#         """
#         Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
#         where the Student's t-distribution is used measure similarity between feature vector and each
#         cluster centroid.

#         :param cluster_number: number of clusters
#         :param embedding_dimension: embedding dimension of feature vectors
#         :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
#         :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
#         """
#         super(ClusterAssignment, self).__init__()
#         self.embedding_dimension = embedding_dimension
#         self.cluster_number = cluster_number
#         self.alpha = alpha
#         if cluster_centers is None:
#             initial_cluster_centers = torch.zeros(
#                 self.cluster_number, self.embedding_dimension, dtype=torch.float
#             )
#             nn.init.xavier_uniform_(initial_cluster_centers)
#         else:
#             initial_cluster_centers = cluster_centers
#         self.cluster_centers = Parameter(initial_cluster_centers)

#     def forward(self, batch: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
#         for each cluster.

#         :param batch: FloatTensor of [batch size, embedding dimension]
#         :return: FloatTensor [batch size, number of clusters]
#         """
#         norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
#         numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
#         power = float(self.alpha + 1) / 2
#         numerator = numerator ** power
#         return numerator / torch.sum(numerator, dim=1, keepdim=True)


class PreCells(nn.Module):
    def __init__(self,cell_num,head_num,hid_dim,out_dim,modality_dicts,dependence,**kwargs):
        super(PreCells,self).__init__()
        self.out_dim=out_dim
        self.head_num=head_num
        self.hid_dim=hid_dim
        self.cell_num=cell_num
        print('cell_num',cell_num)

        self.values=nn.Linear(self.hid_dim,self.head_num*self.hid_dim)
        self.keys=nn.Linear(self.hid_dim,self.head_num*self.hid_dim)
        if dependence:
            self.values_dict={}
            for m in modality_dicts:
                self.values_dict[m]=nn.Linear(self.hid_dim,self.head_num*self.hid_dim)
            self.values=nn.ModuleDict(self.values_dict)
            self.key_dict={}
            for m in modality_dicts:
                self.key_dict[m]=nn.Linear(self.hid_dim,self.head_num*self.hid_dim)
            self.keys=nn.ModuleDict(self.key_dict)
        else:
            self.values=nn.Linear(self.hid_dim,self.head_num*self.hid_dim)
            self.keys=nn.Linear(self.hid_dim,self.head_num*self.hid_dim)
        self.dependence=dependence


        self.tmp=nn.Linear(self.hid_dim,self.out_dim)

        self.fc_out=nn.Linear(self.hid_dim*head_num,out_dim)
        self.memory_cell = nn.Parameter(torch.empty((self.cell_num, self.head_num,self.hid_dim)))
        # nn.init.kaiming_uniform_(self.memory_cell, a=math.sqrt(5))
        nn.init.uniform_(self.memory_cell,-1,1)
        self.memory_cell.requires_grad_()
    
    def forward(self,x_dict,m):
        k=F.relu(self.keys(self.memory_cell).view(-1,self.hid_dim))
        return k,k


class _general_module(nn.Module):
    def __init__(self,cell_num,head_num,hid_dim,out_dim,modality_dicts,dependence,add_cell=0,**kwargs):
        super(_general_module,self).__init__()
        if dependence:
            self.queries_dicts={}
            for m in modality_dicts:
                self.queries_dicts[m]=nn.Linear(hid_dim,head_num*hid_dim)
            self.queries=nn.ModuleDict(self.queries_dicts)

        else:
            self.queries=nn.Linear(hid_dim,head_num*hid_dim)
        
        self.fc_out=nn.Linear(hid_dim*head_num,hid_dim)



class External_Attention(nn.Module):
    def __init__(self,cell_num,head_num,hid_dim,out_dim,modality_dicts,dependence,add_cell=0,topk=10,pretrain_mode=False,**kwargs):
        super(External_Attention,self).__init__()
        head_num=1
        self.out_dim=out_dim
        self.head_num=head_num
        self.hid_dim=hid_dim
        self.cell_num=cell_num
        self.pretrain_mode=pretrain_mode
        # print('cell_num',cell_num)
        self.alpha=1.0
        self.topk=topk
        self.dependence=dependence
        self.add_cell=add_cell
        self.pretrained_cells=PreCells(cell_num,head_num,hid_dim,out_dim,modality_dicts,dependence)
        if add_cell>0:
            self.added_cells=PreCells(add_cell,head_num,hid_dim,out_dim,modality_dicts,dependence)

        self.general_module=_general_module(cell_num,head_num,hid_dim,out_dim,modality_dicts,dependence,add_cell)
        self.queries=self.general_module.queries
        self.fc_out=self.general_module.fc_out
        self.memory_cell =self.pretrained_cells.memory_cell # nn.Parameter(torch.empty((self.cell_num, self.head_num,self.hid_dim)))
        self.attention = ScaledDotProductAttention(scale=np.power(128, 0.5),head_num=head_num)
        self.loss_function = nn.KLDivLoss(size_average=False)
        self.outputs=nn.Linear(hid_dim*2,hid_dim)

    def forward(self,x_dict,m):
        z=x_dict['h']
        batch_num=z.shape[0]
        # z=z.to(self.queries.device)
        # parameters = list(self.parameters())
        # device = parameters[0].device
        # z=z.to(device)
        q=F.relu(self.queries(z).view(-1,self.hid_dim))

        m_k,m_v=self.pretrained_cells(x_dict,m)

        if self.add_cell>0:
            a_k,a_v=self.added_cells(x_dict,m)
            k=torch.cat((m_k,a_k),dim=0)
            v=torch.cat((m_v,a_v),dim=0)
        else:
            k=m_k
            v=m_v
        values, output,selected_top = self.attention(q,k,v,topk=self.topk )
        output=output.view(batch_num,-1)
        selected_top=selected_top.view(batch_num,-1)
        if self.pretrain_mode:
            output=F.relu(self.fc_out(output))
        else:
            output=F.relu(self.fc_out(selected_top))
        c_loss=0
        return output,values,c_loss
    
    def multi_view_contrastive(self):
        cell_nums=self.memory_cell.shape[0]
        loss_sum=0
        for c in range(cell_nums):
            sample_loss=0
            for n_c in range(cell_nums):
                if n_c!=c:
                    cos=F.cosine_similarity(self.memory_cell[c,:,:].view(1,-1),self.memory_cell[n_c,:,:].view(-1,1),dim=1)
                    sample_loss+=torch.exp(cos)
            sample_loss=-torch.log(np.exp(1.0)/sample_loss)
            loss_sum+=sample_loss
        return loss_sum


    def target_distribution(self,batch):
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def clustering_loss(self,batch):
        # term=batch - self.memory_cell
        feature_num=self.memory_cell.shape[-1]
        # print(batch.shape,self.memory_cell.shape)
        norm_squared = torch.sum((batch.unsqueeze(1)-self.memory_cell.view(-1,feature_num))**2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    
    def update_infor(self):
        self.take_out_memory=torch.detach(self.memory_cell)
    def add_new_cells(self,new_cell_num):
        # self.tmp_new = torch.zeros(
        #     new_cell_num, self.head_num,self.hid_dim, dtype=torch.float
        # )
        # nn.init.xavier_uniform_(self.tmp_new)
        self.tmp_new = nn.Parameter(torch.empty((new_cell_num, self.head_num,self.hid_dim)))
        nn.init.kaiming_uniform_(self.tmp_new, a=math.sqrt(5))
        # self.tmp_new.grad(True)

        self.memory_cell=torch.cat((self.memory_cell,self.tmp_new))



    

