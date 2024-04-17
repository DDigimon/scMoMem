from torch import nn
from .autoencoders import VAELatentLayer
from .attention_saves import External_Attention
import torch
from ..utils import DSBNNorm

def create_latent_layer(**config) -> nn.Module:
    if config['type'] == 'vae':
        return VAELatentLayer(**config)
    elif config['type']=='memory':
        return External_Attention(**config)
    else:
        raise ValueError(f"Unrecognized latent model name: {config['type']}")

class PlaceholderLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.is_adversarial = False

    def forward(self, x_dict,m=''):
        return x_dict['h'], torch.tensor(0.).to(x_dict['h'].device),torch.tensor(0.).to(x_dict['h'].device)

class LatentModel(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.layers = nn.ModuleList([PlaceholderLayer()])
        self.alias_dict = {}
        self.config=configs
        if configs is not None:
            for c in configs:
                self.layers.append(create_latent_layer(**c))

    def forward(self, x_dict,m='',types='vae'):
        if types!='memory':
            total_loss = 0
            total_c_loss=0
            for layer in self.layers:
                t= layer(x_dict,m)
                output, loss,_=t
                total_loss += loss
            return output, total_loss
        else:
            output,attention_values,c_loss=self.layers[-1](x_dict,m)
            return output, attention_values,c_loss

    def add_layer(self, **config):
        if 'alias' in config:
            self.alias_dict[config['alias']] = len(self.layers)
        else:
            self.alias_dict[config['type']] = len(self.layers)
        self.layers.append(create_latent_layer(**config))

    def get_layer(self, alias):
        return self.layers[self.alias_dict[alias]]

    def d_train(self, x_dict):
        loss = 0
        for layer in self.layers:
            if layer.is_adversarial:
                loss += layer.d_iter(x_dict)
        return loss
    def add_cells(self,num):
        print(self.layers)
        self.layers[-1].add_new_cells(num)

class PreLatentNorm(nn.Module):
    def __init__(self, type='none', enc_hid=None, dataset_num=None):
        super().__init__()
        self.type = type
        if type not in ['none', 'dsbn', 'ln']:
            raise NotImplementedError(f'"{type}" type of pre latent norm is not implemented.')
        if type == 'dsbn':
            self.norm = DSBNNorm(enc_hid, dataset_num)
        elif type == 'ln':
            self.norm = nn.LayerNorm(enc_hid)

    def forward(self, xdict):
        if self.type == 'dsbn':
            return self.norm(xdict)
        elif self.type == 'ln':
            return self.norm(xdict['h'])
        else:
            return xdict['h']
