import torch
import torch.nn as nn
import numpy as np
import random

class ZINBReconstructionLoss(nn.Module):
    """ZINB loss class."""

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, out_dict, x_dict, ridge_lambda = 0.0):
        """Forward propagation.
        Parameters
        ----------
        x :
            input features.
        mean :
            data mean.
        disp :
            data dispersion.
        pi :
            data dropout probability.
        scale_factor : torch.Tensor
            scale factor of mean.
        ridge_lambda : float optional
            ridge parameter.
        Returns
        -------
        result : float
            ZINB loss.
        """
        eps = 1e-10
        x = x_dict['x_seq'].to_dense()# [x_dict['input_mask']]
        # x = x_dict['x_seq'].index_select(0, x_dict['input_mask']).to_dense()
        mean = out_dict['mean']# [x_dict['input_mask']]
        disp = out_dict['disp']# [x_dict['input_mask']]
        pi = out_dict['pi']# [x_dict['input_mask']]
        # scale_factor = x_dict['scale_factor'][x_dict['input_mask']]
        # scale_factor = scale_factor.unsqueeze(-1)
        scale_factor=torch.div(x.sum(1),mean.sum(1))
        # print(x.shape,mean.shape,scale_factor.shape)
        scale_factor=scale_factor.unsqueeze(-1)
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result



class NBReconstructionLoss(nn.Module):
    """ZINB loss class."""

    def __init__(self, dae=True, **kwargs):
        super().__init__()
        self.dae = dae

    def __call__(self, out_dict, x_dict):
        """Forward propagation.
        Parameters
        ----------
        x :
            input features.
        mean :
            data mean.
        disp :
            data dispersion.
        ridge_lambda : float optional
            ridge parameter.
        Returns
        -------
        result : float
            ZINB loss.
        """
        eps = 1e-10
        # print(x_dict)
        y = x_dict['label'].to_dense()
        truth = y # [:, x_dict['gene_mask']]
        # print(out_dict['mean'].shape,len(x_dict['gene_mask']))
        mean = out_dict['mean']# [:, x_dict['gene_mask']]
        disp = out_dict['disp']# [:, x_dict['gene_mask']]
        # mean=mean[:, x_dict['gene_mask']]
        # disp=disp[:, x_dict['gene_mask']]
        # if self.dae and random.random()>0.5:
        #     truth = truth * x_dict['input_mask']
        #     # mean = out_dict['mean'] * x_dict['input_mask']
        #     # disp = out_dict['disp'] * x_dict['input_mask']
        #     # print(mean.shape,x_dict['input_mask'].shape)
        #     mean = mean * x_dict['input_mask']
        #     disp = disp * x_dict['input_mask']
        # print(x_dict['input_mask'])
        
        # truth = truth[x_dict['input_mask'].sum(1)>=0]
        # mean = mean[x_dict['input_mask'].sum(1)>=0]
        # disp = disp[x_dict['input_mask'].sum(1)>=0]
        # mean = mean / mean.sum(1, keepdim=True) * truth.sum(1, keepdim=True)
        t1 = torch.lgamma(disp + eps) + torch.lgamma(truth + 1.0) - torch.lgamma(truth + disp + eps)
        t2 = (disp + truth) * torch.log(1.0 + (mean / (disp + eps))) + (truth * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        return nb_final.sum(-1).mean()

        # if self.dae and random.random()>0.5:
        #     # result = (nb_final * (x_dict['input_mask'][x_dict['input_mask'].sum(1)>0][:, x_dict['gene_mask']])).sum(-1).mean()
        #     result = nb_final.sum(-1).mean() / (x_dict['input_mask'].float().mean())
        # else:
        #     result = nb_final.sum(-1).mean()

        # log = torch.log
        # lgamma = torch.lgamma
        # log_theta_mu_eps = log(disp + mean + eps)
        # result = -(
        #         disp * (log(disp + eps) - log_theta_mu_eps)
        #         + truth * (log(mean + eps) - log_theta_mu_eps)
        #         + lgamma(truth + disp)
        #         - lgamma(disp)
        #         - lgamma(truth + 1)
        # ).sum(-1).mean()

        # return result