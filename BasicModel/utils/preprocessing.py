# from muon import prot as pt
from copy import deepcopy
import scanpy as sc
import numpy as np



def clr(adata, inplace = True, axis = 0):
    """
    Apply the centered log ratio (CLR) transformation
    to normalize counts in adata.X.

    Args:
        data: AnnData object with protein expression counts.
        inplace: Whether to update adata.X inplace.
        axis: Axis across which CLR is performed.
    """

    if axis not in [0, 1]:
        raise ValueError("Invalid value for `axis` provided. Admissible options are `0` and `1`.")
    x = adata.X

    x.data = x.data/np.repeat(
        np.exp(np.log1p(x).sum(axis=axis).A / x.shape[axis]), x.getnnz(axis=axis)
    )
    np.log1p(x.data, out=x.data)

    adata.X = x

    return None if inplace else adata

def gene_preprocessing(adata):
    # X = torch.log1p(X/X.sum(1, keepdim=True)*1000)
    adata.layers['x']=deepcopy(adata.X)
    sc.pp.log1p(adata)
    # print(adata.X.data)
    return adata
def protein_preprocessing(adata):
    adata.layers={}
    adata.layers['x']=deepcopy(adata.X)
    clr(adata)
    # print(adata.X.data)
    return adata