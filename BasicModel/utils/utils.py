from tqdm import tqdm
import numpy as np
import torch

def get_idxs(x_data,batch_labels,seed):
    train_idx=[]
    val_idx=[]
    test_idx=[]
    for batch in tqdm(range(batch_labels.max() + 1)):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(range(x_data[batch_labels == batch].shape[0]))
        tmp_x=x_data[batch_labels == batch]
        train_temp = torch.zeros([idx.shape[0]])
        val_temp = torch.zeros([idx.shape[0]])
        test_temp = torch.zeros([idx.shape[0]])
        train_samples=[]
        valid_samples=[]
        test_samples=[]
        for sampled_data_idx in range(tmp_x.shape[0]):
            if list(tmp_x.obs['split'])[sampled_data_idx]=='train':
                train_samples.append(sampled_data_idx)
            elif list(tmp_x.obs['split'])[sampled_data_idx]=='val':
                valid_samples.append(sampled_data_idx)
            elif list(tmp_x.obs['split'])[sampled_data_idx]=='test':
                test_samples.append(sampled_data_idx)
        train_temp[train_samples]=1
        val_temp[valid_samples]=1
        test_temp[test_samples]=1
        train_idx.append(train_temp.bool())
        val_idx.append(val_temp.bool())
        test_idx.append(test_temp.bool())
    return train_idx,val_idx,test_idx