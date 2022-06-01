import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributed as dist

import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def data_prep(task, batch_size):
    from torch.utils.data import DataLoader, Dataset

### -------------------------------
    # load data and target
    print('Data Loading start!')
    print('---------------------------')

    path = '/data/che/LL-work/PTB_XL_data/' + task + '/benchmarkdata/'

    # for benchmark
    P_X_train, P_X_test, P_X_val = path+'x_train.npy', path+'x_test.npy', path+'x_val.npy'
    P_y_train, P_y_test, P_y_val = path+'y_train.npy', path+'y_test.npy', path+'y_val.npy'
    ###

    X_train = np.load(P_X_train, allow_pickle=True)
    X_test = np.load(P_X_test, allow_pickle=True)
    X_val = np.load(P_X_val, allow_pickle=True)

    # for benchmark
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    X_val = np.swapaxes(X_val, 1, 2)
    ### 

    y_train = np.load(P_y_train, allow_pickle=True)
    y_test = np.load(P_y_test, allow_pickle=True)
    y_val = np.load(P_y_val, allow_pickle=True)

    # for benchmark
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    y_val = np.argmax(y_val, axis=1)
    ###

    print('Data Loading completed!')
    print('---------------------------')
### -------------------------------
   # define tensor type converter and dataset, dataloader
    def convert(label):
        converted_label = label.astype(np.float32).reshape(-1,1)
        return converted_label

    class dataset(Dataset):

        def __init__(self, signal, label):

            self.data = signal
            self.label = label
        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            ecg = torch.from_numpy(self.data[idx].astype(np.float32))
            target = torch.from_numpy(self.label[idx]).type(torch.long)

            sample = (ecg, target)
            return sample
### --------------------------------
    # convert target to tensor

    converted_y_train = convert(y_train)
    converted_y_test = convert(y_test)
    converted_y_val = convert(y_val)

### -------------------------------
    # create dataset

    trainset = dataset(signal = X_train, label = converted_y_train)
    testset = dataset(signal = X_test, label = converted_y_test)
    valset = dataset(signal = X_val, label = converted_y_val)

### --------------------------------
    # create dataloader
    
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, pin_memory = True, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=2048, pin_memory = True, shuffle=False)
    val_loader = DataLoader(dataset=valset, batch_size=1024, pin_memory = True, shuffle=False)

    return train_loader, test_loader, val_loader

### --------------------------------
    ## version for distributed training

    # trainset = dataset(signal = X_train, label = converted_y_train, local_rank=local_rank)
    # testset = dataset(signal = X_test, label = converted_y_test, local_rank=local_rank)
    # valset = dataset(signal = X_val, label = converted_y_val, local_rank=local_rank)

    # testset_org = dataset(signal = X_test_org, label = converted_y_test_org, local_rank=local_rank)

    # # define sampler for each GPU -- for shuffle on each epoch
    # sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    # if parallel_type is 'None' or 'Dataparallel':
    #     # for data parallel and non-parallel

    #     train_loader = DataLoader(dataset=trainset, batch_size=batch_size, pin_memory = True, shuffle=True)
    #     test_loader = DataLoader(dataset=testset, batch_size=batch_size, pin_memory = True, shuffle=False)
    #     val_loader = DataLoader(dataset=valset, batch_size=1024, pin_memory = True, shuffle=False)

    #     test_loader_org = DataLoader(dataset=testset_org, batch_size=1024, pin_memory = True, shuffle=False)
    # else:
    #     # for model distributed training
        
    #     sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    #     train_loader = DataLoader(trainset,
    #                       batch_size=batch_size,
    #                       shuffle=False,
    #                       pin_memory=True,
    #                       drop_last=True,
    #                       sampler=sampler)
    #     test_loader = DataLoader(dataset=testset, batch_size=batch_size, pin_memory = True, shuffle=False)

    # return train_loader, test_loader, label_encoder, y_test, sampler