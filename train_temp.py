# Copyright (c) 2022 Unnamed Network for ECG classification
# Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training process code for Unnamed Netowrk for ECG classification.
"""
import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import seaborn as sns
# import torch package
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, plot_confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset
# import assistant package
from tqdm import tqdm
import wandb

from Adam import Adam as newAdam
from dataset import data_prep
from loss_library import FocalLoss
# import the model build class and dataloader
from model import (ResBlock, ResNet_PTB,
                   SpectralConv1d)

# check device available
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device = 'cuda'
    else:
        print("Let's use 1 GPU!")
        device = 'cuda'
else:
    device = 'cpu'

# release the GPU
torch.cuda.empty_cache()


def setup_seed(seed):
    # paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def start_train(args, model, train_loader, test_loader, device):
    # learning rate decay and optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = newAdam(model.parameters(), lr=args.lr, weight_decay=2e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dec_step, gamma=args.lr_dec_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 700], gamma=0.1)

    # loss function
    criterion = nn.CrossEntropyLoss()

    print('Start training model...')
    print('---------------------------')
    def train(model, criterion, optimizer, train_loader, device):
        epoch_loss = 0.0
        model.train()
        for data in train_loader:
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.reshape(-1).to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        return model, optimizer, epoch_loss

    @torch.no_grad()
    def test(model, criterion, test_loader, device):
        model.eval()
        test_loss = 0 
        for data in test_loader:
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.reshape(-1).to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            
            test_loss += loss.item()

        return model, test_loss

    @torch.no_grad()
    def get_acc(model, test_loader):
        total_acc = 0
        # prob_a = []
        # target_label = []
        for data in test_loader:
            model.eval()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.reshape(-1).to(device)

            # proba from CNN
            outputs = model(inputs)
            # outputs = F.softmax(outputs, dim=1)

            final_outputs = torch.argmax(outputs, dim=1)

            train_acc = torch.sum(final_outputs.to(device) == labels)
            total_acc += train_acc


        final_acc = total_acc/(len(test_loader.dataset))

        return final_acc

    # def training_loop(model, criterion, optimizer, train_loader, test_loader, epochs, device, print_step=10):
    import copy
    train_loss_list, test_loss_list, test_acc_list = [], [], []
    low_freq_ratio_list, high_freq_ratio_list = [], []
    threshold_list = {}
    threshold_list['fft64'], threshold_list['fft128'], threshold_list['fft256'], threshold_list['fft512'] = [], [], [], []
    # threshold_list['fft32'], threshold_list['fft64'], threshold_list['fft128'], threshold_list['fft256'], threshold_list['fft512'] = [], [], [], [], []

    best_acc = 0
    best_model = None
    best_epoch = 0

    for epoch in tqdm(range(args.epochs)):  # loop over the dataset multiple epochs
        model, optimizer, epoch_loss = train(model, criterion, optimizer, train_loader, device)

        train_loss_list.append(epoch_loss)
        scheduler.step()

        with torch.no_grad():
            model, test_loss = test(model, criterion, test_loader, device)
            test_loss_list.append(test_loss)

        # compute ACC
        test_acc = get_acc(model, test_loader)
        test_acc_list.append(test_acc.cpu().detach().numpy())

        # collect ratio and adaptive threshold
        low_freq_ratio_list.append(model.low_ratio.cpu().detach().numpy())
        high_freq_ratio_list.append(model.high_ratio.cpu().detach().numpy())

        # threshold_list['fft32'].append(model.fft32.threshold.cpu().detach().numpy())
        threshold_list['fft64'].append(model.fft64.threshold.cpu().detach().numpy())
        threshold_list['fft128'].append(model.fft128.threshold.cpu().detach().numpy())
        threshold_list['fft256'].append(model.fft256.threshold.cpu().detach().numpy())
        threshold_list['fft512'].append(model.fft512.threshold.cpu().detach().numpy())

        # update best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        if epoch %args.print_step == 0:
            print(f'Epoch:{epoch}\t'
                f'Train Loss:{epoch_loss}\t'
                f'Val Loss:{test_loss}\t'
                f'Epoch Acc:{test_acc}')
        wandb.log({"acc": test_acc,
                    "loss": test_loss})
    print('Finished Training in {}, Best Accuracy is {} on {}th epoch'.format(args.epochs, best_acc, best_epoch))

    # save model
    if args.save_model:
        model_path = args.model_dir + 'LR=' + str(args.lr) + '-LRDecStep=' + str(args.lr_dec_step) + '-LRDecRate=' + str(args.lr_dec_rate) +\
        '-Batch=' + str(args.batch_size) + '-epoch=' + str(args.epochs) + '-' + args.task + '_' + str(args.task_num) + '-' + 'threshold_ratio=' +\
            str(args.threshold_ratio) + '-model'

        torch.save(best_model, model_path)

    return best_model, train_loss_list, test_loss_list, test_acc_list, low_freq_ratio_list, high_freq_ratio_list, threshold_list


### test function


@torch.no_grad()
def start_test(model, dataloader):
    print('Start testing model...')
    print('---------------------------')
    model.eval()
    label = []
    prob_a = []
    test_label = []

    for data in dataloader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels

        outputs = model(inputs)
        outputs_label = torch.argmax(outputs, dim=1)

        label.append(outputs_label)
        prob_a.append(outputs)
        test_label.append(labels)

    label = torch.cat(label).cpu().detach().numpy()
    prob_pred = torch.cat(prob_a)
    test_label = torch.cat(test_label).cpu().detach().numpy()

    y_pred = prob_pred.cpu().detach().numpy()

   
    report = (classification_report(test_label, label))
    report = classification_report(test_label, label, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df_path = path + '/' + 'report.csv'
    df.to_csv(df_path)
    
    return y_pred  


def plot_result(train_loss_list, test_loss_list, test_acc_list, low_freq_ratio_list, high_freq_ratio_list, threshold_dict):
    result = [train_loss_list, test_loss_list, test_acc_list, low_freq_ratio_list, high_freq_ratio_list, threshold_dict]
    path = args.model_dir + 'train_log.pkl'
    a_file = open(path, "wb")
    pickle.dump(result, a_file)
    a_file.close()

    plt.figure(figsize=(50, 15))
    plt.subplot(2,5,1)
    plt.plot(train_loss_list)
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('train loss')
    
    plt.subplot(2,5,2)
    plt.plot(test_loss_list)
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('test loss')

    plt.subplot(2,5,3)
    plt.plot(test_acc_list)
    plt.xlabel('epochs')
    plt.ylabel('ACC')
    plt.title('test acc')

    plt.subplot(2,5,4)
    plt.plot(low_freq_ratio_list)
    plt.xlabel('epochs')
    plt.ylabel('ratio')
    plt.title('low frequency ratio')

    plt.subplot(2,5,5)
    plt.plot(high_freq_ratio_list)
    plt.xlabel('epochs')
    plt.ylabel('ratio')
    plt.title('high frequency ratios')
    
    # plt.subplot(2,5,6)
    # plt.plot(threshold_dict['fft32'])
    # plt.yscale('log')
    # plt.xlabel('epochs')
    # plt.ylabel('threshold')
    # plt.title('FFT32 threshold')

    plt.subplot(2,5,7)
    plt.plot(threshold_dict['fft64'])
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('threshold')
    plt.title('FFT64 threshold')

    plt.subplot(2,5,8)
    plt.plot(threshold_dict['fft128'])
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('threshold')
    plt.title('FFT128 threshold')

    plt.subplot(2,5,9)
    plt.plot(threshold_dict['fft256'])
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('threshold')
    plt.title('FFT256 threshold')

    plt.subplot(2,5,10)
    plt.plot(threshold_dict['fft512'])
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('threshold')
    plt.title('FFT512 threshold')

    figpath = args.model_dir + str(args.threshold_ratio) + '-result-curve.png' 
    plt.savefig(figpath, dpi=300)
    print('10 figures plotted')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/superclass/shuffle-data')
    parser.add_argument('--task', type=str, default='superclass')
    parser.add_argument('--model_dir', type=str, default='superclass/')
    parser.add_argument('--seed', type=int, default=33)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--criterion", default="focalloss")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_dec_rate", type=float, default=0.1)
    parser.add_argument("--lr_dec_step", type=int, default=150)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--task_num', type=int, default=1)
    parser.add_argument('--threshold_ratio', type=float, default=0.2)
    args = parser.parse_args()

    if args.seed:
        setup_seed(args.seed)
  
    
    path = args.task + '/' + str(args.task_num)

    if not os.path.exists(path):
        print('create directory {} for save result!'.format(args.task_num))
        print('---------------------------')
        os.mkdir(path)
    else:
        print('directory {} existing for save result!'.format(args.task_num))

    args.model_dir = path + '/'  
    train_loader, test_loader, val_loader = data_prep(args.task, args.batch_size)
    print('Training set Prepared!')
    print('---------------------------')
# build the model
    
    if args.task == 'superclass':
        model = ResNet_PTB(ResBlock, SpectralConv1d, args.threshold_ratio, num_classes=5).to(device)
    elif args.task == 'subclass':
        model = ResNet_PTB(ResBlock, SpectralConv1d, args.threshold_ratio, num_classes=23).to(device)
    elif args.task == 'rthym':
        model = ResNet_PTB(ResBlock, SpectralConv1d, args.threshold_ratio, num_classes=12).to(device)
    elif args.task == 'form':
        model = ResNet_PTB(ResBlock, SpectralConv1d, args.threshold_ratio, num_classes=19).to(device)
    else:
        print('Incorrect task name, check --task !!!')
    
    wandb_name = args.task + '-' + str(args.threshold_ratio)
    wandb.init(project="tnnls", entity="cl522", name=wandb_name)

    wandb.config = {
    "task name": args.task,
    "task number": args.task_num,
    "initial threshold": args.threshold_ratio
    }
    model = model.to(device)



    # training
    best_model, train_loss_list, test_loss_list, test_acc_list,\
    low_freq_ratio_list, high_freq_ratio_list, threshold_dict =\
    start_train(args, model, train_loader, test_loader, device)
    
    # ploting
    plot_result(train_loss_list, test_loss_list, test_acc_list, low_freq_ratio_list, high_freq_ratio_list, threshold_dict)

    # testing
    y_pred = start_test(best_model, test_loader)
    result_path = path + '/y_pred'
    np.save(result_path, y_pred)

    
