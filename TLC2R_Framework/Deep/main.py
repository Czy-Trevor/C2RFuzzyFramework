#tlib Reference: https://github.com/thuml/Transfer-Learning-Library
import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os
import torch.nn as nn
from torch.utils import data
from Model import Model
from tllib.alignment.jan import JointMultipleKernelMaximumMeanDiscrepancy
import random


def train_epoch(epoch, model, dataloaders, optimizer,weight):
    
    cos_metric = nn.CosineSimilarity(dim=-1, eps=1e-08)

    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)

    for i in range(1, num_iter):
        data_source, label_source = next(iter_source)
        data_target, _ = next(iter_target)
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target = data_target.cuda()

        optimizer.zero_grad()
        label_source_pred, loss_lmmd,_,_= model(
            data_source, data_target, label_source)
        cos_sim = torch.mean(cos_metric(label_source_pred.squeeze(), label_source))
        loss_regression = F.mse_loss(label_source_pred.squeeze(), label_source)

        loss_trans =  loss_lmmd * weight
        loss_regression_c = loss_regression
        loss = loss_trans + loss_regression_c

        loss.backward()
        optimizer.step()



def cc(tensor1, tensor2):
    total_sum = 0

    # Iterate through the pairs of true and pred
    mean1 = tensor1.mean()
    mean2 = tensor2.mean()

    std1 = tensor1.std()
    std2 = tensor2.std()

    norm_tensor1 = (tensor1 - mean1) / std1
    norm_tensor2 = (tensor2 - mean2) / std2

    covariance = (norm_tensor1 * norm_tensor2).mean()
    pearson_correlation = covariance.item()

    # Compute the final correlation coefficient
    return pearson_correlation

def test(model, dataloader):
    model.eval()
    test_loss = 0
    pcc = 0
    nan_ind = 0
    pred_list = []
    true_list = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            pred_list.append(pred)
            true_list.append(target)
            # sum up batch loss
        pred_list = torch.cat(pred_list)
        true_list = torch.cat(true_list)
        test_loss = F.mse_loss(pred_list, true_list).item()
        pcc = cc(pred_list, true_list)
        rmse = np.sqrt(test_loss)

        print(
            f'Average loss: {test_loss:.4f}, RMSE: {rmse:.4f},CC:{pcc:.4f}')
    return rmse,pcc


def get_args():

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dim_input', type=int,
                        help='Dimension of input vector', default=34)
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=32)
    parser.add_argument('--test_batch_size', type=float,
                        help='batch size', default=256)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=40)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.001, 0.001])
    parser.add_argument('--seed', type=int,
                        help='Seed', default = 2024)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--num_hidden', type=int,
                        help = 'num_hidden of MLP,except cls and feature exactration' ,default=1)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='1')
    parser.add_argument('--num_fs',type = int,
                         help = 'num of fuzzy set',default= 3)
    parser.add_argument('--MMD',type = str,
                         help = 'MMD type',default= 'JANR')
    parser.add_argument('--linear', default=True, action='store_true',
                        help='whether use the linear version')
    parser.add_argument('--val', default=False, action='store_true',
                        help='validation set')
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    return args

if __name__ == '__main__':
    dataset_list = ['PVT','DRIVING','SEED-VIG']
    weight_list = [0.1]
    for weight in weight_list:
        print(f'weight = {weight}')
        for dataset in dataset_list:
            seed_rmse = []
            seed_cc = []
            print(f"dataset = {dataset}")
            for SEED in range(2024,2029):
                args = get_args()
                print(f'SEED = {SEED}')
                np.random.seed(SEED)
                random.seed(SEED)
                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

                jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(linear=args.linear).cuda()

                if dataset == 'PVT':
                    loaded_x = np.load('/mnt/data2/zycui/DATA/PVTpsd.npz')
                    loaded_y = np.load('/mnt/data2/zycui/DATA/PVTlabel.npz')

                elif dataset == 'SEED-VIG':
                    loaded_x = np.load('/mnt/data2/zycui/DATA/SEEDX_db_0norm.npz')
                    loaded_y = np.load('/mnt/data2/zycui/DATA/SEEDY_db_0norm.npz')

                elif dataset == 'DRIVING':
                    loaded_x = np.load('/mnt/data2/zycui/DATA/X_list.npz')
                    loaded_y = np.load('/mnt/data2/zycui/DATA/Y_list.npz')

                else:
                    print('DATA error')

                data_ = [loaded_x[key] for key in loaded_x.files]
                labels_ = [loaded_y[key].reshape(-1, 1) for key in loaded_y.files]
                rmse_all = []
                cc_all = []
                dim_imput = data_[0].shape[1]
                for target in range(len(data_)):
                    if args.val == False:
                        tarx, tary = data_[target], labels_[target]
                        tarx_tensor = torch.from_numpy(tarx).cuda().float()
                        tary_tensor = torch.from_numpy(tary).cuda().float()
                        tar_dataset = data.TensorDataset(tarx_tensor, tary_tensor)
                        tar_train_dataloaders = data.DataLoader(tar_dataset, args.batch_size, shuffle=True, drop_last=True)
                        tar_test_dataloaders = data.DataLoader(tar_dataset, args.test_batch_size, shuffle=False, drop_last=False)
                        sourx = np.empty((0, dim_imput))
                        soury = np.empty((0, 1))

                        for sour in range(len(data_)):
                            if sour!= target:
                                a = data_[sour]
                                sourx = np.vstack((sourx,data_[sour]))
                                soury = np.vstack((soury,labels_[sour]))
                    else:
                        data_train = data_[0:target] + data_[target + 1:]
                        label_train = labels_[0:target] + labels_[target + 1:]
                        random_len = 1
                        val_indices = random.sample(range(len(data_train)), random_len)
                        train_data = []
                        train_label = []
                        val_data = []
                        val_label = []
                        for i, (sample, label) in enumerate(zip(data_train, label_train)):
                            if i in val_indices:
                                val_data.append(sample)
                                val_label.append(label)
                            else:
                                train_data.append(sample)
                                train_label.append(label)
                        sourx = np.empty((0, dim_imput))
                        soury = np.empty((0, 1))
                        tarx_tensor = torch.from_numpy(val_data[0]).cuda().float()
                        tary_tensor = torch.from_numpy(val_label[0]).cuda().float()
                        tar_dataset = data.TensorDataset(tarx_tensor, tary_tensor)
                        tar_train_dataloaders = data.DataLoader(tar_dataset, args.batch_size, shuffle=True, drop_last=True)
                        tar_test_dataloaders = data.DataLoader(tar_dataset, args.test_batch_size, shuffle=False, drop_last=False)
                        for sour in range(len(train_data)):
                            sourx = np.vstack((sourx, train_data[sour]))
                            soury = np.vstack((soury, train_label[sour]))

                    X_tensor = torch.from_numpy(sourx.squeeze()).cuda().float()
                    y_tensor = torch.from_numpy(soury.squeeze()).cuda().float()
                    dataset = data.TensorDataset(X_tensor, y_tensor)
                    dataloaders = data.DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True)
                    model = Model(jmmd_loss,dim_input=dim_imput,num_fs=args.num_fs,num_hidden=args.num_hidden,MMD=args.MMD).cuda()

                    correct = 0
                    stop = 0

                    if args.num_hidden != 0 :
                        optimizer = torch.optim.SGD([
                            {'params': model.feature_layers.parameters()},
                            {'params': model.bottle.parameters(), 'lr': args.lr[1]},
                            {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
                        ], lr=args.lr[1], momentum=args.momentum)
                    else:
                        optimizer = torch.optim.SGD([
                            {'params': model.feature_layers.parameters()},
                            {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
                        ], lr=args.lr[1], momentum=args.momentum)

                    epoch_rmse = []
                    stop = 0
                    last_rmse = np.inf
                    last_cc = 0
                    for epoch in range(1, args.nepoch + 1):

                        jmmd_loss.train()
                        train_epoch(epoch, model, (dataloaders,tar_train_dataloaders,tar_test_dataloaders), optimizer,weight)
                        t_rmse,pcc = test(model, tar_test_dataloaders)
                        epoch_rmse.append(t_rmse)

                    model.eval()
                    jmmd_loss.eval()

                    rmse_all.append(t_rmse)
                    cc_all.append(pcc)
                    print(f"tar{target}-rmse = {t_rmse}cc = {pcc}")

                print(f"rmse_all = {rmse_all}")
                for i in rmse_all:
                    print(i)

                print(f"cc_all = {cc_all}")
                for i in cc_all:
                    print(i)
                print(f"AVE RMSE = {sum(rmse_all)/len(rmse_all)}")
                print(f"AVE CC = {sum(cc_all)/len(cc_all)}")
                seed_rmse.append(sum(rmse_all)/len(rmse_all))
                seed_cc.append(sum(cc_all)/len(cc_all))

            print(seed_rmse)
            print(seed_cc)
            print(sum(seed_rmse)/len(seed_rmse))
            print(sum(seed_cc) / len(seed_cc))
