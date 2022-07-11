#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import DGCNN_cls_encoder
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, DeepMILossNew
import sklearn.metrics as metrics
from sklearn.metrics.pairwise import cosine_similarity
import time 
# from torch.utils.tensorboard import SummaryWriter   
import datetime



# def _init_():
#     if not os.path.exists('checkpoints'):
#         os.makedirs('checkpoints')
#     if not os.path.exists('checkpoints/'+args.exp_name):
#         os.makedirs('checkpoints/'+args.exp_name)
#     if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
#         os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
#     os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
#     os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
#     os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
#     os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')



def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    ############# Try to load models #############
    if args.model == 'dgcnn':
        model = DGCNN_cls_encoder(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    ############# model parallel #############
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    ############# choose opt & scheduler #############
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    ############# init before use #############
    ClsLoss = cal_loss
    MILoss = DeepMILossNew().to(device)
    best_test_acc = 0
    
    ############# epoch start  #############    
    for epoch in range(args.epochs):
        
        start = time.time()

        ####################
        # Train
        ####################
        train_loss = 0.0
        train_clsloss = 0.0
        train_miloss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            #########################################################################################
            x_global, x_global_prime, x_local, x_local_prime, c, c_exp, x = model(data)
            logits = x 
            clsloss = ClsLoss(logits, label)
            lmiloss, gmiloss, tmiloss = MILoss(x_global, x_global_prime, x_local, x_local_prime, c_exp, c)
            miloss = tmiloss
            loss = 0.3*miloss + 0.7*clsloss
            #########################################################################################
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]

            count += batch_size
            # train_loss += loss.item() * batch_size
            # train_clsloss += clsloss.item() * batch_size
            # train_miloss += miloss.item() * batch_size
            train_loss += float(loss.item() * batch_size)
            train_clsloss += float(clsloss.item() * batch_size)
            train_miloss += float(miloss.item() * batch_size)

            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        
        scheduler.step()

        end = time.time()
        t = end - start

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_avg_per_class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, training time: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 train_acc,
                                                                                 train_avg_per_class_acc, t)
        io.cprint(outstr)

        ## write down the train info
        writer.add_scalar('Training_time', t, epoch)
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Train_Clsloss', train_clsloss, epoch)
        writer.add_scalar('Train_MIloss', train_miloss, epoch)
        writer.add_scalar('Train_acc', train_acc, epoch)
        writer.add_scalar('Train_avg_per_class_acc', train_avg_per_class_acc, epoch)

        ####################
        # Test
        ####################
        test_loss = 0.0
        test_clsloss = 0.0
        test_miloss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            #########################################################################################
            x_global, x_global_prime, x_local, x_local_prime, c, c_exp, x = model(data)
            logits = x 
            clsloss = ClsLoss(logits, label)
            lmiloss, gmiloss, tmiloss = MILoss(x_global, x_global_prime, x_local, x_local_prime, c_exp, c)
            miloss = tmiloss
            loss = 0.35*miloss + 0.65*clsloss
            #########################################################################################
            preds = logits.max(dim=1)[1]
            count += batch_size
            # test_loss += loss.item() * batch_size
            # test_clsloss += clsloss.item() * batch_size
            # test_miloss += miloss.item() * batch_size
            test_loss += float(loss.item() * batch_size)
            test_clsloss += float(clsloss.item() * batch_size)
            test_miloss += float(miloss.item() * batch_size)

            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              test_avg_per_class_acc)
        io.cprint(outstr)

        ## write down the test info
        writer.add_scalar('Test_loss', test_loss, epoch)
        writer.add_scalar('Test_Clsloss', test_clsloss, epoch)
        writer.add_scalar('Test_MIloss', test_miloss, epoch)
        writer.add_scalar('Test_acc', test_acc, epoch)
        writer.add_scalar('Test_avg_per_class_acc', test_avg_per_class_acc, epoch)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN_cls_encoder(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        x_global, x_global_prime, x_local, x_local_prime, c, c_exp, x = model(data)
        logits = x
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


def retrieval(args):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN_cls_encoder(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()  

    Label = []
    shapecode = []
    Data = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        x_global, x_global_prime, x_local, x_local_prime, c, c_exp, x = model(data)
        # c -- shape code
        Label.append(label.cpu().numpy())
        shapecode.append(c.detach().cpu().numpy())
        Data.append(data.detach().cpu().numpy())
    # print(data.size()) # 4 3 1024
    Data = np.concatenate(Data)
    # print(len(Data)) # 2468
    Label = np.concatenate(Label)
    # Label = np.array(np.concatenate(Label))
    shapecode = np.concatenate(shapecode)
    # print(len(Label))  # 2468
    # print(len(shapecode[1])) # 2468 64
    cosinedistance = cosine_similarity(shapecode) # distance matrix : dim n*n
    top10 = np.zeros((shapecode.shape[0], 10), dtype=np.int) # the index of the samples with highest similarity
    top10_label = np.zeros((shapecode.shape[0], 10), dtype=np.int) # the corresponding label of top10 samples
    ##### AP ####
    acc = 0
    for i in range(shapecode.shape[0]):
        top10[i] = np.argsort(cosinedistance[i])[-11:-1]
        top10_label[i] = Label[top10[i]]
        acc += (top10_label[i]==Label[i]).sum()
        print('Object:', i, 'Label:', Label[i], 'Retrieval result - Top10:', top10_label[i])
        # if i == 575:
        #     query01 = np.array(Data[575])
        #     print(query01.size())
        #     # top01 = np.array(Data[top10[0]])
        # if i == 574:
        #     query02 = np.array(Data[574])

        # if i == 356:
        #     query03 = np.array(Data[356])
        
        # if i == 344:
        #     query04 = np.array(Data[344])
                
    totacc = acc/shapecode.shape[0]/10
    print('the accurcy of shape retrieval is ', totacc)
    
    ##### mAP #####
    # num_classes = np.unique(Label)
    # acc_list = []
    # for num in num_classes:
    #     acc = 0
    #     num_class = 0
    #     for i in range(shapecode.shape[0]):
    #         if Label[i]==num:
    #             num_class += 1
    #             top10[i] = np.argsort(cosinedistance[i])[-11:-1]
    #             top10_label[i] = Label[top10[i]]
    #             acc += (top10_label[i]==Label[i]).sum()/10

    #     acc_list.append(acc/num_class)
    # acc = np.mean(acc_list)
    # print("mAP:", acc)
    # print("End retrievaling")



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    # parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
    #                     help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cuda', type=bool, default=True, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    # # Used to save tensornoard
    # os.makedirs('./log/', exist_ok=True)
    # logdir = os.path.join('./log/'+ args.exp_name)
    # writer = SummaryWriter(logdir)
    # ##################################

    # _init_()

    # io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    # io.cprint(str(args))
    torch.manual_seed(args.seed)
    print('Using GPU:' + str(torch.cuda.current_device()) + 'from' + str(torch.cuda.device_count()) + ' devices' )

    retrieval(args)
